import mimetypes
import shutil
from mimetypes import guess_type
from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger, PROJECT_ROOT
from app.import_process.agent.state import get_default_state

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.import_process.agent.main_graph import kb_import_app

app = FastAPI(title="import service",description="导入文件处理!")

# 初始化FastAPI应用实例
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus/KG入库)"
)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)

"""
 接口1: 返回import.html文件 
    url:     /import/html 
    method:  get
    参数:    无
    响应:    import.html (FileResponse)
"""
@app.get("/import/html")
def return_import_html():
    #1. 拼接文件地址
    html_file_obj = PROJECT_ROOT / "app" / "import_process" / "page" / "import.html"
    #2. 判断是否存在,404
    if not html_file_obj.exists():
        logger.error(f"本次查询没有找到对应的html文件,返回404异常!!")
        raise HTTPException(status_code=404,detail="没有找到对应的页面文件...")
    #3. 返回页面文件
    return FileResponse(
        path= html_file_obj, # 文件地址
        media_type= guess_type(html_file_obj.name)[0] # 文件的mimetype类型
    )

"""
执行import_graph
"""
def invoke_import_graph(task_id: str, local_dir: str, local_file_path: str):
   try:
       update_task_status(task_id, TASK_STATUS_PROCESSING)  # 任务开启状态 开始解析
       state = get_default_state()
       state["task_id"] = task_id
       state["local_dir"] = local_dir
       state["local_file_path"] = local_file_path
       # kb_import_app.stream(state)
       kb_import_app.invoke(state)
       update_task_status(task_id,TASK_STATUS_COMPLETED)  # 任务结束状态 解析成功
   except Exception as e:
       update_task_status(task_id,TASK_STATUS_FAILED)     # 任务失败状态 解析失败
       logger.exception(f"task_id={task_id}任务的导入流程发生异常信息!{str(e)}")

"""
接口二: 接收上传的文件列表 files = [] 
   url: /upload
   method: post
   参数: 请求体 文件列表 files 
   响应:
        {
            "code": 200,
            "message": f"文件描述",
            "task_ids": task_ids []  文件 -&gt; 一套任务 -&gt; task_id 
        }
  分析流程: 开启文件解析,触发import_process流程 
     1. 接收上传文件 (UploadFile)
     2. 存储上传的文件 /output/YYYYmmdd/task_id/烫金机.pdf
     3. 拼接state属性 [task_id,local_file_path,local_dir]
     4. 异步调用 import_graph_app . invoke(state) -&gt; 8个节点....
     5. 立即给前端返回结果,图片正在处理中...
"""

@app.post("/upload")
async def upload(backgroundtasks: BackgroundTasks, files:List[UploadFile] = File(...)):
    """
    # 1. 接收上传文件 (UploadFile)
    # 2. 存储上传的文件 /output/YYYYmmdd/task_id/烫金机.pdf
    :param backgroundtasks:
    :param files:
    :return:
    """
    task_ids = []
    local_dir_obj = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d")

    for file in files:
        # task_id == 文件 == 一套流程
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        local_dir_obj = local_dir_obj / task_id  # 完整local_dir

        # 有可能没有文件夹
        local_dir_obj.mkdir(parents=True, exist_ok=True)

        # file = UploadFile (filename 文件名  .file 文件数据)
        local_file_path_obj = local_dir_obj / file.filename
        # 保存文件
        with open(local_file_path_obj, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # 3.拼接state属性 [task_id,local_file_path,local_dir]
        # 4.异步调用 import_graph_app . invoke(state) -&gt; 8个节点....
        backgroundtasks.add_task(
            invoke_import_graph,
            task_id=task_id,
            local_dir=str(local_dir_obj),
            local_file_path=str(local_file_path_obj)
        )
        # 5. 立即给前端返回结果,图片正在处理中...
    return {
        "code": 200,
        "message": "文件已经上传成功,正在处理解析中....",
        "task_ids": task_ids
    }

"""
 接口三: 向后端查询任务状态接口 
"""
# --------------------------
# 核心接口：任务状态查询接口 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info

# --------------------------
# 服务启动入口 直接运行此脚本即可启动FastAPI服务，无需额外执行uvicorn命令
# --------------------------
if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8000端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8000  # 服务端口
    )