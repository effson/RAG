"""
   node_pdf_to_md
     参数： state [is_pdf_read_enabled = True | pdf_path = xxx.pdf | local_dir = output ]
     返回： state [md_path = 地址 | md_content = 内容 ]
     1. 日志和任务状态
     2. step_1_validate_paths路径校验
     3. step_2_upload_and_poll minerU的交互
     4. step_3_download_and_extract 下载和解压
     5. 日志和任务状态 return state
   step_1_validate_paths
     参数：state pdf_path = xxx.pdf | local_dir = output
     返回： pdf_path_obj Path  local_dir_obj Path
     1. 非空校验
     2. 文件校验 pdf_path_obj 没有抛异常 local_dir_obj 没有给与默认
     3. 返回完成可用的Path对象即可
   step_2_upload_and_poll
     参数：pdf对应Path  pdf_path_obj
     返回：str zip url地址
     1. 进行申请，获取要上传文件的地址
     2. 进行文件上传 session | requests.put
     3. 轮询获取返回结果 zip_url  （确定一个最大等待时间 1页pdf 1s 间隔时间3 错误码 200 -》 500能容忍）
     4. 返回地址即可
   step_3_download_and_extract
     参数：zip_url , out_dir_obj , 原文件名 path.stem
     返回：解压后的.md的str地址
     1. zip下载 get    output / stem_result.zip
     2. 检查解压的文件夹地址  output / stem
     3. 检查解压的文件夹进行防重复处理
     4. 进行解压 zipFile  extractall(解压的目标文件夹)
     5. 考虑文件名字 原文件件名 还是 full 还是其他
     6. 重命名处理
     7. 路径转成字符串 获取绝对路径最终返回即可！
"""

import os
import sys
import time
import requests
import zipfile
import shutil
from pathlib import Path

# 项目内部库
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config
from app.core.logger import logger, node_log, step_log  # 统一日志工具

# MinerU配置（缓存配置信息）
MINERU_BASE_URL = mineru_config.base_url
MINERU_API_TOKEN = mineru_config.api_key

@step_log("step_1_validate_paths")
def step_1_validate_paths(state: ImportGraphState) -> tuple[Path, Path]:
    """
    步骤1：路径校验与初始化
    校验PDF输入文件与输出目录的有效性，遵循「输入严格校验、输出自动修复」的鲁棒性设计原则：
    1. 校验PDF路径非空且文件真实存在，不存在则直接抛出异常（快速失败）
    2. 校验输出目录，为空则赋予默认值，不存在则自动创建（自动容错）
    3. 统一转换为Path对象处理，保证路径操作的规范性与跨平台兼容性

    :param state:
    :return:
    """
    # 1. 获取路径参数
    pdf_path = state.get("pdf_path", "").strip()
    local_dir = state.get("local_dir", "").strip()

    # 2. 参数非空校验
    if not pdf_path:
        raise ValueError("pdf_path 不能为空，请提供有效的PDF文件路径")
    if not local_dir:
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = str(local_dir)
        logger.warning(f"未指定输出目录，使用默认路径：{local_dir}")
    # 3. 统一转换为Path对象，标准化路径处理
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    # 4. 路径有效性校验（差异化处理：输入严格校验，输出自动修复）
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF文件不存在：{pdf_path_obj}，请检查文件路径是否正确")
    if not local_dir_obj.exists():
        logger.warning(f"输出目录不存在，自动创建：{local_dir_obj}")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj

@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj: Path, output_dir_obj: Path) -> str:
    """
    步骤 2：上传 PDF至 MinerU 并轮询解析任务状态
    核心流程：配置校验 → 获取上传链接 → 文件上传（含重试） → 任务轮询（直至完成/失败/超时）
    参数：pdf_path_obj-已校验的PDF Path对象；output_dir_obj-输出目录Path对象
    返回：解析结果ZIP包下载链接full_zip_url
    异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
    """
    # 1. 前期配置校验，拦截无效配置
    if not MINERU_BASE_URL or not MINERU_API_TOKEN:
        logger.error(f"minerU 配置错误。请检查 MINERU_BASE_URL 和 MINERU_API_TOKEN！")
        raise ValueError("MinerU服务地址(base_url)或API密钥(api_key)未配置，请检查配置文件！")

    url_get_upload = f"{mineru_config.base_url}/file-urls/batch"
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_TOKEN}"
    }

    data = {
        "files": [
            {"name": f"{pdf_path_obj.stem}"}
        ],
        "model_version": "vlm"
    }

    response = requests.post(url_get_upload, headers=request_headers, json=data)
    if response.status_code != 200:
        raise RuntimeError(f"请求MinerU服务失败，状态码：{response.status_code}，响应内容：{response.text}")

    resp_data = response.json()
    if resp_data["code"] != 0:
        logger.error(f"申请上传网络地址成功！但业务失败！错误码：{resp_data["code"]}，msg：{resp_data['msg']}")
        raise RuntimeError(f"MinerU接口返回失败，code：{resp_data['code']}，msg：{resp_data['msg']}")

    # 提取预签名 上传地址与任务批次ID
    file_upload_url = resp_data["data"]["file_urls"][0]
    batch_id = resp_data["data"]["batch_id"]

    #  读取PDF文件二进制内容
    file_data = pdf_path_obj.read_bytes()

    with requests.Session() as session:
        session.trust_env = False
        upload_response = session.put(file_upload_url, data=file_data)

        if upload_response.status_code != 200:
            logger.error(f"上传文件失败！错误码：{upload_response.status_code}")
            raise RuntimeError(f"上传文件失败！，错误码：{upload_response.status_code}")

    poll_url = f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}"
    timeout_seconds = 600  # 最大超时时间10分钟
    poll_interval = 3  # 轮询间隔3秒
    start_time = time.time()
    logger.debug("轮询 MinerU 解析结果...")
    while True:
        # 超时判断
        if time.time() - start_time > timeout_seconds:
            logger.error(f"MinerU 解析任务超时！请检查 MinerU 服务或文件大小")
            raise TimeoutError(f"MinerU 解析任务超时（> {timeout_seconds} s），请检查服务或文件大小")
        try:
            poll_response = session.get(poll_url, headers=request_headers, timeout=10)
        except Exception as e:
            logger.warning(f"轮询请求异常，将重试：{str(e)}")
            time.sleep(poll_interval)
            continue

        status_code = poll_response.status_code
        if status_code != 200:
            if 500 <= status_code < 600:
                logger.warning(f" MinerU 服务端异常, status_code: {status_code}，休眠后自动重试")
                time.sleep(poll_interval)
                continue
            else:
                logger.error(f"客户端异常! status_code: {status_code}，请检查 MinerU API_KEY与服务地址")
                raise RuntimeError(f"轮询请求失败，客户端异常, status_code: {status_code}，请检查API_KEY与服务地址")

        poll_response_dict = poll_response.json()
        code = poll_response_dict["code"]
        if code != 0:
            logger.error(f"MinerU轮询接口异常，code：{code}，msg：{poll_response_dict['msg']}")
            raise RuntimeError(f"MinerU轮询接口异常，code：{code}，msg：{poll_response_dict['msg']}")

        extract_results = poll_response_dict["data"]["extract_result"]
        if not extract_results:
            logger.debug("暂无解析结果，继续轮询...")
            time.sleep(poll_interval)
            continue

        # 根据任务状态执行对应逻辑
        task_state = extract_results[0]["state"]
        if task_state == "done":
            full_zip_url = extract_results[0]["full_zip_url"]
            if not full_zip_url:
                raise RuntimeError("MinerU解析完成，但未返回有效的ZIP下载链接")
            logger.info("PDF解析任务完成，准备下载结果包")
            return full_zip_url
        elif task_state == "failed":
            raise RuntimeError(f" MinerU 解析任务执行失败，err_msg:{extract_results[0]["err_msg"]}")
        else:
            logger.debug(f"MinerU 解析任务处理中，当前状态：{task_state}，继续轮询...")
            time.sleep(poll_interval)
            continue

def step_3_download_and_extract(zip_url: str, output_dir_obj: Path, pdf_stem: str) -> Path:
    """
    步骤 3：下载MinerU解析结果ZIP包并解压，提取目标MD文件（重命名统一规范）
    核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件（按优先级） → 重命名统一为PDF同名
    参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
    返回：最终MD文件的字符串格式绝对路径
    异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)
    """
    # 1.下载zip_url对应的资源
    response = requests.get(zip_url, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"ZIP包下载失败，状态码：{response.status_code}，响应内容：{response.text}")

    # 2.保存zip文件 文件保存和命名 output_dir_obj / 源文件名_result.zip
    zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"
    zip_save_path.write_bytes(response.content)

    # 3.清理旧目录并解压zip包
    # 定义要解压的目录地址
    extract_target_dir = output_dir_obj / pdf_stem
    if extract_target_dir.exists():
        # shutil.copy(源, 目标) 复制文件      shutil.move(源, 目标)   # 移动/重命名
        # extract_target_dir.unlink(missing_ok=True) 删除单个文件    extract_target_dir.rmdir()   删除【空文件夹】
        shutil.rmtree(extract_target_dir)

    # 确保输出目录存在,重新创建下
    extract_target_dir.mkdir(parents=True, exist_ok=True)

    shutil.unpack_archive(zip_save_path, extract_target_dir)
    # 利用zipFile进行解压!
    # with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
    #     zip_ref.extractall(extract_target_dir)

    # 4. 处理下md文件,统一姓名,并且返回md的字符串地址
    # 获取全部文件
    md_file_list = list(extract_target_dir.rglob("*.md"))
    if not md_file_list or len(md_file_list) == 0:
        logger.error(f"文件解压失败！{extract_target_dir} 下没有 md 文件")
        raise FileNotFoundError(f"{extract_target_dir} 下未找到PDF对应的MD文件")

    target_md_file = None
    # 先读取同名的
    for md_file in md_file_list:
        if md_file.stem == pdf_stem:
            target_md_file = md_file
            return target_md_file

    # 再读取叫full.md的
    for md_file in md_file_list:
        if md_file.name.lower() == "full.md":
            target_md_file = md_file
            break

    if not target_md_file:
        target_md_file = md_file_list[0]

    # 如果不是文件名需要重命名,方便后续处理
    # md文件名  二狗子.md  full.md  不知道.md
    # 统一改成  原文件名（stem）.md
    # 不是原名字的时候，我才重命名
    # if target_md_file.stem != pdf_stem:
    #     # 进行重命名
    #     # target_md_file.with_name(f"{stem}.md") 修改path对象 （不涉及文件操作） 返回结果是修改后path对象
    #     # target_md_file.rename(target_md_file.with_name(f"{stem}.md")) 修改磁盘中的文件名称（修改名称了） return 新的路径path
    #     target_md_file = target_md_file.rename(target_md_file.with_name(f"{pdf_stem}.md"))

        # 最终的md文件获取绝对路径，并且返回字符串类型
    final_md_str_path = target_md_file.rename(target_md_file.with_name(f"{pdf_stem}.md"))
    return final_md_str_path

@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    LangGraph工作流节点：PDF 转 MD 核心处理节点
    核心流程：路径校验 → MinerU 上传解析 → 结果下载解压 → 读取 MD内容并更新工作流状态
    参数：state-工作流状态对象，需包含pdf_path/local_dir/task_id
    返回：更新后的工作流状态，新增md_path/md_content

    核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据
    """

    # 1.注解日志和进行和完成任务状态处理
    add_running_task(state['task_id'], 'node_pdf_to_md')

    # 2. step_1_validate_paths 校验路径完整以及是否真实存在
    pdf_path_obj, output_dir_pbj = step_1_validate_paths(state)

    # 3. step_2_upload_and_poll
    zip_url = step_2_upload_and_poll(pdf_path_obj, output_dir_pbj)
    # 4. step_3_download_and_extract
    md_path = step_3_download_and_extract(zip_url, output_dir_pbj, pdf_path_obj.stem)

    # 5. 处理响应结果 state进行赋值处理
    state['md_path'] = str(md_path)
    md_content = md_path.read_text(encoding='utf-8')
    state['md_content'] = md_content

    add_done_task(state['task_id'], 'node_pdf_to_md')
    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "H3C MER系列路由器 用户手册-R0821-6W105-整本手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")