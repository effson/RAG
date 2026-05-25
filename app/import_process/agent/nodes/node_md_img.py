import os
import re
import mimetypes
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

from langchain_core.output_parsers import StrOutputParser
# MinIO相关依赖image_data
from minio import Minio
from minio.datatypes import Object
from minio.deleteobjects import DeleteObject

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger, node_log, step_log
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


# 步骤1：初始化MD核心数据，获取内容、文件路径、图片文件夹路径
@step_log("step_1_get_content")
def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    从全局状态中提取并初始化MD处理所需核心数据
    :param state: 导入流程全局状态对象
    :return: 三元组(MD文件内容, MD文件路径对象, 图片文件夹路径对象)
    :raise FileNotFoundError: 当状态中无有效MD文件路径时抛出
    """
    # 1.获取基本信息
    md_content = state['md_content']
    md_path = state['md_path']
    # 2. 非空校验
    if not md_path:
        logger.error(f"md_path核心参数为空,无法继续!!")
        raise ValueError("md_path核心参数为空,无法继续!!")
    # 3. md_path转成Path对象
    md_path_obj = Path(md_path)

    if not md_content:
        logger.warning("md_content为空,根据地址读取!")
        state["md_content"] = md_path_obj.read_text(encoding="utf-8")

    # 4. 图片文件夹
    images_path_obj = md_path_obj.parent / "images"
    return state["md_content"], md_path_obj, images_path_obj


def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

# 步骤2：扫描图片文件夹，筛选MD中实际引用的支持格式图片
@step_log("step_2_scan_images")
def step_2_scan_images(md_content: str, images_path_obj: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    作用: 找到图片的本身信息和图片的上下文信息
    入参: md_content ,  images_dir_obj
    出参: [("图片名.xx","图片地址",(上文 -> 100,下文 -> 100))]
    步骤:
        1. 获取图片文件夹中所有图片对象
        2. 循环遍历图片对象
        3. 定义对应的正则规则 ![](图片名)
        4. md中进行正在匹配 Match = re.search(md)
        5. Match对象获取图片的坐标信息 ( .span() .star() .end()  .group() )
        6. md中截取上文和下文 md_content [ star-100 : star]   [end , end+100]
        7. 拼接结果内容 (对象.name , str(对象) , (上文,下文))
    """
    # 列表: 接收整体结果
    image_context_list = []

    for image_file in images_path_obj.iterdir():
        image_name = image_file.name  # image_file 每个图片的Path对象
        if not is_supported_image(image_name):
            # 不是图片,无需处理
            logger.warning(f"{image_name}不是图片,无需处理,跳过!!")
            continue
        # 图片格式：![](images/ac26d5ab3a9f599eb2f58c2f2cb89f009fd2172b49782804756ea10c7256d4b4.jpg)
        # rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + ".*?\)")
        rep = re.compile(r'\!\[[^\]]*?\]\([^)]*?' + re.escape(image_name) + r'[^)]*?\)')

        # md中进行正在匹配 Match = re.search(md)
        match_obj = rep.search(md_content)
        # 非空校验
        if not match_obj:
            logger.warning(f"{image_name}没有在md中使用,跳过!")
            continue
        # 获取匹配的坐标  .span() -> (start,end)  | star() start  |  end() -> end  |  group() -> 匹配到的字符串  | .string 获取元数据
        start, end = match_obj.span()
        # 获取上下文 100
        # 问题:下标越界   xxx [start]  [end]xxxx    ( [ {
        pre_context = md_content[max(start - 100, 0):start]
        post_context = md_content[end:min(end + 100, len(md_content))]
        image_context_list.append((image_name, str(image_file), (pre_context, post_context)))
        # 返回结果

    return image_context_list

@step_log("step_3_image_summary")
def step_3_image_summary(image_context_list, stem) -> Dict[str,str]:
    """
    作用: 使用视觉模型识别图片描述内容
    入参: [("图片名.xxx","图片地址",(上文 -> 100,下文 -> 100))]   md_path_obj.stem (图片所在的文件夹的名字)
    出参: dict {图片名 : 图片的总结和描述}
    步骤:
        1. 准备工作 获取模型客户端对象
        2. 循环上次获取图片的上下文
        3. 封装提示词
        4. 使用模型执行
        5. 获取返回结果 装到 dict中
        6. 返回结果
    """
    # 1. 先准备放结果的字典
    image_summaries_dict = {}

    # 2. 获取模型法对象
    vm_model = get_llm_client(lm_config.lv_model)

    # 3. 循环总结每份图片: image_context_list = [(图片名 str , 图片地址 str , (上 , 下) )]
    for image_name, image_path_str, context in image_context_list:
        # 4. 构建提示词  每60秒只能请求3
        apply_api_rate_limit()
        # 构建prompt
        image_context_prompt = load_prompt("image_summary", root_folder=stem, image_content=context)

        # 图片内容   base64.b64encode(字节) 普通字节转成base64的字节 .decode(encoding="utf-8") 转成字符串
        image_path_obj = Path(image_path_str)
        image_data = base64.b64encode(image_path_obj.read_bytes()).decode(encoding="utf-8")
        mime_type = mimetypes.guess_type(image_name)[0] or "image/jpeg"
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        # 1. 图片地址 必须公网地址 http://xxxx [图片上传到minio]
                        # 2. base64的字符串  字节 -> 字符 -> 服务器 -> 字符 -> 字节
                        "url": f"data:{mime_type};base64,{image_data}"
                    },
                },
                # 本次图片的上下文文本描述
                {"type": "text", "text": image_context_prompt},
            ]
        )

        # langchain chain
        chains = vm_model | StrOutputParser()
        summary = chains.invoke([message])
        image_summaries_dict[image_name] = summary

    return image_summaries_dict

@step_log("step_5_backup_md")
def step_5_backup_md(new_md_content, md_path_obj) -> str:
    """
    将新的md_content内容写入到本地磁盘! xx.md -> xx_new.md
    :param new_md_content:
    :param md_path_obj:
    :return:
    """
    # 1. 动态生成 xx_new.md 的路径  with_name 会保留原目录，仅替换文件名。stem 是无后缀名，suffix 是 .md
    new_path = md_path_obj.with_name(f"{md_path_obj.stem}_new{md_path_obj.suffix}")
    with open(new_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(new_md_content)

    # 3. 返回绝对路径的字符串格式，方便后续日志打印或下一步调用
    return str(new_path)

@step_log("step_4_upload_and_replace")
def step_4_upload_and_replace(image_context_list, image_summaries_dict, md_content, stem) -> str:
    """
    作用: 将图片传递到 minio,拼接网络地址,联合混合模型返回的图片描述,一起替换 md_content 内容
    入参: md_content  [("图片名.xx","图片地址",(上文 -> 100,下文 -> 100))] dict {图片名 : 图片的总结和描述}  md_path_obj.stem
    出参: md_content 替换后的  ![](本地地址) -> ![summary](网络地址)
    步骤:
        1.获取 minio的客户端对象 minio_client
        2.删除原文件名对应的minio对象
        3.循环"图片地址"地址向minio服务器传递文件,并且拼接(端点/桶名/对象名)和记录地址 {图片名:图片地址}
        4. {图片名:图片地址}   {图片名 : 图片的总结和描述} -> 合并到一起 -> {图片名 : (图片描述和总结,图片地址)}
        5. md_content内容替换,循环处理和替换即可 正则 |  rep.sub("要替换进入的内容",md_content)
        6. 最终返回md_content
    """
    # 1.获取 minio的客户端对象 minio_client
    minio_client = get_minio_client()

    # 2.删除原文件名对应的minio对象
    # 2.1 先查询,再删除!  Object object_name
    object_list = minio_client.list_objects(
        # 参数1: 要删除桶的名字
        bucket_name=minio_config.bucket_name,
        # 前缀 ([1:]) 不能使用 /开头 必须使用 /结尾
        prefix=f"{minio_config.minio_img_dir[1:]}/{stem}/",
        # 参数3: 递归查询
        recursive=True
    )
    delete_object_list = (DeleteObject(obj.object_name) for obj in object_list) # Iterable[DeleteObject]
    errors = minio_client.remove_objects(
        # 参数1: 要删除桶的名字
        bucket_name=minio_config.bucket_name,
        delete_object_list=delete_object_list
    )
    for error in errors:
        logger.warning(f"删除失败,失败原因:{error}")
    logger.info("------------删除成功------------")

    # 3.循环"图片地址"地址向minio服务器传递文件,并且拼接(端点/桶名/对象名)和记录地址 {图片名:图片地址}
    image_url_dict = {}
    for image_name, image_path_str, _ in image_context_list:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_name}",  #
                file_path=image_path_str,  # 图片地址
                content_type=mimetypes.guess_type(image_name)[0],
            )
            # 拼接图片的网络地址  端点 + 桶 + 对象名 http://{endpoint}/{bucket_name}/{object_name}
            image_minio_url = (
                f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_name}")
            logger.debug(f"图片: {image_name} 上传成功!回显地址: {image_minio_url} ")
            # 存储图片和对应网络地址对
            image_url_dict[image_name] = image_minio_url
        except Exception as e:
            logger.warning(f"本次图片上传失败:{image_name}, 跳过, 继续上传下一张!")
            continue

    # 4. {图片名:图片地址}   {图片名 : 图片的总结和描述} 合并 -> {图片名 : (图片描述和总结,图片地址)}
    total_image_info = {}
    if not image_url_dict:
        logger.warning(f"{stem} 的图片上传全部失败!")
        return md_content

    for image_name, image_url in image_url_dict.items(): # 图片名 : (图片地址 , 总结 )
        total_image_info[image_name] = (image_url, image_summaries_dict[image_name])

    # 5. md_content内容替换,循环处理和替换即可 正则 |  rep.sub("要替换进入的内容",md_content)
    for image_name, (image_url, image_summary) in total_image_info.items():
        # 找到 md_content  ![](/xxx/xxx.jpg) -> ![summary](image_url)
        # rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + ".*?\)")
        rep = re.compile(
            r"\!\[[^\]]*?\]\([^)]*?" + re.escape(image_name) + r"[^)]*?\)"
        )
        # rep .findall finditer match search sub
        # 参数1: 要替换入的内容 参数2: 要对哪个文档进行替换
        # 返回值: 替换后的新文档内容
        # 第一种方案,替换入的字符串也可能出现被正则识别的元字符,也会会出现异常!
        # md_content =  rep.sub(f"![{image_summary}]({image_url})",md_content)
        # 第二种方案,使用lambda表达式处理, 正则就不会处理,只会使用原生的正则返回值!
        md_content = rep.sub(lambda _: f"![{image_summary}]({image_url})", md_content)
    # 6. 最终返回md_content
    return md_content

@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    MD文件图片处理核心节点 - 五步法完成图片全流程处理
    核心流程：
    1. 初始化获取MD内容、文件路径、图片文件夹路径
    2. 扫描图片文件夹，筛选MD中实际引用的支持格式图片
    3. 调用多模态大模型为图片生成内容摘要
    4. 将图片上传至MinIO，替换MD中本地图片路径为MinIO访问URL，并填充图片摘要
    5. 备份原MD文件，保存处理后的新MD文件并更新状态
    :param state: 导入流程全局状态对象，包含task_id、md_path、md_content等核心参数
    :return: 更新后的全局状态对象（md_content/md_path为处理后新值）
    """
    # 记录进行状态
    add_running_task(state['task_id'], "node_md_img")

    # 1. 准备和校验
    md_content, md_path_obj, images_path_obj = step_1_get_content(state)
    # 提前结束识别
    if not images_path_obj.exists() or len(list(images_path_obj.iterdir())) == 0:
        logger.warning(f"图片文件夹为空 或 没有图片, 无需后续处理!")
        return state

    # 2. 扫描图片的上下文 step_2_scan_images  [(图片名,图片地址,(上文,下文))]
    image_context_list = step_2_scan_images(md_content, images_path_obj)
    # logger.info(f"已经获取图片的上下文：{image_context_list}")

    # 3. 调用模型识别图片的描述文本   {图片名 : 描述 } ....
    image_summaries_dict = step_3_image_summary(image_context_list, md_path_obj.stem)
    print(image_summaries_dict)

    # 4. 上传图片,并且替换md_content内容
    new_md_content = step_4_upload_and_replace(image_context_list, image_summaries_dict, md_content, md_path_obj.stem)
    # logger.info(f"MD图片处理完成，新文件已保存：{new_md_file_name}")
    # 5. new_md_content 新的备份
    new_md_path_str = step_5_backup_md(new_md_content, md_path_obj)

    # 6. 更新state数据
    state['md_content'] = new_md_content
    state['md_path'] = new_md_path_str

    add_done_task(state['task_id'], "node_md_img")

    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\H3C MER系列路由器 用户手册-R0821-6W105-整本手册", "H3C MER系列路由器 用户手册-R0821-6W105-整本手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")