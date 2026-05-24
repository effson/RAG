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

def find_image_in_md(md_content: str, image_filename: str, context_len: int = 100) -> List[Tuple[str, str]]:
    """
    查找MD内容中指定图片的所有引用位置，并返回每个位置的上下文文本
    :param md_content: MD文件完整内容
    :param image_filename: 图片文件名（含后缀）
    :param context_len: 上下文截取长度，默认前后各100字符
    :return: 上下文列表，每个元素为(上文, 下文)元组，无匹配则返回空列表
    """
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)")
    results = []

    for m in pattern.finditer(md_content):
        start, end = m.span()
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        # 打印图片上下文，便于调试
        logger.debug(f"图片[{image_filename}]匹配到引用，上文：{pre_text.strip()}")
        logger.debug(f"图片[{image_filename}]匹配到引用，下文：{post_text.strip()}")
        results.append((pre_text, post_text))
    if not results:
        logger.debug(f"MD内容中未找到图片[{image_filename}]的引用")
    return results

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

def encode_image_to_base64(image_path: str) -> str:
    """
    将本地图片文件编码为Base64字符串（用于多模态大模型输入）
    :param image_path: 图片本地完整路径
    :return: 图片的Base64编码字符串（UTF-8解码）
    """
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    logger.debug(f"图片Base64编码完成，文件：{image_path}，编码后长度：{len(base64_str)}")
    return base64_str

def summarize_image(image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
    """
    调用多模态大模型生成图片内容摘要（适配LangChain工具类，复用项目统一LLM客户端）
    生成的摘要用于Markdown图片标题，严格控制50字以内中文描述
    :param image_path: 图片本地完整路径
    :param root_folder: 文档所属文件夹/主名，为大模型提供上下文
    :param image_content: 图片在MD中的上下文元组，格式(上文文本, 下文文本)
    :return: 图片内容摘要（异常时返回默认值"图片描述"）
    """
    # 将图片编码为Base64，适配多模态大模型输入要求
    base64_image = encode_image_to_base64(image_path)
    try:
        # 1. 获取项目统一LLM客户端（自动缓存，传入多模态模型名）
        lvm_client = get_llm_client(model=lm_config.lv_model)

        # 加载并渲染提示词（核心：传入所有占位符对应的变量）
        prompt_text = load_prompt(
            name="image_summary",  # 提示词文件名（不带.prompt）
            root_folder=root_folder,  # 对应{root_folder}
            image_content=image_content  # 对应{image_content[0]}、{image_content[1]}
        )

        # 2. 构造LangChain标准多模态HumanMessage（兼容千问/OpenAI等视觉模型）
        messages = [
            HumanMessage(
                content=[
                    # 文本提示词：携带上下文，限定摘要规则
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    # 多模态核心：Base64编码图片数据
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
        ]

        # 3. LangChain标准调用：invoke方法（工具类已封装超时/重试等参数）
        response = lvm_client.invoke(messages)

        # 4. 解析响应（LangChain统一返回content字段，统一格式无需多层解析）
        summary = response.content.strip().replace("\n", "")
        logger.info(f"图片摘要生成成功：{image_path}，摘要：{summary}")
        return summary

    except LangChainException as e:
        logger.error(f"图片摘要生成失败（LangChain框架异常）：{image_path}，错误信息：{str(e)}")
        return "图片描述"
    except Exception as e:
        logger.error(f"图片摘要生成失败（系统异常）：{image_path}，错误信息：{str(e)}")
        return "图片描述"

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


def clean_minio_directory(minio_client: Minio, prefix: str) -> None:
    """
    幂等性清理MinIO指定目录下的所有旧文件，防止重名文件内容混淆和垃圾文件堆积
    幂等性：多次调用结果一致，无文件时不报错
    :param minio_client: 初始化完成的MinIO客户端对象
    :param prefix: MinIO目录前缀（要清理的目录路径）
    """
    try:
        # 列出指定前缀下的所有对象（递归遍历子目录）
        objects_to_delete = minio_client.list_objects(
            bucket_name=minio_config.bucket_name,
            prefix=prefix,
            recursive=True
        )
        # 构造删除对象列表
        delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]

        if delete_list:
            logger.info(f"开始清理MinIO旧文件，待删除文件数：{len(delete_list)}，目录：{prefix}")
            # 批量删除对象
            errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
            # 遍历删除错误信息，记录异常
            for error in errors:
                logger.error(f"MinIO文件删除失败：{error}")
        else:
            logger.debug(f"MinIO目录无旧文件，无需清理：{prefix}")
    except Exception as e:
        logger.error(f"MinIO目录清理失败：{prefix}，错误信息：{str(e)}")


def upload_images_batch(minio_client: Minio, upload_dir: str, targets: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[
    str, str]:
    """
    批量上传待处理图片至MinIO，返回图片文件名与访问URL的映射关系
    :param minio_client: 初始化完成的MinIO客户端对象
    :param upload_dir: MinIO上传根目录
    :param targets: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
    :return: 图片URL字典，键：图片文件名，值：MinIO访问URL
    """
    urls = {}
    for img_file, img_path, _ in targets:
        # 构造MinIO对象名称
        object_name =  f"{upload_dir}/{img_file}"
        logger.debug(f"构造MinIO对象名称完成：{object_name}")
        # 上传单张图片并获取URL
        """
        := 是 Python 3.8+ 引入的海象运算符（Walrus Operator），核心作用是 **「表达式内赋值 + 结果判断」一体化 **：
        在执行判断、循环等逻辑的同一个表达式中，完成变量赋值和赋值结果的使用 / 判断，替代传统「先赋值、后判断」的两行代码，让逻辑更简洁。
        """
        if img_url := upload_to_minio(minio_client, img_path, object_name):
            urls[img_file] = img_url
    logger.info(f"图片批量上传完成，成功上传{len(urls)}/{len(targets)}张图片")
    return urls

def upload_to_minio(minio_client: Minio, local_path: str, object_name: str) -> str | None:
    """
    将单张本地图片上传至MinIO对象存储，并返回公网可访问URL
    :param minio_client: 初始化完成的MinIO客户端对象
    :param local_path: 图片本地完整路径
    :param object_name: MinIO中要存储的对象名称（带目录）
    :return: 图片MinIO访问URL（上传失败返回None）
    """
    try:
        logger.info(f"开始上传图片至MinIO：本地路径={local_path}，MinIO对象名={object_name}")
        # 上传本地文件至MinIO（fput_object：文件流上传，适合大文件）
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,  # MinIO存储桶名（从配置读取）
            object_name=object_name,  # MinIO对象名称
            file_path=local_path,  # 本地文件路径
            # 自动推断图片Content-Type（如image/png、image/jpeg）
            # 入参：文件路径字符串（可带目录，如/a/b/test.jpg、demo.tar.gz）；
            # 返回值：元组(root, ext)，其中：
            # root：文件主名（含目录，去掉最后一个后缀的完整部分）；
            # ext：文件后缀（以.开头，仅包含最后一个扩展名，如.jpg、.gz，无后缀则为空字符串""）；
            # 关键规则：仅识别 ** 最后一个.** 作为后缀分隔符，多后缀文件仅拆分最后一个（如test.tar.gz拆分为("test.tar", ".gz")）。
            content_type=f"image/{os.path.splitext(local_path)[1][1:]}"
        )

        # 处理路径特殊字符，避免URL解析错误
        # 假设原始 object_name 是：图片\logo.png
        # 替换后变成：图片%5Clogo.png
        # 这个字符串是URL 合法格式，所有服务器 / 浏览器都能正确识别；
        # MinIO 接收到 %5C 后，会自动解析回 \，保证对象名的正确性；
        # 后续通过 URL 访问时，%5C 会被正确解码，不会出现路径错误。
        object_name = object_name.replace("\\", "%5C")
        # 根据配置选择HTTP/HTTPS协议
        protocol = "https" if minio_config.minio_secure else "http"
        # 构造MinIO基础访问URL
        base_url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}"
        # 拼接完整图片访问URL base_url 后面带 / 中间直接两个字符串拼接即可
        img_url = f"{base_url}{object_name}"
        logger.info(f"图片上传成功，访问URL：{img_url}")
        return img_url
    except Exception as e:
        logger.error(f"图片上传MinIO失败：{local_path}，错误信息：{str(e)}")
        return None

def merge_summary_and_url(summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """
    合并图片摘要字典和URL字典，过滤掉上传失败无URL的图片
    :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
    :param urls: 图片URL字典，键：图片文件名，值：MinIO访问URL
    :return: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)元组
    """
    image_info = {}
    # 遍历摘要字典，仅保留有对应URL的图片
    for image_file, summary in summaries.items():
        if url := urls.get(image_file):
            image_info[image_file] = (summary, url)
    logger.info(f"图片摘要与URL合并完成，有效图片信息{len(image_info)}条")
    return image_info


def process_md_file(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
    """
    核心功能：替换MD内容中的本地图片引用为MinIO远程引用
    替换规则：![原描述](本地路径) → ![图片摘要](MinIO访问URL)
    :param md_content: 原始MD文件内容
    :param image_info: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)
    :return: 替换后的新MD内容
    """
    for img_filename, (summary, new_url) in image_info.items():
        # 正则匹配MD图片标签，忽略大小写，兼容不同路径写法
        # 正则规则：![任意描述](任意路径+图片文件名+任意后缀)
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_filename) + r".*?\)",
            re.IGNORECASE
        )
        # 替换匹配内容：使用新摘要作为图片描述，新URL作为图片路径
        # - 如果你的 summary 和 new_url 是完全可控的纯文本（不含反斜杠） ：这两种写法确实 一模一样 。
        # - 如果你想写出“防御性代码”（Defensive Code），防止未来某天被特殊字符坑 ：请坚持使用 Lambda 写法 。它是最稳健、最安全的做法。
        # md_content = pattern.sub(lambda m: f"![{summary}]({new_url})", md_content)
        md_content = pattern.sub( f"![{summary}]({new_url})", md_content)
        logger.debug(f"完成MD图片引用替换：{img_filename} → {new_url}")

    logger.info(f"MD文件图片引用替换完成，共替换{len(image_info)}处图片引用")
    logger.debug(f"替换后MD内容：{md_content[:500]}..." if len(md_content) > 500 else f"替换后MD内容：{md_content}")
    return md_content

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
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
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