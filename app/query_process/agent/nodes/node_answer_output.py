import sys
import time

from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger, step_log, node_log
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re
"""
 1. 日志 + 任务处理
      2. 获取state -> answer -> 存在 return true | false
          存在 -> [非流式] task_utils.set_task_result(session,"answer",模型提供文本答案)
                  [流式]  使用delta事件类型进行数据推送 for ch in answer:  push_to_session(session_id,delta,{"delta":ch}) time.sleep(0.03)
          3. 没有答案答案,获取参数和校验 reranker_docs rewritten_query item_names history
          4. 拼接提示词 reranker_docs rewritten_query item_names history
          5. 封装模型提问 (提示词)   答案
                 content =   llm.invoke(prompt)
              [非流式] task_utils.set_task_result(session,"answer",模型提供文本答案)
                 ch chunk.content =  llm.stream(prompt)
                 大家好,我是谁谁谁!!!
                 大
                 家
                 好,
                 我是
                 谁谁谁
                 result += ch
              [流式]  使用delta事件类型进行数据推送 for ch in answer:  push_to_session(session_id,delta,{"delta":ch})
                 set_task_result()  -> server -> final
         6. 提取chunks切块中的图片链接 -> [] -> state
              reranker_docs : [ {title,text,url,type,score}]
              图片 -> text || url
              reg = re.compile(r"\!\[(.*?\)]\((.*?)\)") -> text -> findall -> [(图片描述,url),(),()]
              image_urls = []
              reranker_docs循环
                   url - 是不是图片 -> 是 -> 判断是否存在 not in  -> 存储起来
                   text -> 提取图片 -> 是 -> 判断是否存在 not int -> 存储起来

              state['images_url'] = image_urls
     7. 存储助手对应的回答历史记录
            mongodb对应工具类
            role
            text 回答
            item_name
            image_urls

"""

@step_log("step_1_validate_answer")
def step_1_validate_answer(state):
    """
    校验有没有answer
       有: 之前item_name的时候,没有获取确认的item_name
       没有: 流程正常
    :param state:
    :return:
    """
    answer = state.get("answer")
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id")

    # 判断answer是否为空
    if answer:
        # 输出结果
        if is_stream:
            # 流式输出
            # 给前端sse响应对应的数据
            # 思路1: task_utils [add_done add_running update_task_status  is_stream]  [任务 任务状态]  process
            # 思路2: sse_utils  push_to_session(session_id,事件类型,数据)
            for ch in answer:
                # 推送到队列
                push_to_session(session_id,SSEEvent.DELTA,{"delta":ch})
                time.sleep(0.3)
        # 非流式输出
        # 存储数据 -> server -> get_task_result -> json返回
        set_task_result(session_id,"answer",answer)
        return True
    return False

@step_log("step_2_data_validates")
def step_2_data_validates(state):
    """
    history,reranked_docs,item_names,rewritten_query
    :param state:
    :return:
    """
    history = state.get("history", [])
    reranked_docs = state.get("reranked_docs", [])
    item_names = state.get("item_names", [])
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    kg_triple_text = state.get("kg_triple_text", "")

    # reranked_docs || rewritten_query 为空无法继续
    if not reranked_docs or len(reranked_docs) == 0 or not  rewritten_query:
        logger.error(f"reranked_docs或者rewritten_query为空,无法使用模型进行答案匹配!")
        raise ValueError("reranked_docs或者rewritten_query为空,无法使用模型进行答案匹配!")

    return history, reranked_docs, item_names, rewritten_query, kg_triple_text

@step_log("step_3_make_prompt")
def step_3_make_prompt(reranked_docs, rewritten_query, item_names, history, kg_triple_text=""):
    """
    组装提示词
        【参考内容】
          {context}  -> 切块的答案
        【历史对话】
          {history}  -> 聊天记录
        【相关商品/实体】
          {item_names} -> item_name
        【用户问题】
          {question} -> 问题
       context -> reranked_docs -> {title,text,type,score,url} {title,text,type,score,url} {title,text,type,score,url}
          第1块: 标题:title 匹配度得分: score 来源: 网络搜索 / 本地向量查询
          内容:  text
          \n\m
          第1块: 标题:title 匹配度得分: score 来源: 网络搜索 / 本地向量查询
          内容:  text
       history -> history -> 聊天记录
          (f"角色:{msg['role']},内容:{ msg['rewritten_query'] if msg['role'] == 'user' else msg['text']}"
             f",关联主体: {'、'.join(msg['item_names'])}\n")
       item_names -> item_name
           本次关联主体: a,b,c,d
    :param reranked_docs:
    :param rewritten_query:
    :param item_names:
    :param history:
    :return:
    """
    context_chunk_list = []
    # 装每块
    for number, chunk in enumerate(reranked_docs, start=1):
        context_chunk_list.append(
            f"第{number}块: 标题:{chunk['title']} 匹配度得分:{chunk['score']} 来源:{'网络搜索' if chunk['type'] == 'web' else '向量查询'}"
            f"\n"
            f"内容:{chunk['text']}"
        )
    context_chunk_str = "\n\n".join(context_chunk_list)

    # 组装history
    history_text = "没有历史聊天记录!"
    if history and len(history) > 0:
        history_text = ""
        for msg in history:
            history_text += \
                (f"角色:{msg['role']},内容:{msg['rewritten_query'] if msg['role'] == 'user' else msg['text']}"
                 f",关联主体: {'、'.join(msg.get('item_names',[]))}\n")

    # item_name
    # 本次关联主体: a, b, c, d
    item_name_str ="本次关联主体:" + ",".join(item_names) if item_names and len(item_names) > 0 else '没有关联主体'

    # KG 三元组文本（来自 node_search_kg step_4e 的 _format_triple_text）
    kg_relations_str = kg_triple_text if kg_triple_text else "无"

    # 加载提示词
    prompt = load_prompt("answer_out", context=context_chunk_str,
                history=history_text, item_names=item_name_str,
                question=rewritten_query, kg_relations=kg_relations_str)
    return prompt

@step_log("step_4_call_lm_final_answer")
def step_4_call_lm_final_answer(state, prompt):
    """
      最后一次调用大语言模型!
      进行最终的答案生成,以及返回给前端数据
    :param state:
    :param prompt:
    :return:
    """

    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id")

    # 1. 获取模型客户端
    lm_client = get_llm_client()
    # 2. 模型执行提示词
    final_result = "" # 接收最终结果
    if is_stream:
        # 流式 chunk 每次结果都随时推送队列中即可!!
        for chunk in lm_client.stream(prompt): # 每次给增量答案  你 好  啊   [dela模式]
            # chunk message {content属性:内容 增量内容}
            delta_content = chunk.content
            final_result += delta_content
            # 立即推送到队列!
            # 事件: DELTA
            # sse工具 生成器 yield -> sse -> 前端 -> 增量展示
            push_to_session(session_id, SSEEvent.DELTA, {"delta": delta_content})
    else:
        # 一次执行,获取结果
        response = lm_client.invoke(prompt)  # 一次给全部答案  你好啊      [非流式]
        content =  response.content
        final_result = content
    #3.结果存储起来
    set_task_result(session_id,"answer",final_result)
    state['answer'] = final_result

@step_log("step_5_extract_chunk_images")
def step_5_extract_chunk_images(state, reranked_docs):
    """
    提取切片中的图片数据! 存储到state中
    最后,final一起返回数据! 进行图片的渲染
    :param state:
    :param reranked_docs: [{title,[text],type,[url],score}]
    :return:
    """
    # 1. 准备工作  定一个接收图片的集合 定义一个正则表达式
    image_urls = []
    # 正则: compile() 提前编译  match 从头匹配 标题行  search 匹配任意匹配一个  finditer 匹配匹配多个 Match  start end span string group(0 / 1)
    # sub findall 匹配内容 [(xx),(xx)]
    reg = re.compile(r"\!\[.*?\]\((.*?)\)")  # () 提取符号  -> 提取括号内
    # 2. 循环数据,进行提取
    for doc in reranked_docs:
        url = doc.get("url")
        text = doc.get("text")
        # 3. url .png .jpg .gif .jpeg .svg  / text ![](url) [什么是图片??]  放重复
        if url:
            if url.endswith((".png", ".jpg",".gif",".jpeg",".svg")):
                if url not in image_urls:
                    image_urls.append(url)
        # text xxxx![()](())xxxx![](())
        # [1,2]  [(1,2),(1,2)]
        if text:
            image_str_list = reg.findall(text)
            for image_url in image_str_list:
                if image_url not in image_urls:
                    image_urls.append(image_url)
    # 3.存到state
    state['image_urls'] = image_urls

@step_log("step_6_save_chat_history")
def step_6_save_chat_history(state):
    """
    保存聊天记录!
    模型返回结果的记录!!
    :param state:
    :return:
    """
    save_chat_message(
        session_id= state['session_id'],
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query") or state.get("original_query"),
        item_names=state.get("item_names",[]),
        image_urls = state.get("image_urls",[])
    )

@node_log("node_answer_output")
def node_answer_output(state):
    """
    节点功能：进行过处理可以是流式输出可以整体输出！
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 1. 日志+任务状态处理
    # 2. answer获取,如果存在,直接进行结果返回 False | True
    # False 没有结果 | True 有结果
    has_answer = step_1_validate_answer(state)
    if not has_answer:
        # 3. 如果没有,获取参数和校验
        history, reranked_docs, item_names, rewritten_query, kg_triple_text = step_2_data_validates(state)
        # 4. 组装返回的提示词
        prompt = step_3_make_prompt(reranked_docs, rewritten_query, item_names, history, kg_triple_text)
        state["prompt"] = prompt
        # 5. 调用模型生成润色结果
        step_4_call_lm_final_answer(state,prompt)
        # 6. 获取chunks对应的图片链接地址
        step_5_extract_chunk_images(state,reranked_docs)
    # 7. 写回聊天记录过程
    step_6_save_chat_history(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state

# def node_answer_output(state):
#     """
#     节点功能：进行过处理可以是流式输出可以整体输出！
#     """
#     print("---node_answer_output 节点处理开始---")
#     add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
#
#     session_id = state["session_id"]
#     is_stream = state.get("is_stream", True)
#     base_answer = state.get("answer") or f"这是关于「{state.get('original_query', '当前问题')}」的测试回答，正在演示打字机流式输出效果。"
#     final_text = ""
#
#     if is_stream:
#         for ch in base_answer:
#             final_text += ch
#             push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
#             time.sleep(0.03)
#         logger.info(f"流式输出完成，总长度: {len(final_text)}")
#     else:
#         final_text = base_answer
#
#     # 执行完毕之前 存储结果
#     set_task_result(session_id,"answer",final_text)
#     add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
#     print("---node_answer_output 节点处理结束---")
#     return {"answer": final_text}


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_answer_output 本地测试")
    print("=" * 50)

    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "type": "milvus",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://local-server/images/panel_view.jpg)

            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "type": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",  # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
            "text": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "type": "milvus",
            "title": "安全注意事项",
            "score": 0.82,
            "text": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？","rewritten_query":"HAK 180 烫金机的具体操作步骤和面板设置方法"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？","rewritten_query":"HAK 180 烫金机的具体操作步骤和面板设置方法"},
        {"role": "user", "text": "HAK 180 烫金机","rewritten_query":"HAK 180 烫金机的具体操作步骤和面板设置方法"}
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,  # 测试非流式
        # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None,  # 初始无答案
        # 模拟 KG 三元组文本（来自 node_search_kg step_4e）
        "kg_triple_text": "[知识图谱实体关系（一跳）]\n1. \"漏电保护装置\" --[依赖]--> \"电源模块\"\n2. \"电源模块\" --[包含]--> \"电池\"",
    }

    try:
        # 运行节点
        result = node_answer_output(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")

        # 1. 验证 Prompt 构建
        if "prompt" in result:
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
            # print(f"Prompt 预览:\n{result['prompt'][:200]}...")
        else:
            print("[FAIL] Prompt 未构建")

        # 2. 验证答案生成
        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案预览: {answer[:50]}...")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")

        # 3. 验证图片提取
        # 我们期望提取到 3 张图片：
        # 1. http://local-server/images/panel_view.jpg (来自 local_101)
        # 2. http://local-server/images/knob_detail.png (来自 local_101)
        # 3. http://example.com/hak180_troubleshooting.jpeg (来自 web 结果的 url 字段)

        # 注意：这里我们没办法直接从 result state 里拿到 image_urls，因为它是作为 SSE 推送出去的，或者存库了
        # 但我们可以通过日志观察 _extract_images_from_docs 的输出
        # 如果需要验证，可以临时修改 node_answer_output 返回 image_urls
        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")