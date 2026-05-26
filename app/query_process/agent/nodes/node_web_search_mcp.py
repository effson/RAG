import asyncio
import os
import json
import sys
from agents.mcp import MCPServerSse # pip install openai-agents
from agents.mcp import MCPServerStreamableHttp # pip install openai-agents

from app.conf.bailian_mcp_config import mcp_config
from app.query_process.agent.nodes.node_search_embedding import step_1_data_validates
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger, node_log, step_log
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_BASE_URL_STREAMABLE_HTTP = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
DASHSCOPE_API_KEY = mcp_config.api_key

@step_log("step_1_data_validate")
def step_1_data_validate(state):
    rewritten_query = state['rewritten_query']
    if not rewritten_query:
        logger.error("rewritten_query不能为空!")
        raise ValueError("rewritten_query不能为空!")
    return rewritten_query

@step_log("node_web_search_mcp_async")
async def node_web_search_mcp_async(rewritten_query: str, count: int = 5):
    """
    使用openai的方式调用 mcp server 提供的工具
    :param rewritten_query:
    :param count:
    :return:
    """
    # 1. 链接 mcp server 服务
    mcp_server = MCPServerStreamableHttp(
        name="rag_search_mcp", # 随便写
        params={
            "url": DASHSCOPE_BASE_URL_STREAMABLE_HTTP, #
            "headers": {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            "timeout": 300,
            "sse_read_timeout": 300
        },
        client_session_timeout_seconds=60,
        cache_tools_list=True,
        max_retry_attempts=3,
    )

    try:
        # 2. 进行mcp_server链接
        await mcp_server.connect()
        # 3. 调用工具
        # https: // openai.github.io / openai - agents - python / ref / mcp / server /  # agents.mcp.server.MCPServer.call_tool
        tool_list =  await mcp_server.list_tools()
        logger.info(f"工具列表:{tool_list}")
        mcp_result = await mcp_server.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": rewritten_query,
                "count": count
            }
        )
        return mcp_result
    finally:
        # 4.释放本次链接资源
        await mcp_server.cleanup()

def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    # 任务和认知
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    # 1. 获取数据和校验
    rewritten_query = step_1_data_validate(state)

    # 2. mcp的调用流程封装成一个异步函数
    mcp_result = asyncio.run(node_web_search_mcp_async(rewritten_query, count=10))

    # 3. 结果解析
    # mcp_result = {content:[{text:"{pages:[{x,x,x}]}"}]}
    text_dict = json.loads(mcp_result.content[0].text)
    pages = text_dict.get('pages', [])

    add_done_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])

    return {"web_search_docs":pages}

if __name__ == '__main__':
    test_state = {
        "session_id":"xxxx",
        "is_stream":False,
        "rewritten_query": "今天上海的天气怎么样？"
    }

    # 调用 websearch_node 函数
    result_state = node_web_search_mcp(test_state)

    # 验证结果
    print("测试结果:")
    print(f"查询内容: {test_state.get('rewritten_query')}")
    print(f"答案内容: {result_state}")
