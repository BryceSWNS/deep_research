"""Deep Research agent 的工具函数与辅助函数。"""

import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.state import ResearchComplete, Summary

##########################
# Tavily 搜索工具辅助函数
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "面向全面、准确、可信结果优化的搜索引擎。"
    "适用于需要回答时效性问题的场景。"
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str], 
    max_results: Annotated[int, InjectedToolArg] = 5, 
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general", 
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
) -> str:
    """从 Tavily search API 抓取并总结搜索结果.

    Args:
        queries:要执行的搜索查询列表
        max_results: 每个查询返回的最大结果数
        topic: 搜索结果的主题过滤器 (general, news, or finance)
        config: 包含 API 密钥和模型设置的运行配置

    Returns:
        包含总结后的搜索结果的格式化字符串 
    """
    # Step 1: 异步执行搜索查询
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    
    # Step 2: 基于URL去重搜索结果，避免重复处理相同内容
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    
    # Step 3: 根据配置初始化摘要模型
    configurable = Configuration.from_runnable_config(config)
    
    # 字符长度上限（可配置），用于控制上下文体量
    max_char_to_include = configurable.max_content_length
    
    # 初始化 summarization model，并启用重试
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    
    # Step 4: 构建 summarization tasks（空内容跳过）
    async def noop():
        """无 raw_content 时的空操作。"""
        return None
    
    summarization_tasks = [
        noop() if not result.get("raw_content") 
        else summarize_webpage(
            summarization_model, 
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]
    
    # Step 5: 并行执行所有 summarization tasks
    summaries = await asyncio.gather(*summarization_tasks)
    
    # Step 6: 合并检索结果与对应摘要
    summarized_results = {
        url: {
            'title': result['title'], 
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(), 
            unique_results.values(), 
            summaries
        )
    }
    
    # Step 7: 格式化最终输出
    if not summarized_results:
        return "未找到有效搜索结果。请尝试更换查询词或使用其他 search API。"
    
    formatted_output = "搜索结果：\n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- 来源 {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"摘要：\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    return formatted_output

async def tavily_search_async(
    search_queries, 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
    config: RunnableConfig = None
):
    """异步执行多条 Tavily 搜索查询。
    
    Args:
        search_queries: 待执行的查询字符串列表
        max_results: 每条查询返回的最大结果数
        topic: 结果主题过滤（general/news/finance）
        include_raw_content: 是否包含网页完整原文
        config: 运行时配置（用于读取 API key）
        
    Returns:
        Tavily API 返回的结果字典列表
    """
    # 用配置中的 API key 初始化 Tavily client
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    
    # 构造并行搜索任务
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]
    
    # 并行执行所有查询并返回结果
    search_results = await asyncio.gather(*search_tasks)
    return search_results

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """使用 AI 模型总结网页内容，并带超时保护。
    
    Args:
        model: 已配置好的 summarization chat model
        webpage_content: 待总结的网页原文
        
    Returns:
        带关键摘录的格式化摘要；若失败则返回原文
    """
    try:
        # 生成带日期上下文的 prompt
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content, 
            date=get_today_str()
        )
        
        # 执行摘要调用，并设置超时避免卡住
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0  # 摘要调用超时 60 秒
        )
        
        # 结构化输出摘要内容
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        
        return formatted_summary
        
    except asyncio.TimeoutError:
        # 摘要超时：返回原文
        logging.warning("摘要调用超过 60 秒，返回原始内容")
        return webpage_content
    except Exception as e:
        # 其他异常：记录日志并返回原文
        logging.warning(f"摘要失败：{str(e)}，返回原始内容")
        return webpage_content

##########################
# 反思工具相关
##########################

@tool(description="用于 research 规划的策略反思工具")
def think_tool(reflection: str) -> str:
    """用于记录 research 进展与决策思考的策略反思工具。

    建议在每次搜索后调用，用于分析结果并系统规划下一步。
    它会在 research workflow 中形成一个有意识的“停顿点”，帮助提高决策质量。

    建议使用时机：
    - 拿到搜索结果后：我找到了哪些关键信息？
    - 决定下一步前：当前信息是否足够完整回答？
    - 评估信息缺口时：还缺哪些具体信息？
    - 结束研究前：是否已经能给出完整结论？

    反思内容建议覆盖：
    1. 当前发现分析：已拿到哪些具体信息？
    2. 缺口评估：还缺哪些关键事实？
    3. 质量评估：证据与案例是否足够支撑高质量回答？
    4. 策略决策：应继续搜索还是输出答案？

    Args:
        reflection: 对研究进展、已有发现、信息缺口与下一步计划的详细反思

    Returns:
        一条“反思已记录”的确认文本
    """
    return f"已记录反思：{reflection}"

##########################
# MCP 相关
##########################

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """通过 OAuth token exchange 将 Supabase token 换成 MCP access token。
    
    Args:
        supabase_token: 有效的 Supabase 认证 token
        base_mcp_url: MCP server 基础 URL
        
    Returns:
        成功时返回 token 字典，失败返回 None
    """
    try:
        # 组装 OAuth token exchange 请求参数
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        
        # 发起 token exchange 请求
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # 成功获取 token
                    token_data = await response.json()
                    return token_data
                else:
                    # 记录错误详情，便于排查
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")
                    
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    
    return None

async def get_tokens(config: RunnableConfig):
    """读取已存储 token，并校验是否过期。
    
    Args:
        config: 运行时配置（包含 thread/user 标识）
        
    Returns:
        token 有效则返回 token 字典，否则返回 None
    """
    store = get_store()
    
    # 从 config 提取必要标识
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    
    # 读取已存 token
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    
    # 校验 token 是否过期
    expires_in = tokens.value.get("expires_in")  # 距离过期的秒数
    created_at = tokens.created_at  # token 创建时间
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    
    if current_time > expiration_time:
        # token 已过期，清理后返回 None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """将认证 token 写入配置存储。
    
    Args:
        config: 运行时配置（包含 thread/user 标识）
        tokens: 待存储的 token 字典
    """
    store = get_store()
    
    # 从 config 提取必要标识
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    
    # 写入 token
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """获取并刷新 MCP token（必要时重新申请）。
    
    Args:
        config: 含认证信息的运行时配置
        
    Returns:
        返回有效 token 字典；无法获取时返回 None
    """
    # 先尝试读取现有有效 token
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    
    # 提取 Supabase token，用于新一轮 token exchange
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    
    # 提取 MCP 配置
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    
    # 使用 Supabase token 换取 MCP token
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # 存储新 token 并返回
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """为 MCP tool 包一层认证与错误处理逻辑。
    
    Args:
        tool: 需要包装的 MCP structured tool
        
    Returns:
        带认证与错误处理能力的增强版 tool
    """
    original_coroutine = tool.coroutine
    
    async def authentication_wrapper(**kwargs):
        """增强后的 coroutine：处理 MCP 错误并返回更友好的报错信息。"""
        
        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """在异常链中递归查找 McpError。"""
            if isinstance(exc, McpError):
                return exc
            
            # 兼容 Python 3.11+ 的 ExceptionGroup
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None
        
        try:
            # 先执行原始 tool
            return await original_coroutine(**kwargs)
            
        except BaseException as original_error:
            # 在异常链中查找 MCP 专属错误
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # 非 MCP 错误，原样抛出
                raise original_error
            
            # 处理 MCP 特定错误
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            
            # 检查“需要认证/交互”的错误
            if error_code == -32003:  # 需要交互的错误码
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"
                
                # 若有可读文本，优先使用
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                
                # 若提供了 URL，附加给用户
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                
                raise ToolException(error_message) from original_error
            
            # 其他 MCP 错误，原样抛出
            raise original_error
    
    # 用增强版 coroutine 替换原始实现
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """加载并配置 MCP（Model Context Protocol）tools（含认证逻辑）。
    
    Args:
        config: 包含 MCP server 信息的运行时配置
        existing_tool_names: 已占用 tool 名称集合（用于避免冲突）
        
    Returns:
        可直接使用的 MCP tools 列表
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Step 1: 如需认证，先处理认证
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    
    # Step 2: 校验配置是否完整可用
    config_valid = (
        configurable.mcp_config and 
        configurable.mcp_config.url and 
        configurable.mcp_config.tools and 
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )
    
    if not config_valid:
        return []
    
    # Step 3: 构建 MCP server 连接参数
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    
    # 若有 token，则附加认证请求头
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
    
    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO: OAP 支持 Multi-MCP Server 后，更新此处实现
    
    # Step 4: 从 MCP server 拉取 tools
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # MCP server 连接失败时返回空列表
        return []
    
    # Step 5: 过滤并配置 tools
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # 跳过名称冲突的 tool
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue
        
        # 仅保留配置中显式声明的 tools
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue
        
        # 包装认证逻辑后加入结果列表
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)
    
    return configured_tools


##########################
# 工具装配相关
##########################

async def get_search_tool(search_api: SearchAPI):
    """根据指定的 search API provider 返回对应搜索工具。
    
    Args:
        search_api: 要使用的 search API provider（Anthropic/OpenAI/Tavily/None）
        
    Returns:
        指定 provider 对应的已配置 search tools 列表
    """
    if search_api == SearchAPI.ANTHROPIC:
        # Anthropic 原生 web search（带调用上限）
        return [{
            "type": "web_search_20250305", 
            "name": "web_search", 
            "max_uses": 5
        }]
        
    elif search_api == SearchAPI.OPENAI:
        # OpenAI web search preview
        return [{"type": "web_search_preview"}]
        
    elif search_api == SearchAPI.TAVILY:
        # 配置 Tavily search tool 元数据
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}), 
            "type": "search", 
            "name": "web_search"
        }
        return [search_tool]
        
    elif search_api == SearchAPI.NONE:
        # 不配置 search 功能
        return []
        
    # 未知类型的兜底返回
    return []
    
async def get_all_tools(config: RunnableConfig):
    """组装完整工具集（research + search + MCP）。
    
    Args:
        config: 指定 search API 与 MCP 配置的运行时参数
        
    Returns:
        research 流程可用的全部 tools 列表
    """
    # 先加入核心研究 tools
    tools = [tool(ResearchComplete), think_tool]
    
    # 加入已配置的 search tools
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    
    # 收集已占用 tool 名称，避免冲突
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    
    # 若配置了 MCP，再追加 MCP tools
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """从 tool 消息中提取 notes 文本。"""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# 模型提供方原生 Web Search 判断相关
##########################

def anthropic_websearch_called(response):
    """判断响应中是否调用了 Anthropic 原生 web search。
    
    Args:
        response: Anthropic API 响应对象
        
    Returns:
        调用了 web search 返回 True，否则 False
    """
    try:
        # 读取响应 metadata 结构
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        
        # 检查 server 侧 tool 使用信息
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        
        # 读取 web search 请求计数
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        
        # 只要请求数大于 0 即视为调用过
        return web_search_requests > 0
        
    except (AttributeError, TypeError):
        # 响应结构异常时兜底返回 False
        return False

def openai_websearch_called(response):
    """判断响应中是否调用了 OpenAI web search。
    
    Args:
        response: OpenAI API 响应对象
        
    Returns:
        调用了 web search 返回 True，否则 False
    """
    # 从响应 metadata 中读取 tool outputs
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False
    
    # 检查是否存在 web_search_call 记录
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True
    
    return False


##########################
# Token 超限判断相关
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """判断异常是否由 token/context 上限触发。
    
    Args:
        exception: 待分析异常对象
        model_name: 可选模型名（用于优化 provider 判定）
        
    Returns:
        若为 token 上限问题返回 True，否则 False
    """
    error_str = str(exception).lower()
    
    # Step 1: 若提供模型名，先推断 provider
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    
    # Step 2: 按 provider 规则检查 token 超限
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    # Step 3: provider 未知时，依次检查所有规则
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """检查是否为 OpenAI token 超限异常。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # 判断是否属于 OpenAI 异常
    is_openai_exception = (
        'openai' in exception_type.lower() or 
        'openai' in module_name.lower()
    )
    
    # 检查常见 OpenAI 请求错误类型
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']
    
    if is_openai_exception and is_request_error:
        # 在错误文本中匹配 token 相关关键词
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    
    # 检查 OpenAI 特定错误码
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')
        
        if (error_code == 'context_length_exceeded' or
            error_type == 'invalid_request_error'):
            return True
    
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """检查是否为 Anthropic token 超限异常。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # 判断是否属于 Anthropic 异常
    is_anthropic_exception = (
        'anthropic' in exception_type.lower() or 
        'anthropic' in module_name.lower()
    )
    
    # 匹配 Anthropic 特有错误模式
    is_bad_request = class_name == 'BadRequestError'
    
    if is_anthropic_exception and is_bad_request:
        # Anthropic token 超限通常使用固定报错文案
        if 'prompt is too long' in error_str:
            return True
    
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """检查是否为 Google/Gemini token 超限异常。"""
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    # 判断是否属于 Google/Gemini 异常
    is_google_exception = (
        'google' in exception_type.lower() or 
        'google' in module_name.lower()
    )
    
    # 匹配 Google 常见资源耗尽错误
    is_resource_exhausted = class_name in [
        'ResourceExhausted', 
        'GoogleGenerativeAIFetchError'
    ]
    
    if is_google_exception and is_resource_exhausted:
        return True
    
    # 补充匹配 Google API resource exhausted 模式
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

# 注意：该映射可能过时或不适用于你的模型，请按需更新。
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}

def get_model_token_limit(model_string):
    """查询指定模型的 token 上限。
    
    Args:
        model_string: 模型标识字符串
        
    Returns:
        找到则返回整数上限；未命中映射则返回 None
    """
    # 在已知映射表中查找
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    
    # 映射表中未找到
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """裁剪消息历史：移除“最后一条 AI 消息”及其之后内容。
    
    常用于 token 超限时回退最近上下文。
    
    Args:
        messages: 待裁剪消息列表
        
    Returns:
        裁剪后的消息列表（不含最后一条 AI 消息）
    """
    # 从后向前查找最后一条 AI 消息
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # 返回该消息之前的全部内容
            return messages[:i]
    
    # 若不存在 AI 消息，返回原列表
    return messages

##########################
# 杂项工具
##########################

def get_today_str() -> str:
    """获取当前日期字符串（用于 prompts 与输出展示）。
    
    Returns:
        可读日期字符串，例如 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """从配置字段中提取值，兼容 enum 与 None。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """从环境变量或配置文件中获取某个特定模型的 API Key"""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("deepseek:"):
            return api_keys.get("DEEPSEEK_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("deepseek:"):
            return os.getenv("DEEPSEEK_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    """从环境变量或配置文件中获取 Tavily 的 API Key."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")
