"""Deep Research Agent 的主 LangGraph 实现"""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model # 初始化聊天模型
from langchain_core.messages import (
    AIMessage, 
    HumanMessage, 
    SystemMessage, 
    ToolMessage, 
    filter_messages, # 过滤消息，只保留 HumanMessage 和 AIMessage
    get_buffer_string, # 将消息转换为字符串
)
from langchain_core.runnables import RunnableConfig # 解释：运行配置
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command # 解释：命令

from open_deep_research.configuration import (
    Configuration,
) 
from open_deep_research.prompts import (
    clarify_with_user_instructions, 
    compress_research_simple_human_message, 
    compress_research_system_prompt,
    final_report_generation_prompt, 
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
) 
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """分析用户消息，并在研究范围不明确时提出澄清问题.
    
    该函数用于在开展研究之前，判断用户的请求是否需要进一步澄清。
    如果未启用澄清机制或无需澄清，则直接进入研究流程。
    
    Args:
        state: 当前 agent state，包含 user messages 
        config: Runtime 配置，包含模型设置和偏好 
        
    Returns:
      返回一个路由命令：
      - 若需要澄清，跳转到 `__end__` 并向用户返回澄清问题；
      - 若无需澄清，跳转到 `write_research_brief` 继续研究流程（附带确认消息）。
    """
    # Step 1: 检查配置中是否启用了 clarification（澄清） 功能
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # 跳过澄清步骤，直接进入研究流程
        return Command(goto="write_research_brief")
    
    # Step 2: 构建“澄清判定”专用模型（结构化输出 + 自动重试 + 运行配置）
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 配置用于澄清判断的模型：structured output + 自动重试
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser) 
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: 分析是否需要澄清
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: 根据模型判断结果，决定“先问清楚再停”还是“直接进入研究”
    if response.need_clarification:
        # 需要澄清时：往 messages 追加一条 AI 的澄清问题
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # 无需澄清时：往 messages 追加一条 AI 的确认消息
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """将 user messages 转换为结构化的 research brief，并初始化 supervisor
    
       该函数会分析 user messages，生成一个聚焦且可执行的 research brief，用于指导 research supervisor 的后续调度与研究
       与此同时，它还会设置 supervisor 的初始上下文（包括 prompts 和 instructions）。
    
    Args:
        state: 当前 agent state（包含 user messages）
        config: Runtime 配置（主要是模型相关设置）
        
    Returns:
        一个用于跳转到 research supervisor 的 Command (携带已初始化好的 supervisor context)
    """
    # Step 1: 配置一个支持 structured output 的 research model
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 配置 model，生成 structured research question
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: 基于user messages 生成 structured research brief 
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 3: 初始化带有 research brief 和 instructions 的 supervisor
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",  
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """作为 lead research supervisor，负责规划 research strategy 并把任务委派给 researchers
    
       supervisor 会先分析 research brief，再决定如何把问题拆成可执行、可管理的子任务。
       它可以使用 `think_tool` 做策略思考，调用 `ConductResearch` 把任务下发给 sub-researchers，
       当认为信息已足够时调用 `ResearchComplete` 结束研究阶段。

    Args:
        state: 当前 supervisor 的 state（包含 messages 和研究上下文）
        config: Runtime 配置（模型与相关参数）
        
    Returns:
        一个 Command，流程将继续到 `supervisor_tools` 执行 tool calls
    """
    # Step 1: 配置带有可用 tools 的 supervisor model
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 可用的 tools: research delegation, completion signaling, strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # 配置带有 tools, retry logic, model settings 的模型
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: 基于当前上下文生成 supervisor response
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: 更新 state，继续执行 tool call
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """执行 supervisor 发起的 tools（包括 research delegation 和 strategic thinking）
    
       这个函数主要处理三类 supervisor tool call：
       1. `think_tool`：做策略反思（strategic reflection），并继续对话循环
       2. `ConductResearch`：把研究任务委派给 sub-researchers
       3. `ResearchComplete`：表示 research phase 可以结束
    
    Args:
        state: 当前 supervisor state（包含 messages 与迭代计数）
        config: Runtime 配置（研究上限、模型参数等）
        
    Returns:
        一个 Command：要么继续 supervisor loop，要么结束 research phase
    """
    # Step 1: 提取当前 state，并检查是否满足退出条件
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # 定义 research phase 的退出判定标准
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # 若命中任一 termination condition，则直接退出
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: 统一处理本轮所有 tool calls（think_tool + ConductResearch）
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # 处理 think_tool call（strategic reflection）
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # 处理 ConductResearch call（research delegation）
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # 限制并发 research units，避免资源耗尽
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # 并行执行 research tasks
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # 基于 research results 构造 tool messages
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # 对超出并发上限的 research calls 返回错误提示
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # 聚合所有 research results 的原始记录
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # 处理 research 执行过程中的错误
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # 若 token 超限或出现其他错误，则结束当前 research phase
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # Step 3: 返回包含全部 tool results 的 command
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor 子图构建
# 创建用于管理 research delegation 与协调的 supervisor workflow
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# 添加用于研究管理的 supervisor 节点
supervisor_builder.add_node("supervisor", supervisor)           # 主 supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool 执行处理节点

# 定义 supervisor workflow 的边
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# 编译 supervisor 子图，供主 workflow 调用
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """负责在特定 topic 上做聚焦研究的 individual researcher
    
        这个 researcher 会接收 supervisor 下发的具体 research topic，
        并使用可用工具（search、think_tool、MCP tools）收集全面信息。
        它也可以在多次搜索之间调用 think_tool 做策略规划。
    
    Args:
        state: 当前 researcher state（包含 messages 与 topic 上下文）
        config: Runtime 配置（模型参数与工具可用性）
        
    Returns:
        一个 Command，流程将进入 researcher_tools 执行 tool calls
    """
    # Step 1: 加载配置，并校验工具是否可用
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # 获取全部可用 research tools（search、MCP、think_tool）
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: 配置 researcher model 及其工具
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # 准备 system prompt；如果有 MCP 配置/说明，就把 MCP context 一并注入
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # 配置 model（tools, retry logic, settings）
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: 基于 system prompt + 对话历史，生成 researcher 的下一步响应
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: 更新 state，并进入 tool execution 节点
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool 执行辅助函数
async def execute_tool_safely(tool, args, config):
    """安全执行 tool，并带有错误处理。"""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """执行 researcher 调用的 tools，包括 search tools 和 strategic thinking
    
    该函数用于处理多种类型的 researcher tool calls：
    1. think_tool - 记录策略反思，为下一步 research 决策提供依据
    2. Search tools (tavily_search, web_search) - 信息检索
    3. MCP tools - 外部工具集成
    4. ResearchComplete - 表示单个 research 任务已完成
    
    Args:
        state: 当前 researcher state，包含 messages 和 iteration count
        config: Runtime 配置，包含 research 限制和 tool 设置
        
    Returns:
        一个 Command，用于决定：是继续 research loop，还是进入 compression 阶段
    """
    # Step 1: 提取当前 state，并检查 early exit 条件
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # 若本轮没有任何 tool calls（包括 native web search），则提前退出
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # Step 2: 处理其他 tool calls（search、MCP tools 等）
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # 并行执行全部 tool calls
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # 根据执行结果创建 tool messages
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: 检查后置退出条件（tool execution 完成后）
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # 结束当前 research，并进入压缩阶段
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # 携带 tool results 继续 research loop
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """压缩并整合子研究结果，产出结构化的 compressed_research。
    
    该函数会读取 researcher 在多轮 tool calls 中积累的 messages（含 tool 输出与 AI 回复），
    然后让 compression model 在“尽量保留信息”的前提下做清洗与整合。
    
    Args:
        state: 当前 researcher state，包含累计的 researcher_messages
        config: Runtime 配置（compression model 与 token 参数）
        
    Returns:
        dict:
        - compressed_research: 压缩后的研究结果
        - raw_notes: 从 tool/AI 消息提取的原始记录
    """
    # Step 1: 配置 compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: 准备压缩阶段输入消息
    researcher_messages = state.get("researcher_messages", [])
    
    # 添加模式切换指令：从 research mode 转到 compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: 执行压缩（含重试逻辑，重点处理 token 超限）
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # 构造聚焦压缩任务的 system prompt
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # 调用模型执行压缩
            response = await synthesizer_model.ainvoke(messages)
            
            # 从 tool/AI 消息提取 raw notes（用于可追溯性）
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # 返回压缩成功结果
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # 若 token 超限：裁剪旧消息后重试
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # 其他异常：继续重试直到达到上限
            continue
    
    # Step 4: 所有尝试失败时返回兜底结果
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher 子图构建
# 创建“单个 researcher 聚焦研究”的工作流
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# 添加 researcher 子图节点（研究执行 + tool execution + 压缩）
researcher_builder.add_node("researcher", researcher)                 # 主 researcher 节点
researcher_builder.add_node("researcher_tools", researcher_tools)     # tool execution 处理节点
researcher_builder.add_node("compress_research", compress_research)   # 研究结果压缩节点

# 定义 researcher 子图边
researcher_builder.add_edge(START, "researcher")           # researcher 入口
researcher_builder.add_edge("compress_research", END)      # 压缩完成后退出

# 编译 researcher 子图（供 supervisor 并行调用）
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """生成最终综合报告，并在 token 超限时进行重试与截断。
    
    该函数会汇总前面阶段收集到的 notes/findings，调用 final_report_model 生成最终报告。
    如果触发上下文长度限制，会按策略逐步截断 findings 后重试。
    
    Args:
        state: Agent state（包含研究上下文、notes、research_brief 等）
        config: Runtime 配置（报告模型参数与 API key）
        
    Returns:
        dict:
        - final_report: 最终报告文本或错误说明
        - messages: 写回会话的一条 AIMessage
        - notes: 清空后的 notes state
    """
    # Step 1: 提取研究 findings，并准备清理 notes state
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # Step 2: 配置 final report model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: 生成报告（含 token-limit 重试逻辑）
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # 构造包含 research brief / 历史消息 / findings 的完整 prompt
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # 调用模型生成 final report
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # 返回成功结果，并清空 notes
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # token 超限：通过逐步截断 findings 进行重试
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # 第一次重试：确定初始截断阈值
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # 以 token*4 近似字符上限，作为首次截断长度
                    findings_token_limit = model_token_limit * 4
                else:
                    # 后续重试：每轮再缩减 10%
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # 截断 findings 后重试
                findings = findings[:findings_token_limit]
                continue
            else:
                # 非 token-limit 错误：直接返回
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: 重试耗尽后返回失败结果
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# 主 Deep Researcher 图构建
# 从 user input 到 final report 的完整工作流
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# 添加主流程节点
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # 用户澄清阶段
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # 研究简报生成阶段
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # 研究执行（supervisor 子图）
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # 最终报告生成阶段

# 定义主流程顺序边
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # 入口
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # 研究完成后进入报告生成
deep_researcher_builder.add_edge("final_report_generation", END)                   # 最终出口

# 编译完整 deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
