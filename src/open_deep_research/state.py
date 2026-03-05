"""Deep Research agent 的 Graph state 定义与数据结构。"""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# 结构化输出定义
###################
class ConductResearch(BaseModel):
    """调用该 tool 以对特定主题开展研究"""
    research_topic: str = Field(
            description="需要研究的主题。应为单一主题，并且应以高度详细的方式进行描述（至少一整段）。",
        )

class ResearchComplete(BaseModel):
    """调用该 tool 表示研究阶段完成。"""

class Summary(BaseModel):
    """包含关键发现的研究摘要。"""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """用户澄清请求的结构化模型。"""
    
    need_clarification: bool = Field(
        description="是否需要向用户提出澄清问题。",
    )
    question: str = Field(
        description="用于澄清报告范围的问题。",
    )
    verification: str = Field(
        description="当信息已足够时，确认将开始研究的提示消息。",
    )

class ResearchQuestion(BaseModel):
    """用于指导研究的 research question / brief。"""
    
    research_brief: str = Field(
        description="用于指导后续研究执行的 research question。",
    )


###################
# State 定义
###################

def override_reducer(current_value, new_value):
    """支持在 state 中执行覆盖写入的 reducer。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """输入 state：仅包含 `messages`。"""

class AgentState(MessagesState):
    """主 agent state，包含消息与研究相关数据。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """supervisor 使用的 state（用于管理研究任务）。"""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """单个 researcher 执行研究时使用的 state。"""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """单个 researcher 的输出 state。"""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
