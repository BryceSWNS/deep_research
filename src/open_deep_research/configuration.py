"""Open Deep Research 的配置管理。"""

import os
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """可用 search API provider 枚举。"""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """单个 MCP（Model Context Protocol）server 的配置。"""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """MCP 服务器的 URL"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """可供 LLM 使用的 tools 列表"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """MCP 服务器是否需要认证"""
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        optional=True,
    )
    """访问 MCP server 时附带的自定义请求头（例如 Authorization）"""

class Configuration(BaseModel):
    """Deep Research Agent 的主配置类"""
    
    # 通用配置
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "模型进行 structured output 调用时的最大重试次数"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否允许 researcher 在开始研究前向用户提出澄清性问题"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "允许同时运行的最大研究单元数量。这将使 researcher 能够使用多个sub-Agents 并行开展研究。注意：并发数增加可能会触发速率限制"
            }
        }
    )
    # 研究相关配置
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "用于开展研究的 search API。注意：请确保所选的 Researcher Model 支持所选择的 search API",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Research Supervisor 的最大研究迭代次数。即 Research Supervisor 对研究过程进行反思并提出后续问题的次数上限"
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "在单个 researcher 步骤中允许进行的 tool calling 迭代次数上限"
            }
        }
    )
    # 模型相关配置
    summarization_model: str = Field(
        default="deepseek:deepseek-chat",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "deepseek:deepseek-chat",
                "description": "用于对 Tavily 搜索结果进行总结的模型"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "summarization model 允许生成的最大输出 token 数量"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "在进行 summarization 前，网页内容允许的最大字符长度"
            }
        }
    )
    research_model: str = Field(
        default="deepseek:deepseek-chat",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "deepseek:deepseek-chat",
                "description": "用于开展研究的模型。注意：请确保所选的 Researcher Model 支持所选择的 search API"
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "research model 允许生成的最大输出 token 数量"
            }
        }
    )
    compression_model: str = Field(
        default="deepseek:deepseek-chat",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "deepseek:deepseek-chat",
                "description": "用于压缩 sub-Agents 的 research findingss 的模型。注意：请确保所选的 Compression Model 支持所选择的 search API"
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "compression model 允许生成的最大输出 token 数量"
            }
        }
    )
    final_report_model: str = Field(
        default="deepseek:deepseek-reasoner",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "deepseek:deepseek-reasoner",
                "description": "基于所有研究 findings，撰写最终报告的模型"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "final report model 允许生成的最大输出 token 数量"
            }
        }
    )
    # MCP 配置（支持多 server）
    mcp_servers: Optional[List[MCPConfig]] = Field(
        default_factory=lambda: [
            MCPConfig(
                url="https://api.githubcopilot.com",
                tools=["search_repositories", "get_file_contents"],
                auth_required=False,
                headers={"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN', '')}"},
            )
        ],
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP servers 配置（支持多个）。"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default="你可以按需使用可用的 MCP tools 补充信息来源，并与其他工具结果综合分析。",
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "传递给 Agent 的关于其可用的 MCP tools 的任何附加说明"
            }
        }
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 RunnableConfig 创建一个 Configuration 实例"""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic 配置。"""
        
        arbitrary_types_allowed = True
