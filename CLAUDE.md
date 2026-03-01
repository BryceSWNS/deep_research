# Open Deep Research Repository Overview

## Project Description
Open Deep Research 是一个可配置、完全开源的深度研究 Agent，支持多种模型提供方、搜索工具以及 MCP（Model Context Protocol）服务器。它支持并行处理与自动化研究流程，并生成结构化的综合研究报告。

## Repository Structure

### Root Directory
- `README.md` - 完整的项目文档与快速开始指南
- `pyproject.toml` - Python 项目配置与依赖定义
- `langgraph.json` - LangGraph 配置文件，定义主图（graph）入口
- `uv.lock` - UV 包管理器锁文件
- `LICENSE` - MIT 许可证
- `.env.example` - 环境变量模板文件（不纳入版本控制）

### Core Implementation (`src/open_deep_research/`)
- `deep_researcher.py` - 主要的 LangGraph 实现（入口函数：`deep_researcher`）
- `configuration.py` - 配置管理与参数设置
- `state.py` - 图状态定义与数据结构 
- `prompts.py` - 系统提示词与提示模板
- `utils.py` - 工具函数与辅助方法
- `files/` - 研究输出文件与示例文件

### Legacy Implementations (`src/legacy/`)
包含两个较早版本的研究实现：
- `graph.py` - 带有人类参与（human-in-the-loop）的计划-执行工作流
- `multi_agent.py` - Supervisor–Researcher 多 Agent 架构
- `legacy.md` - 旧版本实现说明文档
- `CLAUDE.md` - 旧版本专用 Claude 使用说明
- `tests/` - 旧版本相关测试

### Security (`src/security/`)
- `auth.py` - 用于 LangGraph 部署的身份认证处理模块

### Testing (`tests/`)
- `run_evaluate.py` - 主评测脚本，配置为在 deep research 基准上运行
- `evaluators.py` - 专用评测函数
- `prompts.py` - 评测提示词与评估标准
- `pairwise_evaluation.py` - 对比式评测工具
- `supervisor_parallel_evaluation.py` - 多线程评测实现

### Examples (`examples/`)
- `arxiv.md` - ArXiv 研究示例
- `pubmed.md` - PubMed 研究示例
- `inference-market.md` - 推理市场分析示例

## Key Technologies
- **LangGraph** - 工作流编排与图执行引擎
- **LangChain** - LLM 集成与工具调用框架
- **Multiple LLM Providers** - 支持 OpenAI、Anthropic、Google、Groq、DeepSeek 等模型提供方
- **Search APIs** - 支持 Tavily、OpenAI/Anthropic 原生搜索、DuckDuckGo、Exa
- **MCP Servers** - 通过 Model Context Protocol 扩展能力
## Development Commands
- `uvx langgraph dev` - 启动带 LangGraph Studio 的开发服务器
- `python tests/run_evaluate.py` - 运行完整评测流程
- `ruff check` - 代码静态检查
- `mypy` - 类型检查

## Configuration
所有配置项均可通过以下方式修改：
- 环境变量（`.env` 文件）
- LangGraph Studio Web 界面
- 直接修改配置文件

关键配置包括：模型选择、搜索 API 选择、并发限制，以及 MCP 服务器相关配置。