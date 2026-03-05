# 🔬 Open Deep Research

Deep research 是一个简单、可配置的深度研究 Agent，能够兼容多种模型提供方、搜索工具以及 MCP 服务器。

### 🚀 快速开始

1. 克隆仓库并激活虚拟环境：

```bash
git clone https://github.com/BryceSWNS/deep_research.git
cd open_deep_research
conda create -n deep_research python=3.11 -y # conda 管理
conda activate deep_research
pip install uv

git clone https://github.com/BryceSWNS/deep_research.git
cd open_deep_research
uv venv # uv 管理
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. 安装依赖:

```bash
uv sync
# 或
uv pip install -r pyproject.toml  # 若前面使用 conda 则该步骤使用
```

3. 设置 `.env` 文件，用于自定义环境变量（如模型选择、搜索工具和其他配置项）：

```bash
cp .env.example .env
```

4. 在本地启动 LangGraph server 运行 Agent：

```bash
# 安装依赖并启动 LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

启动后会在浏览器中打开 LangGraph Studio UI。

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

在 `messages` 输入框里输入问题并点击 `Submit`。你也可以在 “Manage Assistants” 标签页中选择不同配置。

### ⚙️ 配置说明

#### LLM :brain:

Open Deep Research 通过 `init_chat_model() API` 支持多种 LLM provider。系统会在不同阶段使用不同模型。更详细的模型字段请查看 `configuration.py`，这些配置也可以在 LangGraph Studio UI 中修改。

- **Summarization**（默认：`deepseek:deepseek-chat`）：总结 search API 返回的结果
- **Research**（默认：`deepseek:deepseek-chat`）：调动 research agent 执行检索和研究
- **Compression**（默认：`deepseek:deepseek-chat`）：压缩子研究结果
- **Final Report Model**（默认：`deepseek:deepseek-reasoner`）：撰写最终报告

> 注意：所选模型需要支持 structured outputs 和 tool calling

#### Search API :mag:

Open Deep Research 支持多种 search tools。默认使用 [Tavily](https://www.tavily.com/) search API，并兼容 MCP，也支持 Anthropic 和 OpenAI 的 native web search。更多细节请查看 [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) 中的 `search_api` 和 `mcp_config` 字段，这些配置也可以在 LangGraph Studio UI 中修改。

#### 其他

更多可定制项请参考 `configuration.py` 中的字段配置。 

### 🚀 部署与使用

#### LangGraph Studio

按照 [quickstart](#-quickstart) 在本地启动 LangGraph 服务，并在 LangGraph Studio 中测试该 Agent。

#### 托管部署

你可以选择部署到 [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 

#### Open Agent Platform

Open Agent Platform（OAP）是一个可视化界面，非常适合让各类用户为自己的 Deep Researcher 配置不同的 MCP 工具和搜索 API 以匹配具体需求。

如果需要部署 OAP 实例，可参考下面的内容。

1. [Deploy Open Agent Platform](https://docs.oap.langchain.com/quickstart)
2. [Add Deep Researcher to OAP](https://docs.oap.langchain.com/setup/agents)
