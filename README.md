# 🔬 Open Deep Research

Deep research 是一个简单、可配置的深度研究 Agent，能够兼容多种模型提供方、搜索工具以及 MCP 服务器。其性能与许多主流深度研究 Agent 相当。 ([see Deep Research Bench leaderboard](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)).

### 🚀 Quickstart

1. Clone the repository and activate a virtual environment:

```bash
conda create -n deep_research python=3.11 -y # conda 管理
conda activate deep_research
pip install uv
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research


git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
uv venv # uv 管理
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. 安装依赖:

```bash
uv sync
# or
uv pip install -r pyproject.toml  # 若前面使用 conda 则该步骤使用
```

3. Set up your `.env` file to customize the environment variables (for model selection, search tools, and other configuration settings):

```bash
cp .env.example .env
```

4. Launch agent with the LangGraph server locally:

```bash
# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

This will open the LangGraph Studio UI in your browser.

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Ask a question in the `messages` input field and click `Submit`. Select different configuration in the "Manage Assistants" tab.

### ⚙️ Configurations

#### LLM :brain:

Open Deep Research supports a wide range of LLM providers via the [init_chat_model() API](https://python.langchain.com/docs/how_to/chat_models_universal_init/). It uses LLMs for a few different tasks. See the below model fields in the [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) file for more details. This can be accessed via the LangGraph Studio UI. 

- **Summarization** (default: `openai:gpt-4.1-mini`): Summarizes search API results
- **Research** (default: `openai:gpt-4.1`): Power the search agent
- **Compression** (default: `openai:gpt-4.1`): Compresses research findings
- **Final Report Model** (default: `openai:gpt-4.1`): Write the final report

> Note: the selected model will need to support [structured outputs](https://python.langchain.com/docs/integrations/chat/) and [tool calling](https://python.langchain.com/docs/how_to/tool_calling/).

> Note: For OpenRouter: Follow [this guide](https://github.com/langchain-ai/open_deep_research/issues/75#issuecomment-2811472408) and for local models via Ollama  see [setup instructions](https://github.com/langchain-ai/open_deep_research/issues/65#issuecomment-2743586318).

#### Search API :mag:

Open Deep Research supports a wide range of search tools. By default it uses the [Tavily](https://www.tavily.com/) search API. Has full MCP compatibility and work native web search for Anthropic and OpenAI. See the `search_api` and `mcp_config` fields in the [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) file for more details. This can be accessed via the LangGraph Studio UI. 

#### Other

See the fields in the [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) for various other settings to customize the behavior of Open Deep Research. 

### 📊 Evaluation

Open Deep Research is configured for evaluation with [Deep Research Bench](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard). This benchmark has 100 PhD-level research tasks (50 English, 50 Chinese), crafted by domain experts across 22 fields (e.g., Science & Tech, Business & Finance) to mirror real-world deep-research needs. It has 2 evaluation metrics, but the leaderboard is based on the RACE score. This uses LLM-as-a-judge (Gemini) to evaluate research reports against a golden set of reports compiled by experts across a set of metrics.

#### Usage

> Warning: Running across the 100 examples can cost ~$20-$100 depending on the model selection.

The dataset is available on [LangSmith via this link](https://smith.langchain.com/public/c5e7a6ad-fdba-478c-88e6-3a388459ce8b/d). To kick off evaluation, run the following command:

```bash
# Run comprehensive evaluation on LangSmith datasets
python tests/run_evaluate.py
```

This will provide a link to a LangSmith experiment, which will have a name `YOUR_EXPERIMENT_NAME`. Once this is done, extract the results to a JSONL file that can be submitted to the Deep Research Bench.

```bash
python tests/extract_langsmith_data.py --project-name "YOUR_EXPERIMENT_NAME" --model-name "you-model-name" --dataset-name "deep_research_bench"
```

This creates `tests/expt_results/deep_research_bench_model-name.jsonl` with the required format. Move the generated JSONL file to a local clone of the Deep Research Bench repository and follow their [Quick Start guide](https://github.com/Ayanami0730/deep_research_bench?tab=readme-ov-file#quick-start) for evaluation submission.
                                           |


### 🚀 部署与使用

#### LangGraph Studio

按照 [quickstart](#-quickstart) 在本地启动 LangGraph 服务，并在 LangGraph Studio 中测试该 Agent。

#### 托管部署

你可以轻松部署到 [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 

#### Open Agent Platform

Open Agent Platform（OAP）是一个可视化界面，使非技术用户也能构建和配置自己的 Agent。OAP 非常适合让用户为 Deep Researcher 配置不同的 MCP 工具和搜索 API，以匹配其具体需求和要解决的问题。

我们已将 Open Deep Research 部署到 OAP 的公共演示实例上。你只需添加自己的 API Key，即可亲自体验 Deep Researcher。[可在此访问](https://oap.langchain.com)

你也可以自行部署 OAP 实例，并将自己定制的 Agent（例如 Deep Researcher）发布给用户使用。

1. [Deploy Open Agent Platform](https://docs.oap.langchain.com/quickstart)
2. [Add Deep Researcher to OAP](https://docs.oap.langchain.com/setup/agents)

