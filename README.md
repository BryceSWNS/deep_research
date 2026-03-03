# 🔬 Open Deep Research

Deep research 是一个简单、可配置的深度研究 Agent，能够兼容多种模型提供方、搜索工具以及 MCP 服务器。其性能与许多主流深度研究 Agent 相当。 ([see Deep Research Bench leaderboard](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)).

### 🔥 Recent Updates

**August 14, 2025**: See our free course [here](https://academy.langchain.com/courses/deep-research-with-langgraph) (and course repo [here](https://github.com/langchain-ai/deep_research_from_scratch)) on building open deep research.

**August 7, 2025**: Added GPT-5 and updated the Deep Research Bench evaluation w/ GPT-5 results.

**August 2, 2025**: Achieved #6 ranking on the [Deep Research Bench Leaderboard](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard) with an overall score of 0.4344. 

**July 30, 2025**: Read about the evolution from our original implementations to the current version in our [blog post](https://rlancemartin.github.io/2025/07/30/bitter_lesson/).

**July 16, 2025**: Read more in our [blog](https://blog.langchain.com/open-deep-research/) and watch our [video](https://www.youtube.com/watch?v=agGiWUpxkhg) for a quick overview.

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

1. Install dependencies:

```bash
uv sync
# or
uv pip install -r pyproject.toml  # 若前面使用 conda 则该步骤使用
```

1. Set up your `.env` file to customize the environment variables (for model selection, search tools, and other configuration settings):

```bash
cp .env.example .env
```

1. Launch agent with the LangGraph server locally:

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

#### 结果


| Name                           | Commit                                                                                                                  | Summarization       | Research                           | Compression    | Total Cost | Total Tokens | RACE Score | Experiment                                                                                                                                                                                                |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------- | ---------------------------------- | -------------- | ---------- | ------------ | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPT-5                          | [ca3951d](https://github.com/langchain-ai/open_deep_research/pull/168/commits)                                          | openai:gpt-4.1-mini | openai:gpt-5                       | openai:gpt-4.1 |            | 204,640,896  | 0.4943     | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-613c-4bda-8bde-f64f0422bbf3/compare?selectedSessions=4d5941c8-69ce-4f3d-8b3e-e3c99dfbd4cc&baseline=undefined) |
| Defaults                       | [6532a41](https://github.com/langchain-ai/open_deep_research/commit/6532a4176a93cc9bb2102b3d825dcefa560c85d9)           | openai:gpt-4.1-mini | openai:gpt-4.1                     | openai:gpt-4.1 | $45.98     | 58,015,332   | 0.4309     | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=cf4355d7-6347-47e2-a774-484f290e79bc&baseline=undefined)                                              |
| Claude Sonnet 4                | [f877ea9](https://github.com/langchain-ai/open_deep_research/pull/163/commits/f877ea93641680879c420ea991e998b47aab9bcc) | openai:gpt-4.1-mini | anthropic:claude-sonnet-4-20250514 | openai:gpt-4.1 | $187.09    | 138,917,050  | 0.4401     | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=04f6002d-6080-4759-bcf5-9a52e57449ea&baseline=undefined)                                              |
| Deep Research Bench Submission | [c0a160b](https://github.com/langchain-ai/open_deep_research/commit/c0a160b57a9b5ecd4b8217c3811a14d8eff97f72)           | openai:gpt-4.1-nano | openai:gpt-4.1                     | openai:gpt-4.1 | $87.83     | 207,005,549  | 0.4344     | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=e6647f74-ad2f-4cb9-887e-acb38b5f73c0&baseline=undefined)                                              |


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

