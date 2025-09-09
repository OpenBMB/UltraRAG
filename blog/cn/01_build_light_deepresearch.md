
# 基于UltraRAG 本地搭建轻量级DeepResearch

## 什么是DeepResearch

Deep Research（也称为 Agentic Deep Research）是指大语言模型（LLM）协同工具（如搜索、浏览器、代码执行、记忆存储等），以“多轮推理→检索→验证→融合”的闭环方式，完成复杂任务的研究型智能代理。

不同于单次检索的 RAG（Retrieval-Augmented Generation），Deep Research 更像人的专家思路——先制定计划，再不断探索、调整方向、核实信息，最终输出结构完整、有出处的报告。

## 前置准备 

在本次开发中，我们将基于 UltraRAG 框架 完成示例。考虑到大多数小伙伴可能没有算力服务器，我们全程在一台 MacBook Air (M2) 上实现，确保环境轻量、易于复现。

### UltraRAG环境配置

```bash
# 创建并激活 Conda 环境
conda create -n ultrarag python=3.11
conda activate ultrarag

# 克隆项目并进入目录
git clone https://github.com/OpenBMB/UltraRAG.git
cd UltraRAG


# 使用 uv 安装依赖
pip install uv
uv pip install -e .
```

### API准备

- 检索 API：我们采用 [Tavily Web Search](https://www.tavily.com/)，初次注册即可免费获得 1000 次调用额度。
- LLM API：你可以根据自己的习惯选择任意大模型服务。本教程中，我们使用 gpt-5-nano 作为示例

### API设置

我们提供了两种方式传入 API Key：环境变量和显式参数。其中推荐使用 环境变量，更安全，也能避免 API Key 在日志中泄漏。

```bash
export LLM_API_KEY="your llm key"
export TAVILY_API_KEY="your retriever key"
```

## 动手开发

在本示例中，我们将实现一个轻量级的 Deep Research Pipeline。它具备以下基本功能：

- Plan 制定：模型先根据用户问题制定解决方案的计划；
- 子问题生成与检索：将大问题分解为可检索的子问题，并调用 Web 搜索工具获取相关资料；
- 报告整理与填充：逐步完善研究报告的内容；
- 推理与最终生成：在报告完成后，模型给出最终答案。

流程图如下所示：

<p align="left">
  <img src="../image/01/pipe.png" alt="检索" />
</p>

该 pipeline 主要分为两个阶段：

1. **初始化阶段:** 模型会根据用户问题生成一份 plan，并据此构造出初始的报告 page。

2. **迭代填充阶段:**
   
- 系统会检查当前报告 page 是否已填充完整。
- 判断标准为：page 中是否仍存在 "to be filled" 字符串。
- 如果报告尚未完成，模型会结合用户问题、plan 与当前 page，生成一个新的子问题并触发 Web 检索。
- 检索到的文档会被用于更新 page，然后进入下一轮检查。
- 这个过程会持续迭代，直到 page 被填满。
  
最后，模型会基于用户问题与最终报告 page，生成完整的答案。

这个示例的代码实现非常简洁，主要依赖 router 与 prompt tool 的自定义扩展。感兴趣的小伙伴可以直接查看源码。

其中涉及到的 UltraRAG 开发技巧包括：

- [Prompt Tool开发](https://ultrarag.openbmb.cn/pages/cn/tutorials/part_1/prompt)
- [分支判断Tool开发](https://ultrarag.openbmb.cn/pages/cn/tutorials/part_1/router)
- [循环型结构Pipeline](https://ultrarag.openbmb.cn/pages/cn/tutorials/part_2/loop)
- [分支型结构Pipeline](https://ultrarag.openbmb.cn/pages/cn/tutorials/part_2/branch)
- [参数重命名机制](https://ultrarag.openbmb.cn/pages/cn/tutorials/part_2/data_and_params)

以下是完整的 pipeline 定义（可见 examples/light_deepresearch.yaml）：

```yaml
# light deepresearch demo

# MCP Server
servers:
  benchmark: servers/benchmark
  generation: servers/generation
  retriever: servers/retriever
  prompt: servers/prompt
  router: servers/router

# MCP Client Pipeline
pipeline:
- benchmark.get_data
# generate plan
- prompt.webnote_gen_plan
- generation.generate:
    output:
      ans_ls: plan_ls
# Initialize page
- prompt.webnote_init_page
- generation.generate:
    output:
      ans_ls: page_ls
# Iteratively generate sub-questions, retrieve information, and progressively fill the page
- loop:
    times: 10
    steps:
    # Trigger Check: determine whether the page is complete
    - branch:
        router:
        - router.webnote_check_page
        branches:
          # If the page is not complete, continue.
          incomplete:
          # generate sub question
          - prompt.webnote_gen_subq
          - generation.generate:
              output:
                ans_ls: subq_ls
          # retrieve answer
          - retriever.retriever_tavily_search:
              input:
                query_list: subq_ls
              output:
                ret_psg: psg_ls
          # fill page
          - prompt.webnote_fill_page
          - generation.generate:
              output:
                ans_ls: page_ls
          # If the page has been completed, stop.
          complete: []
# generate answer
- prompt.webnote_gen_answer
- generation.generate
```

## 启动

### 构建问题数据

首先，在 data 文件夹下新建一个名为 sample_light_ds.jsonl 的文件，并写入你要研究的问题。例如：

```json
{
    "id": 0, 
    "question": "介绍一下提瓦特大陆", 
    "golden_answers": [], 
    "meta_data": {}
}
```

### 构建参数配置文件

执行以下命令生成 pipeline 对应的参数文件：

```bash
ultrarag build examples/light_deepresearch.yaml
```

此时会在 examples/parameter/ 下生成 light_deepresearch_parameter.yaml。打开该文件并根据实际情况进行修改，例如：

```yaml
benchmark:
  benchmark:
    key_map:
      gt_ls: golden_answers
      q_ls: question
    limit: -1
    name: ds
    path: data/sample_light_ds.jsonl
    seed: 42
    shuffle: false
generation:
  api_key: ''
  base_url: your api url
  model_name: gpt-5-nano
  sampling_params:
    max_tokens: 4096
    temperature: 0.7
prompt:
  webnote_fill_page_template: prompt/webnote_fill_page.jinja
  webnote_gen_answer_template: prompt/webnote_gen_answer.jinja
  webnote_gen_plan_template: prompt/webnote_gen_plan.jinja
  webnote_gen_subq_template: prompt/webnote_gen_subq.jinja
  webnote_init_page_template: prompt/webnote_init_page.jinja
retriever:
  top_k: 5
```

### 设置 API Key

在运行前，请务必设置你的 API Key。虽然也可以通过参数传递，但推荐使用环境变量方式，避免密钥泄漏到日志文件中：

```bash
export LLM_API_KEY="your llm key"
export TAVILY_API_KEY="your retriever key"
```

### 启动

```bash
ultrarag run examples/light_deepresearch.yaml
```

运行完成后，你可以通过 Case Study Viewer 可视化地查看生成内容：

```python
python ./script/case_study.py \
  --data output/memory_ds_light_deepresearch_20250909_152727.json   \
  --host 127.0.0.1 \
  --port 8070 \
  --title "Case Study Viewer"
```

这样即可在浏览器中打开结果页面，直观地分析 pipeline 的执行过程与生成内容。

<p align="left">
  <img src="../image/01/result.png" alt="检索" />
</p>