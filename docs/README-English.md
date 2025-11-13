<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../docs/ultrarag_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="../docs/ultrarag.svg">
    <img alt="UltraRAG" src="./docs/ultrarag.svg" width="55%">
  </picture>
</p>

<h3 align="center">
Less Code, Lower Barrier, Faster Deployment
</h3>

<p align="center">
| 
<a href="https://ultrarag.openbmb.cn/pages/en/getting_started/introduction"><b>Documentation</b></a> 
| 
<a href="https://github.com/OpenBMB/UltraRAG/tree/rag-paper-daily/rag-paper-daily"><b>Paper Daily</b></a> 
| 
<a href="../README.md"><b>简体中文</b></a>
|
<b>English</b>
|
</p>

---

*Latest News* 🔥

- **[2025.11.11]** 🎉 UltraRAG 2.1 Released: Enhanced knowledge ingestion & multimodal support, with a more complete unified evaluation system!

<details>
<summary>Previous News</summary>

- [2025.09.23] New daily RAG paper digest, updated every day 👉 |[📖 Papers](https://github.com/OpenBMB/UltraRAG/tree/rag-paper-daily/rag-paper-daily)|
- [2025.09.09] Released a Lightweight DeepResearch Pipeline local setup tutorial 👉 |[📺 bilibili](https://www.bilibili.com/video/BV1p8JfziEwM/?spm_id_from=333.337.search-card.all.click)|[📖 blog](https://github.com/OpenBMB/UltraRAG/blob/page/project/blog/en/01_build_light_deepresearch.md)|
- [2025.09.01] Released a step-by-step UltraRAG installation and full RAG walkthrough video 👉 |[📺 bilibili](https://www.bilibili.com/video/BV1B9apz4E7K/?share_source=copy_web&vd_source=7035ae721e76c8149fb74ea7a2432710)|[📖 blog](https://github.com/OpenBMB/UltraRAG/blob/page/project/blog/en/00_Installing_and_Running_RAG.md)|
- [2025.08.28] 🎉 Released UltraRAG 2.0! UltraRAG 2.0 is fully upgraded: build a high-performance RAG with just a few dozen lines of code, empowering researchers to focus on ideas and innovation!
- [2025.01.23] UltraRAG Released! Enabling large models to better comprehend and utilize knowledge bases. The UltraRAG 1.0 code is still available at [v1](https://github.com/OpenBMB/UltraRAG/tree/v1).

</details>

---
## UltraRAG v2: Accelerating RAG Research

Retrieval-Augmented Generation (RAG) systems are evolving from early-stage simple concatenations of “retrieval + generation” to complex knowledge systems integrating **adaptive knowledge organization**, **multi-turn reasoning**, and **dynamic retrieval** (typical examples include *DeepResearch* and *Search-o1*). However, this increase in complexity imposes high engineering costs on researchers when it comes to **method reproduction** and **rapid iteration of new ideas**.

To address this challenge, [THUNLP](https://nlp.csai.tsinghua.edu.cn/), [NEUIR](https://neuir.github.io), [OpenBMB](https://www.openbmb.cn/home), and [AI9stars](https://github.com/AI9Stars) jointly launched UltraRAG v2— the first RAG framework based on the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/getting-started/intro) architecture design. This design allows researchers to declare complex logic such as sequential, loop, and conditional branching simply by writing YAML files, enabling rapid implementation of multi-stage reasoning systems with minimal code.

Its core ideas are:
- Modular encapsulation: Encapsulate RAG core components as **standardized independent MCP Servers**;
- Flexible invocation and extension: Provide **function-level Tool** interfaces to support flexible function calls and extensions;
- Lightweight workflow orchestration: Use **MCP Client** to build a top-down simplified linkage;

Compared with traditional frameworks, UltraRAG v2 significantly lowers the **technical threshold and learning cost** of complex RAG systems, allowing researchers to focus more on **experimental design and algorithm innovation** rather than lengthy engineering implementations.

## 🌟 Key Highlights

- 🚀 **Low-code construction of complex pipelines**  
  - Natively supports **sequential, loop, and conditional-branch** reasoning.  
  - Complex iterative RAG pipelines can be built with only YAML.

- 🖼️ **Native multimodal support: retrieval, generation & evaluation in one system**  
  - Fully unifies Retriever, Generator, and Evaluator modules.  
  - Supports a closed-loop flow: **local PDF corpus → multimodal retrieval → multimodal generation**.  
  - Significant performance improvements on complex document understanding and VQA.

- ⚡ **Rapid reproduction & flexible extension (MCP-native)**  
  - Every module is an independent MCP Server.  
  - Add new functionalities by simply adding a Tool-level function.  
  - Seamlessly supports external MCP Servers to extend RAG capability.

- 📚 **Automated knowledge ingestion & corpus construction**  
  - Supports PDF, Markdown, HTML, TXT parsing & chunking.  
  - Integrated with **MinerU** for structured extraction + multimodal chunking (text/table/image).  
  - One-click personal or enterprise-level knowledge base construction.

- 🔗 **Unified pipeline for RAG construction & evaluation**  
  - Supports diverse retrieval engines & generation engines.  
  - Built-in standardized evaluation with full visualization.  

- 📊 **Unified evaluation & comparison**  
  - Supports many mainstream research benchmarks.  
  - Continuously integrates new baselines.  
  - Enables systematic comparisons for research.

## The Secret Sauce: MCP Architecture and Native Pipeline Control

In different RAG systems, core capabilities such as retrieval and generation share high functional similarity, but due to diverse implementation strategies by developers, modules often lack unified interfaces, making cross-project reuse difficult. The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/getting-started/intro) is an open protocol that standardizes the way to provide context for large language models (LLMs) and adopts a **Client–Server** architecture, enabling MCP-compliant Server components to be seamlessly reused across different systems.

Inspired by this, UltraRAG v2 is based on the **MCP architecture**, abstracting and encapsulating core functions such as retrieval, generation, and evaluation in RAG systems into independent **MCP Servers**, and invoking them through standardized function-level **Tool interfaces**. This design ensures flexible module function extension and allows new modules to be “hot-plugged” without invasive modifications to global code. In scientific research scenarios, this architecture enables researchers to quickly adapt new models or algorithms with minimal code while maintaining overall system stability and consistency.

<p align="center">
  <picture>
    <img alt="UltraRAG" src="../docs/architecture.png" width=90%>
  </picture>
</p>

Developing complex RAG inference frameworks is significantly challenging. UltraRAG v2’s ability to support complex systems under **low-code** conditions lies in its native support for multi-structured **pipeline workflow control**. Whether sequential, loop, or conditional branching, all control logic can be defined and orchestrated at the YAML level, covering various workflow expression forms needed for complex inference tasks. During runtime, inference workflow scheduling is executed by the built-in **Client**, whose logic is fully described by user-written external **Pipeline YAML scripts**, achieving decoupling from the underlying implementation. Developers can call instructions like loop and step as if using programming language keywords, quickly constructing multi-stage inference workflows in a declarative manner.

By deeply integrating the **MCP architecture** with **native workflow control**, UltraRAG v2 makes building complex RAG systems as natural and efficient as “orchestrating workflows.” Additionally, the framework includes 17 mainstream benchmark tasks and multiple high-quality baselines, combined with a unified evaluation system and knowledge base support, further enhancing system development efficiency and experiment reproducibility.

## Installation

### Create a virtual environment using Conda:

```shell
conda create -n ultrarag python=3.11
conda activate ultrarag
```
Clone the project locally or on a server via git:

```shell
git clone https://github.com/OpenBMB/UltraRAG.git --depth 1
cd UltraRAG
```

We recommend using uv for package management, providing faster and more reliable Python dependency management:

```shell
pip install uv
uv pip install -e .
```

If you prefer pip, you can directly run:

```shell
pip install -e .
```

Verify the installation:

```shell
# On success you'll see 'Hello, UltraRAG 2.0!'
ultrarag run examples/sayhello.yaml
```


[Optional] UltraRAG v2 supports rich Server components; developers can flexibly install dependencies according to actual tasks:

```shell
# Retriever/Reranker Server dependencies:
# infinity
uv pip install infinity_emb
# sentence_transformers
uv pip install sentence_transformers
# openai
uv pip install openai
# bm25
uv pip install bm25s
# faiss (you need to manually compile and install the CPU or GPU version according to your hardware environment)
# CPU version:
uv pip install faiss-cpu
# GPU version (example: CUDA 12.x)
uv pip install faiss-gpu-cu12
# For other CUDA versions, install the corresponding package (e.g., CUDA 11.x uses faiss-gpu-cu11)
# websearch
# exa
uv pip install exa_py
# tavily
uv pip install tavily-python
# One-click installation:
uv pip install -e ".[retriever]"

# Generation Server dependencies:
# vllm
uv pip install vllm
# openai
uv pip install openai
# hf
uv pip install transformers
# One-click installation:
uv pip install -e ".[generation]"

# Corpus Server dependencies:
# chonkie
uv pip install chonkie
# pymupdf
uv pip install pymupdf
# mineru
uv pip install "mineru[core]"
# One-click installation:
uv pip install -e ".[corpus]"

# Install all dependencies:
uv pip install -e ".[all]"
# Or use conda to import the environment:
conda env create -f environment.yml
```

### Build and Run Environment with Docker

(Option 1) Clone the project to your local machine or server via git:

```shell
git clone https://github.com/OpenBMB/UltraRAG.git --depth 1
cd UltraRAG
```

Build the image:

```shell
docker build -t ultrarag:v0.2.1 .
```

Run an interactive environment:

```shell
docker run -it --gpus all ultrarag:v0.2.1 /bin/bash
```

(Option 2) Use the Prebuilt Image

Pull the prebuilt image:

```shell
docker pull hdxin2002/ultrarag:v0.2.1
```

Run an interactive environment:

```shell
docker run -it --gpus all hdxin2002/ultrarag:v0.2.1 /bin/bash
```

Run the following command to verify whether the installation is successful:

```shell
# If successful, it will display the welcome message 'Hello, UltraRAG 2.0!'
ultrarag run examples/sayhello.yaml
```

## Quick Start

We provide end‑to‑end tutorials from basics to advanced. Visit the <a href="https://ultrarag.openbmb.cn/pages/en/getting_started/introduction"><b>documentation</b></a> to get started with UltraRAG v2.

Read the <a href="https://ultrarag.openbmb.cn/pages/en/getting_started/quick_start"><b>Quick Start</b></a> to learn how to run a complete RAG pipeline with UltraRAG.



## Support

UltraRAG v2 ships ready‑to‑use evaluation **datasets** and **large‑scale corpora**, published on [ModelScope](https://modelscope.cn/datasets/UltraRAG/UltraRAG_Benchmark) and [Huggingface](https://huggingface.co/datasets/UltraRAG/UltraRAG_Benchmark).  
Download and use directly — no extra cleaning or conversion needed — and plug into UltraRAG’s evaluation pipelines. You can also customize by following the <a href="https://ultrarag.openbmb.cn/pages/en/develop_guide/dataset"><b>DataFormat</b></a>.

### 1) Supported Datasets

| Task Type        | Dataset Name            | Raw Size                                | Eval Samples |
|:------------------|:------------------------|:-----------------------------------------|:-------------|
| QA               | [NQ](https://huggingface.co/datasets/google-research-datasets/nq_open) | 3,610 | 1,000 |
| QA               | [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) | 11,313 | 1,000 |
| QA               | [PopQA](https://huggingface.co/datasets/akariasai/PopQA) | 14,267 | 1,000 |
| QA               | [AmbigQA](https://huggingface.co/datasets/sewon/ambig_qa) | 2,002 | 1,000 |
| QA               | [MarcoQA](https://huggingface.co/datasets/microsoft/ms_marco/viewer/v2.1/validation) | 55,636 | 1,000 |
| QA               | [WebQuestions](https://huggingface.co/datasets/stanfordnlp/web_questions) | 2,032 | 1,000 |
| VQA              | [MP-DocVQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-MP-DocVQA) | 591 | 591 |
| VQA              | [ChartQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-ChartQA) | 63 | 63 |
| VQA              | [InfoVQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-InfoVQA) | 718 | 718 |
| VQA              | [PlotQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-PlotQA) | 863 | 863 |
| Multi-hop QA     | [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) | 7,405 | 1,000 |
| Multi-hop QA     | [2WikiMultiHopQA](https://www.dropbox.com/scl/fi/heid2pkiswhfaqr5g0piw/data.zip?e=2&file_subpath=%2Fdata&rlkey=ira57daau8lxfj022xvk1irju) | 12,576 | 1,000 |
| Multi-hop QA     | [Musique](https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view) | 2,417 | 1,000 |
| Multi-hop QA     | [Bamboogle](https://huggingface.co/datasets/chiayewken/bamboogle) | 125 | 125 |
| Multi-hop QA     | [StrategyQA](https://huggingface.co/datasets/tasksource/strategy-qa) | 2,290 | 1,000 |
| Multi-hop VQA    | [SlideVQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-SlideVQA) | 556 | 556 |
| Multiple-choice  | [ARC](https://huggingface.co/datasets/allenai/ai2_arc) | 3,548 | 1,000 |
| Multiple-choice  | [MMLU](https://huggingface.co/datasets/cais/mmlu) | 14,042 | 1,000 |
| Multiple-choice VQA | [ArXivQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-ArxivQA) | 816 | 816 |
| Long-form QA     | [ASQA](https://huggingface.co/datasets/din0s/asqa) | 948 | 948 |
| Fact Verification | [FEVER](https://fever.ai/dataset/fever.html) | 13,332 | 1,000 |
| Dialogue         | [WoW](https://huggingface.co/datasets/facebook/kilt_tasks) | 3,054 | 1,000 |
| Slot-filling     | [T-REx](https://huggingface.co/datasets/facebook/kilt_tasks) | 5,000 | 1,000 |

---

### 2) Supported Corpora

| Corpus Name | # Documents |
|:-------------|:-----------|
| Wiki-2018    | 21,015,324 |
| Wiki-2024    | 30,463,973 |
| MP-DocVQA    | 741 |
| ChartQA      | 500 |
| InfoVQA      | 459 |
| PlotQA       | 9,593 |
| SlideVQA     | 1,284 |
| ArXivQA      | 8,066 |

---

### 3) Supported Baselines (continuously updated)

| Baseline | Script |
|:---------|:-------|
| Vanilla LLM   | examples/vanilla_llm.yaml |
| Vanilla RAG   | examples/rag.yaml |
| [IRCoT](https://arxiv.org/abs/2212.10509) | examples/IRCoT.yaml |
| [IterRetGen](https://arxiv.org/abs/2305.15294) | examples/IterRetGen.yaml |
| [RankCoT](https://arxiv.org/abs/2502.17888) | examples/RankCoT.yaml |
| [R1-searcher](https://arxiv.org/abs/2503.05592) | examples/r1_searcher.yaml |
| [Search-o1](https://arxiv.org/abs/2501.05366) | examples/search_o1.yaml |
| [Search-r1](https://arxiv.org/abs/2503.09516) | examples/search_r1.yaml |
| [VisRAG](https://arxiv.org/abs/2410.10594)   | examples/visrag.yaml     |
| [VisRAG 2.0](https://arxiv.org/abs/2510.09733)   | examples/evisrag.yaml     |

## Contributing

Thanks to the following contributors for their code submissions and testing. We also welcome new members to join us in collectively building a comprehensive RAG ecosystem!

You can contribute by following the standard process: **fork this repository, submit issues, and create pull requests (PRs)**.

<a href="https://github.com/OpenBMB/UltraRAG/contributors">
  <img src="https://contrib.rocks/image?repo=OpenBMB/UltraRAG&nocache=true" />
</a>

## Support Us

If you find this repository helpful for your research, please consider giving us a ⭐ to show your support.

<a href="https://star-history.com/#OpenBMB/UltraRAG&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=OpenBMB/UltraRAG&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=OpenBMB/UltraRAG&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=OpenBMB/UltraRAG&type=Date" />
 </picture>
</a>


## Contact Us

- For technical issues and feature requests, please use [GitHub Issues](https://github.com/OpenBMB/UltraRAG/issues).  
- For questions about usage, feedback, or any discussions related to RAG technologies, you are welcome to join our [WeChat group](https://github.com/OpenBMB/UltraRAG/blob/main/docs/wechat_qr.png), [Feishu group](https://github.com/OpenBMB/UltraRAG/blob/main/docs/feishu_qr.png), and [Discord](https://discord.gg/yRFFjjJnnS) to exchange ideas with us.

<table>
  <tr>
    <td align="center">
      <img src="wechat_qr.png" alt="WeChat Group QR Code" width="220"/><br/>
      <b>WeChat Group</b>
    </td>
    <td align="center">
      <img src="feishu_qr.png" alt="Feishu Group QR Code" width="220"/><br/>
      <b>Feishu Group</b>
    </td>
    <td align="center">
      <a href="https://discord.gg/yRFFjjJnnS">
        <img src="https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white" alt="Join Discord"/>
      </a><br/>
      <b>Discord</b>
  </td>
  </tr>
</table>
