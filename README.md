<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/ultrarag_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/ultrarag.svg">
    <img alt="UltraRAG" src="./docs/ultrarag.svg" width="55%">
  </picture>
</p>

<h3 align="center">
更少代码，更低门槛，更快实现
</h3>

<p align="center">
| 
<a href="https://ultrarag.openbmb.cn"><b>教程文档</b></a> 
| 
<a href="https://github.com/OpenBMB/UltraRAG/tree/rag-paper-daily/rag-paper-daily"><b>每日论文</b></a> 
| 
<b>简体中文</b>
|
<a href="./docs/README-English.md"><b>English</b></a>
|
</p>

---

*更新日志* 🔥

- [2025.11.11] 🎉 UltraRAG 2.1 更新：强化知识接入与多模态支持，完善统一评估体系！

<details>
<summary>历史更新</summary>

- [2025.09.23] 新增每日 RAG 论文分享，每日更新最新前沿 RAG 工作 👉 |[📖 论文](https://github.com/OpenBMB/UltraRAG/tree/rag-paper-daily/rag-paper-daily)|
- [2025.09.09] 发布轻量级 DeepResearch Pipeline 本地搭建教程 👉 |[📺 bilibili](https://www.bilibili.com/video/BV1p8JfziEwM/?spm_id_from=333.337.search-card.all.click)|[📖 博客](https://github.com/OpenBMB/UltraRAG/blob/page/project/blog/cn/01_build_light_deepresearch.md)|
- [2025.09.01] 发布 UltraRAG 安装与完整 RAG 跑通视频 👉 |[📺 bilibili](https://www.bilibili.com/video/BV1B9apz4E7K/?share_source=copy_web&vd_source=7035ae721e76c8149fb74ea7a2432710)|[📖 博客](https://github.com/OpenBMB/UltraRAG/blob/page/project/blog/cn/00_Installing_and_Running_RAG.md)|
- [2025.08.28] 🎉 发布 UltraRAG 2.0！UltraRAG 2.0 全新升级：几十行代码实现高性能 RAG，让科研专注思想创新！
- [2025.01.23] 发布 UltraRAG！让大模型读懂善用知识库！我们保留了UltraRAG 1.0的代码，可以点击 [v1](https://github.com/OpenBMB/UltraRAG/tree/v1) 查看。

</details>

---

## UltraRAG v2：面向科研的“RAG实验”加速器 

检索增强生成系统（RAG）正从早期“检索+生成”的简单拼接，走向融合 **自适应知识组织**、**多轮推理**、**动态检索** 的复杂知识系统。但这种复杂度的提升，使科研人员在 **方法复现**、**快速迭代新想法** 时，面临着高昂的工程实现成本。

为了解决这一痛点，清华大学 [THUNLP](https://nlp.csai.tsinghua.edu.cn/) 实验室、东北大学 [NEUIR](https://neuir.github.io) 实验室、[OpenBMB](https://www.openbmb.cn/home) 与 [AI9stars](https://github.com/AI9Stars) 联合推出 UltraRAG v2 —— 首个基于 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/getting-started/intro) 架构设计的 RAG 框架。这一设计让科研人员只需编写 YAML 文件，就可以直接声明串行、循环、条件分支等复杂逻辑，从而以极低的代码量快速实现多阶段推理系统。

其核心思路是：
- 组件化封装：将RAG 的核心组件封装为**标准化的独立 MCP Server**；
- 灵活调用与扩展：提供 **函数级 Tool** 接口，支持功能的灵活调用与扩展；
- 轻量流程编排：借助 **MCP Client**，建立自上而下的简洁化链路搭建；

与传统框架相比，UltraRAG v2 显著降低了复杂 RAG 系统的 **技术门槛与学习成本**，让研究者能够将更多精力投入到 **实验设计与算法创新** 上，而不是陷入冗长的工程实现。

## 🌟 核心亮点

- 🚀 **低代码构建复杂 Pipeline**  
  - 原生支持 **串行、循环、条件分支** 等推理控制结构。开发者只需编写 YAML 文件，即可实现几十行代码构建的 **迭代式 RAG 流程**。  

- 🖼️ **原生多模态支持：检索、生成、评估一体化**
	- 统一检索、生成与评估，构建真正意义上的 多模态 RAG 全链路；
	- 实现从 本地 PDF 建库 → 多模态检索 → 多模态生成 的闭环流程，显著提升复杂文档场景下的理解与问答能力。

- ⚡ **快速复现与功能扩展**  
  基于 **MCP 架构**，所有模块均封装为独立、可复用的 **Server**。  
  - 用户可按需自定义 Server 或直接复用现有模块；  
  - 每个 Server 的功能以函数级 **Tool** 注册，新增功能仅需添加一个函数即可接入完整流程；  
  - 同时支持调用 **外部 MCP Server**，轻松扩展 Pipeline 能力与应用场景。 

- 📚 **知识接入与语料构建自动化**
	- 支持 PDF、Markdown、HTML、TXT 等多格式文档解析与分块建库；
	- 与 MinerU 无缝集成，自动完成结构化抽取、多模态切块（文本/表格/图片）；
	- 一键构建个人化与企业级知识库，适用于科研、企业文档、私有知识管理等场景。

- 🔗 统一构建与评估的 RAG 工作流
	- 同时适配多种检索引擎与多种生成推理后端；
	- 内置标准化评估体系，支持全链路可视化调试与结果分析；

- 📊 **统一评测与对比**  
  内置 **标准化评测流程与指标管理**，开箱即用支持多个主流科研 Benchmark。  
  - 持续集成最新基线；  
  - 方便科研人员进行系统性对比与优化实验。  

## 秘诀：MCP 架构与原生流程控制

在不同的 RAG 系统中，检索、生成等核心能力在功能上具有高度相似性，但由于开发者实现策略各异，模块之间往往缺乏统一接口，难以跨项目复用。[Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/getting-started/intro) 作为一种开放协议，规范了为大型语言模型（LLMs）提供上下文的标准方式，并采用 **Client–Server** 架构，使得遵循该协议开发的 Server 组件可以在不同系统间无缝复用。

受此启发，UltraRAG v2 基于 **MCP 架构**，将 RAG 系统中的检索、生成、评测等核心功能抽象并封装为相互独立的 **MCP Server**，并通过标准化的函数级 **Tool 接口**实现调用。这一设计既保证了模块功能扩展的灵活性，又允许新模块以“热插拔”的方式接入，无需对全局代码进行侵入式修改。在科研场景中，这种架构让研究者能够以极低的代码量快速适配新的模型或算法，同时保持整体系统的稳定性与一致性。

<p align="center">
  <picture>
    <img alt="UltraRAG" src="./docs/architecture.png" width=90%>
  </picture>
</p>

复杂 RAG 推理框架的开发具有显著挑战，而 UltraRAG v2 之所以能够在**低代码**条件下支持复杂系统的构建，核心在于其底层对多结构 **Pipeline 流程控制**的原生支持。无论是串行、循环还是条件分支，所有控制逻辑均可在 YAML 层完成定义与调度，覆盖复杂推理任务所需的多种流程表达方式。在实际运行中，推理流程的调度由内置 **Client** 执行，其逻辑完全由用户编写的外部 **Pipeline YAML 脚本** 脚本描述，从而实现与底层实现的解耦。开发者可以像使用编程语言关键字一样调用 loop、step 等指令，以声明的方式快速构建多阶段推理流程。

通过将 **MCP 架构** 与 **原生流程控制**深度融合，UltraRAG v2 让复杂 RAG 系统的搭建像“编排流程”一样自然高效。此外，框架内置多个主流 benchmark 任务与多种高质量 baseline，配合统一的评测体系与知识库支持，进一步提升了系统开发的效率与实验的可复现性。

## 安装

### 使用 Conda 创建虚拟环境：

```shell
conda create -n ultrarag python=3.11
conda activate ultrarag
```
通过 git 克隆项目到本地或服务器：

```shell
git clone https://github.com/OpenBMB/UltraRAG.git --depth 1
cd UltraRAG
```

我们推荐使用 uv 来进行包管理，提供更快、更可靠的 Python 依赖管理体验：

```shell
pip install uv
uv pip install -e .
```

如果您更习惯 pip，也可以直接运行：

```shell
pip install -e .
```

运行以下命令验证安装是否成功：

```shell
# 成功运行显示'Hello, UltraRAG 2.0!' 欢迎语
ultrarag run examples/sayhello.yaml
```


【可选】UltraRAG v2 支持丰富的Server组件，开发者可根据实际任务灵活安装所需依赖：

```shell
# Retriever/Reranker Server依赖：
# infinity
uv pip install infinity_emb
# sentence_transformers
uv pip install sentence_transformers
# openai
uv pip install openai
# bm25
uv pip install bm25s
# faiss（需要根据自己的硬件环境，手动编译安装 CPU 或 GPU 版本的 FAISS）
# CPU版本：
uv pip install faiss-cpu
# GPU 版本（示例：CUDA 12.x）
uv pip install faiss-gpu-cu12
# 其他 CUDA 版本请安装对应的包（例如：CUDA 11.x 使用 faiss-gpu-cu11）
# websearch
# exa
uv pip install exa_py
# tavily
uv pip install tavily-python
# 一键安装：
uv pip install -e ".[retriever]"

# Generation Server依赖：
# vllm
uv pip install vllm
# openai
uv pip install openai
# hf
uv pip install transformers
# 一键安装：
uv pip install -e ".[generation]"

# Corpus Server依赖：
# chonkie
uv pip install chonkie
# pymupdf
uv pip install pymupdf
# mineru
uv pip install "mineru[core]"
# 一键安装：
uv pip install -e ".[corpus]"

# 安装所有依赖：
uv pip install -e ".[all]"
# 或使用conda导入环境：
conda env create -f environment.yml
```

### 使用 Docker 构建运行环境

#### （方式一）本地构建镜像

通过 git 克隆项目到本地或服务器：

```shell
git clone https://github.com/OpenBMB/UltraRAG.git --depth 1
cd UltraRAG
```

构建镜像：

```shell
docker build -t ultrarag:v0.2.1 .
```

运行交互环境：

```shell
docker run -it --gpus all ultrarag:v0.2.1 /bin/bash
```

#### （方式二）使用预构建好的镜像

拉取构建好的镜像：

```shell
docker pull hdxin2002/ultrarag:v0.2.1
```

运行交互环境：

```shell
docker run -it --gpus all hdxin2002/ultrarag:v0.2.1 /bin/bash
```

运行以下命令验证安装是否成功：

```shell
# 成功运行显示'Hello, UltraRAG 2.0!' 欢迎语
ultrarag run examples/sayhello.yaml
```

## 快速开始

我们提供了从入门到进阶的完整教学示例，欢迎访问[教程文档](https://ultrarag.openbmb.cn
)快速上手 UltraRAG v2！

阅读[快速开始](https://ultrarag.openbmb.cn/pages/cn/getting_started/quick_start)，了解如何基于 UltraRAG 运行一个完整的 RAG Pipeline。

## 支持

UltraRAG v2 开箱即用，已在 [ModelScope](https://modelscope.cn/datasets/UltraRAG/UltraRAG_Benchmark) 和 [Huggingface](https://huggingface.co/datasets/UltraRAG/UltraRAG_Benchmark) 上同步发布当前 RAG 领域最常用的 **公开评测数据集**以及**大规模语料库**。
用户可直接下载使用，无需额外清洗或转换，即可与 UltraRAG 的评测管线无缝对接。除此之外还可以参考[数据格式说明](https://ultrarag.openbmb.cn/pages/cn/develop_guide/dataset)，灵活地自定义并添加任意数据集或语料库。

### 1. 支持的数据集

| 任务类型         | 数据集名称           | 原始数据数量                               | 评测采样数量       |
|:------------------|:----------------------|:--------------------------------------------|:--------------------|
| QA               | [NQ](https://huggingface.co/datasets/google-research-datasets/nq_open)                   | 3,610                                      | 1,000              |
| QA               | [TriviaQA](https://nlp.cs.washington.edu/triviaqa/)             | 11,313                                     | 1,000              |
| QA               | [PopQA](https://huggingface.co/datasets/akariasai/PopQA)                | 14,267                                     | 1,000              |
| QA               | [AmbigQA](https://huggingface.co/datasets/sewon/ambig_qa)              | 2,002                                      | 1,000              |
| QA               | [MarcoQA](https://huggingface.co/datasets/microsoft/ms_marco/viewer/v2.1/validation)              | 55,636         | 1,000|
| QA               | [WebQuestions](https://huggingface.co/datasets/stanfordnlp/web_questions)         | 2,032                                      | 1,000              |
| VQA         | [MP-DocVQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-MP-DocVQA)               | 591                        | 591                        |
| VQA         | [ChartQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-ChartQA)               | 63                        | 63                         |
| VQA         | [InfoVQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-InfoVQA)                | 718                         | 718                        |
| VQA         | [PlotQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-PlotQA)                | 863                         | 863                         |
| Multi-hop QA     | [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)             | 7,405                                      | 1,000              |
| Multi-hop QA     | [2WikiMultiHopQA](https://www.dropbox.com/scl/fi/heid2pkiswhfaqr5g0piw/data.zip?e=2&file_subpath=%2Fdata&rlkey=ira57daau8lxfj022xvk1irju)      | 12,576                                     | 1,000              |
| Multi-hop QA     | [Musique](https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view)              | 2,417                                      | 1,000              |
| Multi-hop QA     | [Bamboogle](https://huggingface.co/datasets/chiayewken/bamboogle)            | 125                                        | 125                |
| Multi-hop QA     | [StrategyQA](https://huggingface.co/datasets/tasksource/strategy-qa)          | 2,290                                      | 1,000              |
| Multi-hop VQA         | [SlideVQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-SlideVQA)          | 556                        | 556                       |
| Multiple-choice  | [ARC](https://huggingface.co/datasets/allenai/ai2_arc)                  | 3,548    | 1,000              |
| Multiple-choice  | [MMLU](https://huggingface.co/datasets/cais/mmlu)                 | 14,042                      | 1,000              |
| Multiple-choice VQA    | [ArXivQA](https://huggingface.co/datasets/openbmb/VisRAG-Ret-Test-ArxivQA)                 | 816      | 816                |
| Long-form QA     | [ASQA](https://huggingface.co/datasets/din0s/asqa)                 | 948                                        | 948                |
| Fact-verification| [FEVER](https://fever.ai/dataset/fever.html)                | 13,332    | 1,000              |
| Dialogue         | [WoW](https://huggingface.co/datasets/facebook/kilt_tasks)                  | 3,054                                      | 1,000              |
| Slot-filling     | [T-REx](https://huggingface.co/datasets/facebook/kilt_tasks)                | 5,000                                      | 1,000              |

---

### 2. 支持的语料库

| 语料库名称 | 文档数量     |
|:--------------|:--------------|
| Wiki-2018     | 21,015,324   |
| Wiki-2024     | 30,463,973     |
| MP-DocVQA    | 741   |
| ChartQA     | 500  |
| InfoVQA     | 459   |
| PlotQA     | 9,593   |
| SlideVQA     | 1,284  |
| ArXivQA     | 8,066   |

---

### 3. 支持的基线方法（持续更新）

| 基线名称 | 脚本     |
|:------------|:--------------|
| Vanilla LLM   | examples/vanilla_llm.yaml   |
| Vanilla RAG   | examples/rag.yaml     |
| [IRCoT](https://arxiv.org/abs/2212.10509)   | examples/IRCoT.yaml   |
| [IterRetGen](https://arxiv.org/abs/2305.15294)   | examples/IterRetGen.yaml     |
| [RankCoT](https://arxiv.org/abs/2502.17888)   | examples/RankCoT.yaml   |
| [R1-searcher](https://arxiv.org/abs/2503.05592)   | examples/r1_searcher.yaml     |
| [Search-o1](https://arxiv.org/abs/2501.05366)   | examples/search_o1.yaml   |
| [Search-r1](https://arxiv.org/abs/2503.09516)   | examples/search_r1.yaml     |
| [VisRAG](https://arxiv.org/abs/2410.10594)   | examples/visrag.yaml     |
| [VisRAG 2.0](https://arxiv.org/abs/2510.09733)   | examples/evisrag.yaml     |

## 贡献

感谢以下贡献者在代码提交和测试中的付出。我们也欢迎新的成员加入，共同构建完善的 RAG 生态！

您可以通过以下标准流程来贡献：**Fork 本仓库 → 提交 Issue → 发起 Pull Request (PR)**。

<a href="https://github.com/OpenBMB/UltraRAG/contributors">
  <img src="https://contrib.rocks/image?repo=OpenBMB/UltraRAG&nocache=true" />
</a>

## 支持我们

如果您觉得本项目对您的研究有所帮助，欢迎点亮一颗 ⭐ 来支持我们！

<a href="https://star-history.com/#OpenBMB/UltraRAG&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=OpenBMB/UltraRAG&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=OpenBMB/UltraRAG&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=OpenBMB/UltraRAG&type=Date" />
 </picture>
</a>

## 联系我们

- 关于技术问题及功能请求，请使用 [GitHub Issues](https://github.com/OpenBMB/UltraRAG/issues) 功能。
- 关于使用上的问题、意见以及任何关于 RAG 技术的讨论，欢迎加入我们的[微信群组](https://github.com/OpenBMB/UltraRAG/blob/main/docs/wechat_qr.png)，[飞书群组](https://github.com/OpenBMB/UltraRAG/blob/main/docs/feishu_qr.png)和[discord](https://discord.gg/yRFFjjJnnS)，与我们共同交流。

<table>
  <tr>
    <td align="center">
      <img src="docs/wechat_qr.png" alt="WeChat Group QR Code" width="220"/><br/>
      <b>微信群组</b>
    </td>
    <td align="center">
      <img src="docs/feishu_qr.png" alt="Feishu Group QR Code" width="220"/><br/>
      <b>飞书群组</b>
    </td>
    <td align="center">
      <a href="https://discord.gg/yRFFjjJnnS">
        <img src="https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white" alt="Join Discord"/>
      </a><br/>
      <b>Discord</b>
  </td>
  </tr>
</table>
