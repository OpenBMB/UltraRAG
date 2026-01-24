<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/ultrarag_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/ultrarag.svg">
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
<a href="https://modelscope.cn/datasets/UltraRAG/UltraRAG_Benchmark"><b>Dataset</b></a> 
| 
<a href="https://github.com/OpenBMB/UltraRAG/tree/rag-paper-daily/rag-paper-daily"><b>Paper Daily</b></a> 
| 
<a href="./docs/README_zh.md"><b>简体中文</b></a>
|
<b>English</b>
|
</p>

---

*Latest News* 🔥

- [2026.01.23] 🎉 UltraRAG 3.0 Released: Say no to "black box" development—make every line of reasoning logic clearly visible 👉|[📖 Blog](https://github.com/OpenBMB/UltraRAG/blob/page/project/blog/en/ultrarag3_0.md)|
- [2026.01.20] 🎉 AgentCPM-Report Model Released! DeepResearch is finally localized: 8B on-device writing agent AgentCPM-Report is open-sourced 👉 |[🤗 Model](https://huggingface.co/openbmb/AgentCPM-Report)|

<details>
<summary>Previous News</summary>

- [2025.11.11] 🎉 UltraRAG 2.1 Released: Enhanced knowledge ingestion & multimodal support, with a more complete unified evaluation system!
- [2025.09.23] New daily RAG paper digest, updated every day 👉 |[📖 Papers](https://github.com/OpenBMB/UltraRAG/tree/rag-paper-daily/rag-paper-daily)|
- [2025.09.09] Released a Lightweight DeepResearch Pipeline local setup tutorial 👉 |[📺 bilibili](https://www.bilibili.com/video/BV1p8JfziEwM)|[📖 Blog](https://github.com/OpenBMB/UltraRAG/blob/page/project/blog/en/01_build_light_deepresearch.md)|
- [2025.09.01] Released a step-by-step UltraRAG installation and full RAG walkthrough video 👉 |[📺 bilibili](https://www.bilibili.com/video/BV1B9apz4E7K/?share_source=copy_web&vd_source=7035ae721e76c8149fb74ea7a2432710)|[📖 Blog](https://github.com/OpenBMB/UltraRAG/blob/page/project/blog/en/00_Installing_and_Running_RAG.md)|
- [2025.08.28] 🎉 UltraRAG 2.0 Released! UltraRAG 2.0 is fully upgraded: build a high-performance RAG with just a few dozen lines of code, empowering researchers to focus on ideas and innovation! We have preserved the UltraRAG v2 code, which can be viewed at [v2](https://github.com/OpenBMB/UltraRAG/tree/v2).
- [2025.01.23] UltraRAG Released! Enabling large models to better comprehend and utilize knowledge bases. The UltraRAG 1.0 code is still available at [v1](https://github.com/OpenBMB/UltraRAG/tree/v1).

</details>

---

## About UltraRAG

<a href="https://trendshift.io/repositories/18747" target="_blank"><img src="https://trendshift.io/api/badge/repositories/18747" alt="OpenBMB%2FUltraRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

UltraRAG is the first lightweight RAG development framework based on the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/getting-started/intro) architecture design, jointly launched by [THUNLP](https://nlp.csai.tsinghua.edu.cn/) at Tsinghua University, [NEUIR](https://neuir.github.io) at Northeastern University, [OpenBMB](https://www.openbmb.cn/home), and [AI9stars](https://github.com/AI9Stars).

Designed for research exploration and industrial prototyping, UltraRAG standardizes core RAG components (Retriever, Generation, etc.) as independent **MCP Servers**, combined with the powerful workflow orchestration capabilities of the **MCP Client**. Developers can achieve precise orchestration of complex control structures such as conditional branches and loops simply through YAML configuration.

<p align="center">
  <picture>
    <img alt="UltraRAG" src="./docs/architecture.png" width=90%>
  </picture>
</p>

### UltraRAG UI

UltraRAG UI transcends the boundaries of traditional chat interfaces, evolving into a visual RAG Integrated Development Environment (IDE) that combines orchestration, debugging, and demonstration.

The system features a powerful built-in Pipeline Builder that supports bidirectional real-time synchronization between "Canvas Construction" and "Code Editing," allowing for granular online adjustments of pipeline parameters and prompts. Furthermore, it introduces an Intelligent AI Assistant to empower the entire development lifecycle, from pipeline structural design to parameter tuning and prompt generation. Once constructed, logic flows can be converted into interactive dialogue systems with a single click. The system seamlessly integrates Knowledge Base Management components, enabling users to build custom knowledge bases for document Q&A. This truly realizes a one-stop closed loop, spanning from underlying logic construction and data governance to final application deployment.

<!-- <p align="center">
  <picture>
    <img alt="UltraRAG_UI" src="./docs/chat_menu.png" width=80%>
  </picture>
</p> -->


https://github.com/user-attachments/assets/fcf437b7-8b79-42f2-bf4e-e3b7c2a896b9


### Key Highlights

- 🚀 **Low-Code Orchestration of Complex Workflows**  
  - **Inference Orchestration**: Natively supports control structures such as sequential, loop, and conditional branches. Developers only need to write YAML configuration files to implement complex iterative RAG logic in dozens of lines of code.

- ⚡ **Modular Extension and Reproduction**
	- **Atomic Servers**: Based on the MCP architecture, functions are decoupled into independent Servers. New features only need to be registered as function-level Tools to seamlessly integrate into workflows, achieving extremely high reusability.

- 📊 **Unified Evaluation and Benchmark Comparison**  
  - **Research Efficiency**: Built-in standardized evaluation workflows, ready-to-use mainstream research benchmarks. Through unified metric management and baseline integration, significantly improves experiment reproducibility and comparison efficiency.  
	
- ✨ **Rapid Interactive Prototype Generation**  
  - **One-Click Delivery**: Say goodbye to tedious UI development. With just one command, Pipeline logic can be instantly converted into an interactive conversational Web UI, shortening the distance from algorithm to demonstration.  


## Installation

We provide two installation methods: local source code installation (recommended using `uv` for package management) and Docker container deployment

### Method 1: Source Code Installation

We strongly recommend using [uv](https://github.com/astral-sh/uv) to manage Python environments and dependencies, as it can greatly improve installation speed.

**Prepare Environment**

If you haven't installed uv yet, please execute:

```shell
## Direct installation
pip install uv
## Download
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Download Source Code**

```shell
git clone https://github.com/OpenBMB/UltraRAG.git --depth 1
cd UltraRAG
```

**Install Dependencies**

Choose one of the following modes to install dependencies based on your use case:

**A: Create a New Environment** Use `uv sync` to automatically create a virtual environment and synchronize dependencies:

- Core dependencies: If you only need to run basic core functions, such as only using UltraRAG UI:
  ```shell
  uv sync
  ```

- Full installation: If you want to fully experience UltraRAG's retrieval, generation, corpus processing, and evaluation functions, please run:
  ```shell
  uv sync --all-extras
  ```
- On-demand installation: If you only need to run specific modules, keep the corresponding `--extra` as needed, for example:

  ```shell
  uv sync --extra retriever   # Retrieval module only
  uv sync --extra generation  # Generation module only
  ```

Once installed, activate the virtual environment:

```shell
# Windows CMD
.venv\Scripts\activate.bat

# Windows Powershell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

**B: Install into an Existing Environment** To install UltraRAG into your currently active Python environment, use `uv pip`:

```shell
# Core dependencies
uv pip install -e .

# Full installation
uv pip install -e ".[all]"

# On-demand installation
uv pip install -e ".[retriever]"
```

### Method 2: Docker Container Deployment

If you prefer not to configure a local Python environment, you can deploy using Docker.

**Get Code and Images**

```shell
# 1. Clone the repository
git clone https://github.com/OpenBMB/UltraRAG.git --depth 1
cd UltraRAG

# 2. Prepare the image (choose one)
# Option A: Pull from Docker Hub
docker pull hdxin2002/ultrarag:v0.3.0-base-cpu # Base version (CPU)
docker pull hdxin2002/ultrarag:v0.3.0-base-gpu # Base version (GPU)
docker pull hdxin2002/ultrarag:v0.3.0          # Full version (GPU)

# Option B: Build locally
docker build -t ultrarag:v0.3.0 .

# 3. Start container (port 5050 is automatically mapped)
docker run -it --gpus all -p 5050:5050 <docker_image_name>
```

**Start the Container**

```shell
# Start the container (Port 5050 is mapped by default)
docker run -it --gpus all -p 5050:5050 <docker_image_name>
```

Note: After the container starts, UltraRAG UI will run automatically. You can directly access `http://localhost:5050` in your browser to use it.

### Verify Installation

After installation, run the following example command to check if the environment is normal:

```shell
ultrarag run examples/sayhello.yaml
```

If you see the following output, the installation is successful:

```
Hello, UltraRAG v3!
```


## Quick Start

We provide complete tutorial examples from beginner to advanced. Whether you are conducting academic research or building industrial applications, you can find guidance here. Welcome to visit the [Documentation](https://ultrarag.openbmb.cn/pages/en/getting_started/introduction) for more details.

### Research Experiments
Designed for researchers, providing data, experimental workflows, and visualization analysis tools.
- [Getting Started](https://ultrarag.openbmb.cn/pages/en/getting_started/quick_start): Learn how to quickly run standard RAG experimental workflows based on UltraRAG.
- [Evaluation Data](https://ultrarag.openbmb.cn/pages/en/develop_guide/dataset): Download the most commonly used public evaluation datasets in the RAG field and large-scale retrieval corpora, directly for research benchmark testing.
- [Case Analysis](https://ultrarag.openbmb.cn/pages/en/develop_guide/case_study): Provides a visual Case Study interface to deeply track each intermediate output of the workflow, assisting in analysis and error attribution.
- [Code Integration](https://ultrarag.openbmb.cn/pages/en/develop_guide/code_integration): Learn how to directly call UltraRAG components in Python code to achieve more flexible customized development.

### Demo Systems
Designed for developers and end users, providing complete UI interaction and complex application cases.
- [Quick Start](https://ultrarag.openbmb.cn/pages/en/ui/start): Learn how to start UltraRAG UI and familiarize yourself with various advanced configurations in administrator mode.
- [Deployment Guide](https://ultrarag.openbmb.cn/pages/en/ui/prepare): Detailed production environment deployment tutorials, covering the setup of Retriever, Generation models (LLM), and Milvus vector database.
- [Deep Research](https://ultrarag.openbmb.cn/pages/en/demo/deepresearch): Flagship case, deploy a Deep Research Pipeline. Combined with the AgentCPM-Report model, it can automatically perform multi-step retrieval and integration to generate tens of thousands of words of survey reports.

## Contributing

Thanks to the following contributors for their code submissions and testing. We also welcome new members to join us in collectively building a comprehensive RAG ecosystem!

You can contribute by following the standard process: **Fork this repository → Submit Issues → Create Pull Requests (PRs)**.

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
- If you have any questions, feedback, or would like to get in touch, please feel free to reach out to us via email at yanyk.thu@gmail.com

<table>
  <tr>
    <td align="center">
      <img src="./docs/wechat_qr.png" alt="WeChat Group QR Code" width="220"/><br/>
      <b>WeChat Group</b>
    </td>
    <td align="center">
      <img src="./docs/feishu_qr.png" alt="Feishu Group QR Code" width="220"/><br/>
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
