# 📚 RAG Paper Daily

### 📅 2025-11-06
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody></tbody></table>

### 📅 2025-11-05
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody></tbody></table>

### 📅 2025-11-04
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody></tbody></table>

### 📅 2025-11-03
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody></tbody></table>

### 📅 2025-11-02
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody></tbody></table>

### 📅 2025-11-01
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2511.00739v1">A CPU-Centric Perspective on Agentic AI</a></td><td><details><summary>展开</summary>Agentic AI frameworks add a decision-making orchestrator embedded with
external tools, including web search, Python interpreter, contextual database,
and others, on top of monolithic LLMs, turning them from passive text oracles
into autonomous problem-solvers that can plan, call tools, remember past steps,
and adapt on the fly.
  This paper aims to characterize and understand the system bottlenecks
introduced by agentic AI workloads from a largely overlooked CPU-centric
perspective. We first systematically characterize Agentic AI on the basis of
orchestrator/decision making component, inference path dynamics and
repetitiveness of the agentic flow which directly influences the system-level
performance. Thereafter, based on the characterization, we choose five
representative agentic AI workloads- Haystack RAG, Toolformer, ChemCrow,
Langchain and SWE-Agent to profile latency, throughput and energy metrics and
demystify the significant impact of CPUs on these metrics relative to GPUs. We
observe that - 1. Tool processing on CPUs can take up to 90.6% of the total
latency; 2. Agentic throughput gets bottlenecked either by CPU factors -
coherence, synchronization and over-subscription of cores or GPU factors - main
memory capacity and bandwidth; \circled{3} CPU dynamic energy consumes up to
44% of the total dynamic energy at large batch sizes. Based on the profiling
insights, we present two key optimizations- 1. CPU and GPU-Aware Micro-batching
(CGAM) and 2. Mixed Agentic Workload Scheduling (MAWS) for homogeneous and
heterogeneous agentic workloads respectively to demonstrate the potential to
improve the performance, efficiency, and scalability of agentic AI. We achieve
up to 2.1x and 1.41x P50 latency speedup compared to the multi-processing
benchmark for homogeneous and heterogeneous agentic workloads respectively.</details></td><td><details><summary>展开</summary>这篇论文从CPU中心视角分析了Agentic AI框架（如Haystack RAG、Langchain等）的系统性能瓶颈，揭示了CPU在工具处理延迟和能耗中的关键作用，并提出了针对同质/异构工作负载的优化方案（CGAM和MAWS），最终实现了显著的延迟降低和效率提升。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2511.00505v2">Zero-RAG: Towards Retrieval-Augmented Generation with Zero Redundant Knowledge</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation has shown remarkable results to address Large
Language Models' hallucinations, which usually uses a large external corpus to
supplement knowledge to LLMs. However, with the development of LLMs, the
internal knowledge of LLMs has expanded significantly, thus causing significant
knowledge redundancy between the external corpus and LLMs. On the one hand, the
indexing cost of dense retrieval is highly related to the corpus size and thus
significant redundant knowledge intensifies the dense retrieval's workload. On
the other hand, the redundant knowledge in the external corpus is not helpful
to LLMs and our exploratory analysis shows that it instead hurts the RAG
performance on those questions which the LLM can answer by itself. To address
these issues, we propose Zero-RAG to tackle these challenges. Specifically, we
first propose the Mastery-Score metric to identify redundant knowledge in the
RAG corpus to prune it. After pruning, answers to "mastered" questions rely
primarily on internal knowledge of the LLM. To better harness the internal
capacity, we propose Query Router and Noise-Tolerant Tuning to avoid the
irrelevant documents' distraction and thus further improve the LLM's
utilization of internal knowledge with pruned corpus. Experimental results show
that Zero-RAG prunes the Wikipedia corpus by 30\% and accelerates the retrieval
stage by 22\%, without compromising RAG's performance.</details></td><td><details><summary>展开</summary>这篇论文探讨了RAG技术中外部知识库与大型语言模型（LLM）内部知识冗余的问题，提出了Zero-RAG方法，通过Mastery-Score指标识别并修剪冗余知识，结合Query Router和Noise-Tolerant Tuning优化LLM对内部知识的利用。实验表明，该方法将维基百科语料库缩减30%，检索速度提升22%，同时保持RAG性能不受影响。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2511.00489v1">ToM: Leveraging Tree-oriented MapReduce for Long-Context Reasoning in Large Language Models</a></td><td><details><summary>展开</summary>Large Language Models (LLMs), constrained by limited context windows, often
face significant performance degradation when reasoning over long contexts. To
address this, Retrieval-Augmented Generation (RAG) retrieves and reasons over
chunks but frequently sacrifices logical coherence due to its reliance on
similarity-based rankings. Similarly, divide-and-conquer frameworks (DCF) split
documents into small chunks for independent reasoning and aggregation. While
effective for local reasoning, DCF struggles to capture long-range dependencies
and risks inducing conflicts by processing chunks in isolation. To overcome
these limitations, we propose ToM, a novel Tree-oriented MapReduce framework
for long-context reasoning. ToM leverages the inherent hierarchical structure
of long documents (e.g., main headings and subheadings) by constructing a
DocTree through hierarchical semantic parsing and performing bottom-up
aggregation. Using a Tree MapReduce approach, ToM enables recursive reasoning:
in the Map step, rationales are generated at child nodes; in the Reduce step,
these rationales are aggregated across sibling nodes to resolve conflicts or
reach consensus at parent nodes. Experimental results on 70B+ LLMs show that
ToM significantly outperforms existing divide-and-conquer frameworks and
retrieval-augmented generation methods, achieving better logical coherence and
long-context reasoning. Our code is available at
https://github.com/gjn12-31/ToM .</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为ToM（Tree-oriented MapReduce）的新框架，用于解决大语言模型（LLMs）在长上下文推理中的性能下降问题。与传统的检索增强生成（RAG）和分治框架（DCF）相比，ToM通过利用长文档的层次结构（如主标题和副标题），构建DocTree并进行自底向上的聚合，从而提高了逻辑连贯性和长上下文推理能力。实验结果表明，ToM在70B+的LLMs上显著优于现有的RAG和DCF方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2511.00340v1">Better Call CLAUSE: A Discrepancy Benchmark for Auditing LLMs Legal Reasoning Capabilities</a></td><td><details><summary>展开</summary>The rapid integration of large language models (LLMs) into high-stakes legal
work has exposed a critical gap: no benchmark exists to systematically
stress-test their reliability against the nuanced, adversarial, and often
subtle flaws present in real-world contracts. To address this, we introduce
CLAUSE, a first-of-its-kind benchmark designed to evaluate the fragility of an
LLM's legal reasoning. We study the capabilities of LLMs to detect and reason
about fine-grained discrepancies by producing over 7500 real-world perturbed
contracts from foundational datasets like CUAD and ContractNLI. Our novel,
persona-driven pipeline generates 10 distinct anomaly categories, which are
then validated against official statutes using a Retrieval-Augmented Generation
(RAG) system to ensure legal fidelity. We use CLAUSE to evaluate leading LLMs'
ability to detect embedded legal flaws and explain their significance. Our
analysis shows a key weakness: these models often miss subtle errors and
struggle even more to justify them legally. Our work outlines a path to
identify and correct such reasoning failures in legal AI.</details></td><td><details><summary>展开</summary>这篇论文介绍了CLAUSE基准，用于评估大语言模型（LLM）在法律推理中的脆弱性，通过生成7500多份扰动合同来测试模型检测细微差异的能力，并使用了RAG系统验证异常类别的法律可信性，揭示了模型在识别和解释法律漏洞方面的不足。</details></td></tr></tbody></table>

### 📅 2025-10-31
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2511.00265v1">AgentBnB: A Browser-Based Cybersecurity Tabletop Exercise with Large Language Model Support and Retrieval-Aligned Scaffolding</a></td><td><details><summary>展开</summary>Traditional cybersecurity tabletop exercises (TTXs) provide valuable training
but are often scripted, resource-intensive, and difficult to scale. We
introduce AgentBnB, a browser-based re-imagining of the Backdoors & Breaches
game that integrates large language model teammates with a Bloom-aligned,
retrieval-augmented copilot (C2D2). The system expands a curated corpus into
factual, conceptual, procedural, and metacognitive snippets, delivering
on-demand, cognitively targeted hints. Prompt-engineered agents employ a
scaffolding ladder that gradually fades as learner confidence grows. In a
solo-player pilot with four graduate students, participants reported greater
intention to use the agent-based version compared to the physical card deck and
viewed it as more scalable, though a ceiling effect emerged on a simple
knowledge quiz. Despite limitations of small sample size, single-player focus,
and narrow corpus, these early findings suggest that large language model
augmented TTXs can provide lightweight, repeatable practice without the
logistical burden of traditional exercises. Planned extensions include
multi-player modes, telemetry-driven coaching, and comparative studies with
larger cohorts.</details></td><td><details><summary>展开</summary>这篇论文介绍了AgentBnB系统，该网络浏览器系统基于大型语言模型和RAG技术（通过检索增强助手C2D2），提供按需的认知目标提示，旨在改进传统的网络安全桌面练习（TTXs）。研究结果表明，这种方法比传统方法更轻量、可扩展且易于重复，尽管样本规模较小且存在一些限制。计划中的扩展包括多人模式和更大规模的比较研究。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.27569v1">MARAG-R1: Beyond Single Retriever via Reinforcement-Learned Multi-Tool Agentic Retrieval</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) excel at reasoning and generation but are
inherently limited by static pretraining data, resulting in factual
inaccuracies and weak adaptability to new information. Retrieval-Augmented
Generation (RAG) addresses this issue by grounding LLMs in external knowledge;
However, the effectiveness of RAG critically depends on whether the model can
adequately access relevant information. Existing RAG systems rely on a single
retriever with fixed top-k selection, restricting access to a narrow and static
subset of the corpus. As a result, this single-retriever paradigm has become
the primary bottleneck for comprehensive external information acquisition,
especially in tasks requiring corpus-level reasoning. To overcome this
limitation, we propose MARAG-R1, a reinforcement-learned multi-tool RAG
framework that enables LLMs to dynamically coordinate multiple retrieval
mechanisms for broader and more precise information access. MARAG-R1 equips the
model with four retrieval tools -- semantic search, keyword search, filtering,
and aggregation -- and learns both how and when to use them through a two-stage
training process: supervised fine-tuning followed by reinforcement learning.
This design allows the model to interleave reasoning and retrieval,
progressively gathering sufficient evidence for corpus-level synthesis.
Experiments on GlobalQA, HotpotQA, and 2WikiMultiHopQA demonstrate that
MARAG-R1 substantially outperforms strong baselines and achieves new
state-of-the-art results in corpus-level reasoning tasks.</details></td><td><details><summary>展开</summary>这篇论文提出了一种基于强化学习的多工具RAG框架MARAG-R1，通过动态协调四种检索工具（语义搜索、关键词搜索、过滤和聚合）来提升大语言模型对外部知识的获取能力，解决了传统单检索器RAG在语料级推理任务中的局限性，并在多个数据集上验证其优于现有方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.27568v1">SIGMA: Search-Augmented On-Demand Knowledge Integration for Agentic Mathematical Reasoning</a></td><td><details><summary>展开</summary>Solving mathematical reasoning problems requires not only accurate access to
relevant knowledge but also careful, multi-step thinking. However, current
retrieval-augmented models often rely on a single perspective, follow
inflexible search strategies, and struggle to effectively combine information
from multiple sources. We introduce SIGMA (Search-Augmented On-Demand Knowledge
Integration for AGentic Mathematical reAsoning), a unified framework that
orchestrates specialized agents to independently reason, perform targeted
searches, and synthesize findings through a moderator mechanism. Each agent
generates hypothetical passages to optimize retrieval for its analytic
perspective, ensuring knowledge integration is both context-sensitive and
computation-efficient. When evaluated on challenging benchmarks such as
MATH500, AIME, and PhD-level science QA GPQA, SIGMA consistently outperforms
both open- and closed-source systems, achieving an absolute performance
improvement of 7.4%. Our results demonstrate that multi-agent, on-demand
knowledge integration significantly enhances both reasoning accuracy and
efficiency, offering a scalable approach for complex, knowledge-intensive
problem-solving. We will release the code upon publication.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为SIGMA的多智能体检索增强框架，用于解决数学推理问题。通过协调专门化的智能体进行独立推理、定向检索和结果合成，SIGMA优化了上下文敏感且高效的知识整合，在多项基准测试中表现优于现有系统，实现了7.4%的绝对性能提升。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.27566v1">Interact-RAG: Reason and Interact with the Corpus, Beyond Black-Box Retrieval</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) has significantly enhanced LLMs by
incorporating external information. However, prevailing agentic RAG approaches
are constrained by a critical limitation: they treat the retrieval process as a
black-box querying operation. This confines agents' actions to query issuing,
hindering its ability to tackle complex information-seeking tasks. To address
this, we introduce Interact-RAG, a new paradigm that elevates the LLM agent
from a passive query issuer into an active manipulator of the retrieval
process. We dismantle the black-box with a Corpus Interaction Engine, equipping
the agent with a set of action primitives for fine-grained control over
information retrieval. To further empower the agent on the entire RAG pipeline,
we first develop a reasoning-enhanced workflow, which enables both zero-shot
execution and the synthesis of interaction trajectories. We then leverage this
synthetic data to train a fully autonomous end-to-end agent via Supervised
Fine-Tuning (SFT), followed by refinement with Reinforcement Learning (RL).
Extensive experiments across six benchmarks demonstrate that Interact-RAG
significantly outperforms other advanced methods, validating the efficacy of
our reasoning-interaction strategy.</details></td><td><details><summary>展开</summary>该论文提出了一种名为Interact-RAG的新范式，通过引入Corpus Interaction Engine和细粒度控制机制，将大语言模型从被动的查询发起者转变为检索过程的主动操控者，并利用增强的工作流程和强化学习优化整个RAG流程，实验证明其在多个基准测试中优于现有方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.27537v1">AstuteRAG-FQA: Task-Aware Retrieval-Augmented Generation Framework for Proprietary Data Challenges in Financial Question Answering</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) shows significant promise in
knowledge-intensive tasks by improving domain specificity, enhancing temporal
relevance, and reducing hallucinations. However, applying RAG to finance
encounters critical challenges: restricted access to proprietary datasets,
limited retrieval accuracy, regulatory constraints, and sensitive data
interpretation. We introduce AstuteRAG-FQA, an adaptive RAG framework tailored
for Financial Question Answering (FQA), leveraging task-aware prompt
engineering to address these challenges. The framework uses a hybrid retrieval
strategy integrating both open-source and proprietary financial data while
maintaining strict security protocols and regulatory compliance. A dynamic
prompt framework adapts in real time to query complexity, improving precision
and contextual relevance. To systematically address diverse financial queries,
we propose a four-tier task classification: explicit factual, implicit factual,
interpretable rationale, and hidden rationale involving implicit causal
reasoning. For each category, we identify key challenges, datasets, and
optimization techniques within the retrieval and generation process. The
framework incorporates multi-layered security mechanisms including differential
privacy, data anonymization, and role-based access controls to protect
sensitive financial information. Additionally, AstuteRAG-FQA implements
real-time compliance monitoring through automated regulatory validation systems
that verify responses against industry standards and legal obligations. We
evaluate three data integration techniques - contextual embedding, small model
augmentation, and targeted fine-tuning - analyzing their efficiency and
feasibility across varied financial environments.</details></td><td><details><summary>展开</summary>AstuteRAG-FQA是一个专为金融问答（FQA）设计的自适应RAG框架，通过混合检索策略、动态提示框架和四层任务分类优化检索与生成过程，并采用多层安全机制和实时合规监测来解决金融领域的独特挑战（如数据敏感性、监管限制等）。研究还比较了三种数据集成技术的效率与可行性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2511.00122v1">Engineering.ai: A Platform for Teams of AI Engineers in Computational Design</a></td><td><details><summary>展开</summary>In modern engineering practice, human engineers collaborate in specialized
teams to design complex products, with each expert completing their respective
tasks while communicating and exchanging results and data with one another.
While this division of expertise is essential for managing multidisciplinary
complexity, it demands substantial development time and cost. Recently, we
introduced OpenFOAMGPT (1.0, 2.0), which functions as an autonomous AI engineer
for computational fluid dynamics, and turbulence.ai, which can conduct
end-to-end research in fluid mechanics draft publications and PhD theses.
Building upon these foundations, we present Engineering.ai, a platform for
teams of AI engineers in computational design. The framework employs a
hierarchical multi-agent architecture where a Chief Engineer coordinates
specialized agents consisting of Aerodynamics, Structural, Acoustic, and
Optimization Engineers, each powered by LLM with domain-specific knowledge.
Agent-agent collaboration is achieved through file-mediated communication for
data provenance and reproducibility, while a comprehensive memory system
maintains project context, execution history, and retrieval-augmented domain
knowledge to ensure reliable decision-making across the workflow. The system
integrates FreeCAD, Gmsh, OpenFOAM, CalculiX, and BPM acoustic analysis,
enabling parallel multidisciplinary simulations while maintaining computational
accuracy. The framework is validated through UAV wing optimization. This work
demonstrates that agentic-AI-enabled AI engineers has the potential to perform
complex engineering tasks autonomously. Remarkably, the automated workflow
achieved a 100% success rate across over 400 parametric configurations, with
zero mesh generation failures, solver convergence issues, or manual
interventions required, validating that the framework is trustworthy.</details></td><td><details><summary>展开</summary>这篇文章介绍了Engineering.ai平台，采用多智能体架构协作完成计算设计任务，其中智能体通过检索增强的领域知识确保决策可靠性，并验证了该系统在无人机翼优化中的成功应用。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.27080v1">Adapting Large Language Models to Emerging Cybersecurity using Retrieval Augmented Generation</a></td><td><details><summary>展开</summary>Security applications are increasingly relying on large language models
(LLMs) for cyber threat detection; however, their opaque reasoning often limits
trust, particularly in decisions that require domain-specific cybersecurity
knowledge. Because security threats evolve rapidly, LLMs must not only recall
historical incidents but also adapt to emerging vulnerabilities and attack
patterns. Retrieval-Augmented Generation (RAG) has demonstrated effectiveness
in general LLM applications, but its potential for cybersecurity remains
underexplored. In this work, we introduce a RAG-based framework designed to
contextualize cybersecurity data and enhance LLM accuracy in knowledge
retention and temporal reasoning. Using external datasets and the
Llama-3-8B-Instruct model, we evaluate baseline RAG, an optimized hybrid
retrieval approach, and conduct a comparative analysis across multiple
performance metrics. Our findings highlight the promise of hybrid retrieval in
strengthening the adaptability and reliability of LLMs for cybersecurity tasks.</details></td><td><details><summary>展开</summary>这篇论文探讨了RAG在网络安全领域的应用，提出了一个基于RAG的框架，旨在利用外部数据集和优化混合检索方法提升大语言模型在网络安全任务中的知识记忆和时间推理能力，并通过实验验证了混合检索在增强模型适应性和可靠性方面的潜力。</details></td></tr></tbody></table>

### 📅 2025-10-30
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.27054v1">LLM-Centric RAG with Multi-Granular Indexing and Confidence Constraints</a></td><td><details><summary>展开</summary>This paper addresses the issues of insufficient coverage, unstable results,
and limited reliability in retrieval-augmented generation under complex
knowledge environments, and proposes a confidence control method that
integrates multi-granularity memory indexing with uncertainty estimation. The
method builds a hierarchical memory structure that divides knowledge
representations into different levels of granularity, enabling dynamic indexing
and retrieval from local details to global context, and thus establishing
closer semantic connections between retrieval and generation. On this basis, an
uncertainty estimation mechanism is introduced to explicitly constrain and
filter low-confidence paths during the generation process, allowing the model
to maintain information coverage while effectively suppressing noise and false
content. The overall optimization objective consists of generation loss,
entropy constraints, and variance regularization, forming a unified confidence
control framework. In the experiments, comprehensive sensitivity tests and
comparative analyses were designed, covering hyperparameters, environmental
conditions, and data structures, to verify the stability and robustness of the
proposed method across different scenarios. The results show that the method
achieves superior performance over existing models in QA accuracy, retrieval
recall, ranking quality, and factual consistency, demonstrating the
effectiveness of combining multi-granularity indexing with confidence control.
This study not only provides a new technical pathway for retrieval-augmented
generation but also offers practical evidence for improving the reliability and
controllability of large models in complex contexts.</details></td><td><details><summary>展开</summary>这篇论文针对复杂知识环境下检索增强生成（RAG）存在的覆盖不足、结果不稳定和可靠性有限等问题，提出了一种融合多粒度记忆索引与不确定性估计的置信度控制方法。该方法通过构建分层记忆结构实现动态知识检索，并结合生成过程中的不确定性约束过滤低置信路径，实验表明其在问答准确性、检索召回率、排序质量和事实一致性上优于现有模型。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.27051v1">Adaptive Data Flywheel: Applying MAPE Control Loops to AI Agent Improvement</a></td><td><details><summary>展开</summary>Enterprise AI agents must continuously adapt to maintain accuracy, reduce
latency, and remain aligned with user needs. We present a practical
implementation of a data flywheel in NVInfo AI, NVIDIA's Mixture-of-Experts
(MoE) Knowledge Assistant serving over 30,000 employees. By operationalizing a
MAPE-driven data flywheel, we built a closed-loop system that systematically
addresses failures in retrieval-augmented generation (RAG) pipelines and
enables continuous learning. Over a 3-month post-deployment period, we
monitored feedback and collected 495 negative samples. Analysis revealed two
major failure modes: routing errors (5.25\%) and query rephrasal errors
(3.2\%). Using NVIDIA NeMo microservices, we implemented targeted improvements
through fine-tuning. For routing, we replaced a Llama 3.1 70B model with a
fine-tuned 8B variant, achieving 96\% accuracy, a 10x reduction in model size,
and 70\% latency improvement. For query rephrasal, fine-tuning yielded a 3.7\%
gain in accuracy and a 40\% latency reduction. Our approach demonstrates how
human-in-the-loop (HITL) feedback, when structured within a data flywheel,
transforms enterprise AI agents into self-improving systems. Key learnings
include approaches to ensure agent robustness despite limited user feedback,
navigating privacy constraints, and executing staged rollouts in production.
This work offers a repeatable blueprint for building robust, adaptive
enterprise AI agents capable of learning from real-world usage at scale.</details></td><td><details><summary>展开</summary>这篇论文介绍了NVIDIA在其企业内部知识助手NVInfo AI中实施的基于MAPE（监测-分析-规划-执行）驱动的数据飞轮系统，通过闭环反馈持续优化检索增强生成（RAG）管道的性能。文章重点分析了RAG流程中路由错误和查询重述错误两大故障模式，并利用微调技术显著提升了模型效能（如用8B模型替代70B模型实现96%准确率），最终构建了一个可自我迭代的企业级AI代理框架，同时探讨了隐私约束和分阶段部署等实践经验。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.26941v1">LLM-based Multi-class Attack Analysis and Mitigation Framework in IoT/IIoT Networks</a></td><td><details><summary>展开</summary>The Internet of Things has expanded rapidly, transforming communication and
operations across industries but also increasing the attack surface and
security breaches. Artificial Intelligence plays a key role in securing IoT,
enabling attack detection, attack behavior analysis, and mitigation suggestion.
Despite advancements, evaluations remain purely qualitative, and the lack of a
standardized, objective benchmark for quantitatively measuring AI-based attack
analysis and mitigation hinders consistent assessment of model effectiveness.
In this work, we propose a hybrid framework combining Machine Learning (ML) for
multi-class attack detection with Large Language Models (LLMs) for attack
behavior analysis and mitigation suggestion. After benchmarking several ML and
Deep Learning (DL) classifiers on the Edge-IIoTset and CICIoT2023 datasets, we
applied structured role-play prompt engineering with Retrieval-Augmented
Generation (RAG) to guide ChatGPT-o3 and DeepSeek-R1 in producing detailed,
context-aware responses. We introduce novel evaluation metrics for quantitative
assessment to guide us and an ensemble of judge LLMs, namely ChatGPT-4o,
DeepSeek-V3, Mixtral 8x7B Instruct, Gemini 2.5 Flash, Meta Llama 4, TII Falcon
H1 34B Instruct, xAI Grok 3, and Claude 4 Sonnet, to independently evaluate the
responses. Results show that Random Forest has the best detection model, and
ChatGPT-o3 outperformed DeepSeek-R1 in attack analysis and mitigation.</details></td><td><details><summary>展开</summary>这篇论文提出了一种结合机器学习和大语言模型（LLMs）的混合框架，用于物联网（IoT）安全中的多类攻击检测、行为分析和缓解建议，并应用了检索增强生成（RAG）技术来优化LLMs的上下文感知响应，同时引入了新的评估指标和多个LLM法官进行独立评估。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.26457v1">SecureReviewer: Enhancing Large Language Models for Secure Code Review through Secure-aware Fine-tuning</a></td><td><details><summary>展开</summary>Identifying and addressing security issues during the early phase of the
development lifecycle is critical for mitigating the long-term negative impacts
on software systems. Code review serves as an effective practice that enables
developers to check their teammates' code before integration into the codebase.
To streamline the generation of review comments, various automated code review
approaches have been proposed, where LLM-based methods have significantly
advanced the capabilities of automated review generation. However, existing
models primarily focus on general-purpose code review, their effectiveness in
identifying and addressing security-related issues remains underexplored.
Moreover, adapting existing code review approaches to target security issues
faces substantial challenges, including data scarcity and inadequate evaluation
metrics. To address these limitations, we propose SecureReviewer, a new
approach designed for enhancing LLMs' ability to identify and resolve
security-related issues during code review. Specifically, we first construct a
dataset tailored for training and evaluating secure code review capabilities.
Leveraging this dataset, we fine-tune LLMs to generate code review comments
that can effectively identify security issues and provide fix suggestions with
our proposed secure-aware fine-tuning strategy. To mitigate hallucination in
LLMs and enhance the reliability of their outputs, we integrate the RAG
technique, which grounds the generated comments in domain-specific security
knowledge. Additionally, we introduce SecureBLEU, a new evaluation metric
designed to assess the effectiveness of review comments in addressing security
issues. Experimental results demonstrate that SecureReviewer outperforms
state-of-the-art baselines in both security issue detection accuracy and the
overall quality and practical utility of generated review comments.</details></td><td><details><summary>展开</summary>本文提出SecureReviewer方法，通过构建安全代码审查数据集、微调LLM生成安全审查意见，并引入RAG技术增强领域安全知识参考以减少幻觉，同时设计SecureBLEU评估指标，显著提升了代码安全问题的检测精度与评论质量。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.26345v1">MisSynth: Improving MISSCI Logical Fallacies Classification with Synthetic Data</a></td><td><details><summary>展开</summary>Health-related misinformation is very prevalent and potentially harmful. It
is difficult to identify, especially when claims distort or misinterpret
scientific findings. We investigate the impact of synthetic data generation and
lightweight fine-tuning techniques on the ability of large language models
(LLMs) to recognize fallacious arguments using the MISSCI dataset and
framework. In this work, we propose MisSynth, a pipeline that applies
retrieval-augmented generation (RAG) to produce synthetic fallacy samples,
which are then used to fine-tune an LLM model. Our results show substantial
accuracy gains with fine-tuned models compared to vanilla baselines. For
instance, the LLaMA 3.1 8B fine-tuned model achieved an over 35% F1-score
absolute improvement on the MISSCI test split over its vanilla baseline. We
demonstrate that introducing synthetic fallacy data to augment limited
annotated resources can significantly enhance zero-shot LLM classification
performance on real-world scientific misinformation tasks, even with limited
computational resources. The code and synthetic dataset are available on
https://github.com/mxpoliakov/MisSynth.</details></td><td><details><summary>展开</summary>该论文提出MisSynth方法，利用检索增强生成（RAG）生成虚假论点合成数据，并通过轻量微调提升大语言模型在科学谬误识别任务中的性能（如LLaMA 3.1 8B模型F1值提升35%），解决了健康领域错误信息检测中标注数据匮乏的问题。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.26309v1">GraphCompliance: Aligning Policy and Context Graphs for LLM-Based Regulatory Compliance</a></td><td><details><summary>展开</summary>Compliance at web scale poses practical challenges: each request may require
a regulatory assessment. Regulatory texts (e.g., the General Data Protection
Regulation, GDPR) are cross-referential and normative, while runtime contexts
are expressed in unstructured natural language. This setting motivates us to
align semantic information in unstructured text with the structured, normative
elements of regulations. To this end, we introduce GraphCompliance, a framework
that represents regulatory texts as a Policy Graph and runtime contexts as a
Context Graph, and aligns them. In this formulation, the policy graph encodes
normative structure and cross-references, whereas the context graph formalizes
events as subject-action-object (SAO) and entity-relation triples. This
alignment anchors the reasoning of a judge large language model (LLM) in
structured information and helps reduce the burden of regulatory interpretation
and event parsing, enabling a focus on the core reasoning step. In experiments
on 300 GDPR-derived real-world scenarios spanning five evaluation tasks,
GraphCompliance yields 4.1-7.2 percentage points (pp) higher micro-F1 than
LLM-only and RAG baselines, with fewer under- and over-predictions, resulting
in higher recall and lower false positive rates. Ablation studies indicate
contributions from each graph component, suggesting that structured
representations and a judge LLM are complementary for normative reasoning.</details></td><td><details><summary>展开</summary>这篇文章介绍了GraphCompliance框架，通过将法规文本和运行时上下文分别表示为政策图和上下文图并对其进行对齐，以增强大型语言模型在规范性推理中的表现。实验表明，与纯LLM和RAG基线相比，GraphCompliance在多个任务中展现出更高的性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.26242v1">Retrieval Augmented Generation-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles</a></td><td><details><summary>展开</summary>With increasing urban traffic complexity, Traffic Signal Control (TSC) is
essential for optimizing traffic flow and improving road safety. Large Language
Models (LLMs) emerge as promising approaches for TSC. However, they are prone
to hallucinations in emergencies, leading to unreliable decisions that may
cause substantial delays for emergency vehicles. Moreover, diverse intersection
types present substantial challenges for traffic state encoding and
cross-intersection training, limiting generalization across heterogeneous
intersections. Therefore, this paper proposes Retrieval Augmented Generation
(RAG)-enhanced distributed LLM agents with Emergency response for Generalizable
TSC (REG-TSC). Firstly, this paper presents an emergency-aware reasoning
framework, which dynamically adjusts reasoning depth based on the emergency
scenario and is equipped with a novel Reviewer-based Emergency RAG (RERAG) to
distill specific knowledge and guidance from historical cases, enhancing the
reliability and rationality of agents' emergency decisions. Secondly, this
paper designs a type-agnostic traffic representation and proposes a
Reward-guided Reinforced Refinement (R3) for heterogeneous intersections. R3
adaptively samples training experience from diverse intersections with
environment feedback-based priority and fine-tunes LLM agents with a designed
reward-weighted likelihood loss, guiding REG-TSC toward high-reward policies
across heterogeneous intersections. On three real-world road networks with 17
to 177 heterogeneous intersections, extensive experiments show that REG-TSC
reduces travel time by 42.00%, queue length by 62.31%, and emergency vehicle
waiting time by 83.16%, outperforming other state-of-the-art methods.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为REG-TSC的交通信号控制系统，通过结合检索增强生成（RAG）技术和大型语言模型（LLM）来优化应急响应和异构路口的泛化能力。论文设计了紧急感知推理框架（RERAG）和奖励引导的强化细化方法（R3），显著减少了交通时间、排队长度和紧急车辆等待时间，并在多种现实路网中验证了其优越性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.26205v2">Towards Global Retrieval Augmented Generation: A Benchmark for Corpus-Level Reasoning</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) has emerged as a leading approach to
reducing hallucinations in large language models (LLMs). Current RAG evaluation
benchmarks primarily focus on what we call local RAG: retrieving relevant
chunks from a small subset of documents to answer queries that require only
localized understanding within specific text chunks. However, many real-world
applications require a fundamentally different capability -- global RAG --
which involves aggregating and analyzing information across entire document
collections to derive corpus-level insights (for example, "What are the top 10
most cited papers in 2023?"). In this paper, we introduce GlobalQA -- the first
benchmark specifically designed to evaluate global RAG capabilities, covering
four core task types: counting, extremum queries, sorting, and top-k
extraction. Through systematic evaluation across different models and
baselines, we find that existing RAG methods perform poorly on global tasks,
with the strongest baseline achieving only 1.51 F1 score. To address these
challenges, we propose GlobalRAG, a multi-tool collaborative framework that
preserves structural coherence through chunk-level retrieval, incorporates
LLM-driven intelligent filters to eliminate noisy documents, and integrates
aggregation modules for precise symbolic computation. On the Qwen2.5-14B model,
GlobalRAG achieves 6.63 F1 compared to the strongest baseline's 1.51 F1,
validating the effectiveness of our method.</details></td><td><details><summary>展开</summary>这篇论文提出了一种新的评估基准GlobalQA，用于测试“全局RAG”能力，即从整个文档集合中聚合信息以回答需要综合分析的问题（如统计或排序任务）。研究发现现有RAG方法在全局任务上表现不佳，并提出了GlobalRAG框架（结合分块检索、智能过滤和聚合模块），显著提升了Qwen2.5-14B模型在全局任务上的性能（F1分数从1.51提升至6.63）。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.26130v2">Beyond Synthetic Benchmarks: Evaluating LLM Performance on Real-World Class-Level Code Generation</a></td><td><details><summary>展开</summary>Large language models (LLMs) have demonstrated strong performance on
function-level code generation benchmarks, yet real-world software development
increasingly demands class-level implementations that integrate multiple
methods, attributes, and dependencies within authentic project contexts. This
gap between benchmark performance and practical utility raises critical
questions about LLMs' readiness for production code assistance, particularly
regarding their ability to generalize across familiar and novel codebases.
  We introduce a benchmark derived from real-world open-source repositories,
comprising classes divided into seen and unseen partitions to evaluate
generalization under practical conditions. We systematically examine how input
specification completeness and retrieval-augmented generation affect
class-level correctness across multiple state-of-the-art LLMs.
  Our evaluation reveals a substantial performance gap: while LLMs achieve 84
to 89% correctness on synthetic benchmarks, they attain only 25 to 34% on
real-world class tasks, with minimal distinction between familiar and novel
codebases. Comprehensive documentation provides marginal improvements (1 to
3%), whereas retrieval augmentation yields greater gains (4 to 7%) by supplying
concrete implementation patterns. Error analysis identifies AttributeError,
TypeError, and AssertionError as dominant failure modes, with distinct patterns
between synthetic and real-world scenarios.
  These findings provide actionable insights for enhancing context modelling,
documentation strategies, and retrieval integration in production code
assistance tools.</details></td><td><details><summary>展开</summary>本文研究了大型语言模型（LLMs）在真实世界类级别代码生成任务中的表现，发现其在合成基准测试和实际代码库间存在显著性能差距，并系统评估了输入规范完整性和检索增强生成（RAG）对模型性能的影响。结果表明，RAG通过提供具体实现模式能显著提升模型生成正确性（4-7%），同时错误分析揭示了主要失败模式及合成与真实场景的差异。研究为生产代码辅助工具中的上下文建模、文档策略和检索集成提供了改进方向。</details></td></tr></tbody></table>

### 📅 2025-10-29
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.25724v1">BambooKG: A Neurobiologically-inspired Frequency-Weight Knowledge Graph</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation allows LLMs to access external knowledge,
reducing hallucinations and ageing-data issues. However, it treats retrieved
chunks independently and struggles with multi-hop or relational reasoning,
especially across documents. Knowledge graphs enhance this by capturing the
relationships between entities using triplets, enabling structured, multi-chunk
reasoning. However, these tend to miss information that fails to conform to the
triplet structure. We introduce BambooKG, a knowledge graph with
frequency-based weights on non-triplet edges which reflect link strength,
drawing on the Hebbian principle of "fire together, wire together". This
decreases information loss and results in improved performance on single- and
multi-hop reasoning, outperforming the existing solutions.</details></td><td><details><summary>展开</summary>这篇论文探讨了RAG技术的局限性（如独立处理检索片段、多跳推理困难），并提出了一种改进方法BambooKG——通过基于频率加权的非三元组边增强知识图谱结构，减少信息损失并提升单/多跳推理性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.25718v1">Retrieval-Augmented Search for Large-Scale Map Collections with ColPali</a></td><td><details><summary>展开</summary>Multimodal approaches have shown great promise for searching and navigating
digital collections held by libraries, archives, and museums. In this paper, we
introduce map-RAS: a retrieval-augmented search system for historic maps. In
addition to introducing our framework, we detail our publicly-hosted demo for
searching 101,233 map images held by the Library of Congress. With our system,
users can multimodally query the map collection via ColPali, summarize search
results using Llama 3.2, and upload their own collections to perform
inter-collection search. We articulate potential use cases for archivists,
curators, and end-users, as well as future work with our system in both machine
learning and the digital humanities. Our demo can be viewed at:
http://www.mapras.com.</details></td><td><details><summary>展开</summary>本文介绍了map-RAS，一个针对历史地图的检索增强搜索系统，结合多模态查询（如ColPali）、大模型（Llama 3.2）生成摘要及跨馆藏搜索功能，并提供了美国国会图书馆10万+地图的公开演示，探讨了其在数字人文领域的应用前景。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.25621v1">FARSIQA: Faithful and Advanced RAG System for Islamic Question Answering</a></td><td><details><summary>展开</summary>The advent of Large Language Models (LLMs) has revolutionized Natural
Language Processing, yet their application in high-stakes, specialized domains
like religious question answering is hindered by challenges like hallucination
and unfaithfulness to authoritative sources. This issue is particularly
critical for the Persian-speaking Muslim community, where accuracy and
trustworthiness are paramount. Existing Retrieval-Augmented Generation (RAG)
systems, relying on simplistic single-pass pipelines, fall short on complex,
multi-hop queries requiring multi-step reasoning and evidence aggregation. To
address this gap, we introduce FARSIQA, a novel, end-to-end system for Faithful
Advanced Question Answering in the Persian Islamic domain. FARSIQA is built
upon our innovative FAIR-RAG architecture: a Faithful, Adaptive, Iterative
Refinement framework for RAG. FAIR-RAG employs a dynamic, self-correcting
process: it adaptively decomposes complex queries, assesses evidence
sufficiency, and enters an iterative loop to generate sub-queries,
progressively filling information gaps. Operating on a curated knowledge base
of over one million authoritative Islamic documents, FARSIQA demonstrates
superior performance. Rigorous evaluation on the challenging IslamicPCQA
benchmark shows state-of-the-art performance: the system achieves a remarkable
97.0% in Negative Rejection - a 40-point improvement over baselines - and a
high Answer Correctness score of 74.3%. Our work establishes a new standard for
Persian Islamic QA and validates that our iterative, adaptive architecture is
crucial for building faithful, reliable AI systems in sensitive domains.</details></td><td><details><summary>展开</summary>该论文提出FARSIQA系统，基于创新的FAIR-RAG架构（一种忠实、自适应、迭代优化的RAG框架），通过动态分解复杂查询、评估证据充分性并迭代生成子查询来解决波斯伊斯兰领域多跳问答问题，在权威知识库上实现了97%的负例拒绝率和74.3%答案正确率，显著提升了专业领域问答的可靠性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.25518v1">Retrieval Augmented Generation (RAG) for Fintech: Agentic Design and Evaluation</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) systems often face limitations in
specialized domains such as fintech, where domain-specific ontologies, dense
terminology, and acronyms complicate effective retrieval and synthesis. This
paper introduces an agentic RAG architecture designed to address these
challenges through a modular pipeline of specialized agents. The proposed
system supports intelligent query reformulation, iterative sub-query
decomposition guided by keyphrase extraction, contextual acronym resolution,
and cross-encoder-based context re-ranking. We evaluate our approach against a
standard RAG baseline using a curated dataset of 85 question--answer--reference
triples derived from an enterprise fintech knowledge base. Experimental results
demonstrate that the agentic RAG system outperforms the baseline in retrieval
precision and relevance, albeit with increased latency. These findings suggest
that structured, multi-agent methodologies offer a promising direction for
enhancing retrieval robustness in complex, domain-specific settings.</details></td><td><details><summary>展开</summary>这篇论文提出了一种面向金融科技等专业领域的多智能体RAG架构，通过模块化智能体实现查询重构、子查询分解、术语解析和上下文重排序，实验表明其在检索精度和相关性上优于基线RAG系统，但存在延迟增加的问题。</details></td></tr></tbody></table>

### 📅 2025-10-28
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.24652v1">Optimizing Retrieval for RAG via Reinforced Contrastive Learning</a></td><td><details><summary>展开</summary>As retrieval-augmented generation (RAG) becomes increasingly widespread, the
role of information retrieval (IR) is shifting from retrieving information for
human users to retrieving contextual knowledge for artificial intelligence (AI)
systems, where relevance becomes difficult to define or annotate beforehand. To
address this challenge, we propose R3, a Retrieval framework optimized for RAG
through trialand-feedback Reinforced contrastive learning. Unlike prior
approaches that rely on annotated or synthetic data for supervised fine-tuning,
R3 enables the retriever to dynamically explore and optimize relevance within
the RAG environment. During training, the retrieved results interact with the
environment to produce contrastive signals that automatically guide the
retriever's self-improvement. Extensive experiments across diverse tasks
demonstrate that R3 improves RAG performance by 5.2% over the original
retriever and surpasses state-of-the-art retrievers by 4.9%, while achieving
comparable results to LLM-augmented retrieval and RAG systems built on
post-trained or instruction-tuned LLMs. It is both efficient and practical,
requiring only 4 GPUs and completing training within a single day.</details></td><td><details><summary>展开</summary>本文提出了一种名为R3的检索框架，通过基于试错反馈的强化对比学习优化RAG中的检索过程，能够动态探索和优化相关性，无需依赖预标注或合成数据。实验表明，R3显著提升了RAG性能，优于现有检索方法，且训练效率高（仅需4块GPU和1天时间）。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.24476v1">Mitigating Hallucination in Large Language Models (LLMs): An Application-Oriented Survey on RAG, Reasoning, and Agentic Systems</a></td><td><details><summary>展开</summary>Hallucination remains one of the key obstacles to the reliable deployment of
large language models (LLMs), particularly in real-world applications. Among
various mitigation strategies, Retrieval-Augmented Generation (RAG) and
reasoning enhancement have emerged as two of the most effective and widely
adopted approaches, marking a shift from merely suppressing hallucinations to
balancing creativity and reliability. However, their synergistic potential and
underlying mechanisms for hallucination mitigation have not yet been
systematically examined. This survey adopts an application-oriented perspective
of capability enhancement to analyze how RAG, reasoning enhancement, and their
integration in Agentic Systems mitigate hallucinations. We propose a taxonomy
distinguishing knowledge-based and logic-based hallucinations, systematically
examine how RAG and reasoning address each, and present a unified framework
supported by real-world applications, evaluations, and benchmarks.</details></td><td><details><summary>展开</summary>这篇论文探讨了大型语言模型（LLMs）中的幻觉问题，重点分析了检索增强生成（RAG）和推理增强作为缓解幻觉的有效策略，并研究了它们协同作用的潜力及其在智能代理系统中的整合。文章提出了基于知识和逻辑的幻觉分类，系统性地评估了RAG和推理方法如何应对各类幻觉，并通过实际应用和基准测试提供了一个统一框架。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.24469v1">Iterative Critique-Refine Framework for Enhancing LLM Personalization</a></td><td><details><summary>展开</summary>Personalized text generation requires models not only to produce coherent
text but also to align with a target user's style, tone, and topical focus.
Existing retrieval-augmented approaches such as LaMP and PGraphRAG enrich
profiles with user and neighbor histories, but they stop at generation and
often yield outputs that drift in tone, topic, or style. We present PerFine, a
unified, training-free critique-refine framework that enhances personalization
through iterative, profile-grounded feedback. In each iteration, an LLM
generator produces a draft conditioned on the retrieved profile, and a critic
LLM - also conditioned on the same profile - provides structured feedback on
tone, vocabulary, sentence structure, and topicality. The generator then
revises, while a novel knockout strategy retains the stronger draft across
iterations. We further study additional inference-time strategies such as
Best-of-N and Topic Extraction to balance quality and efficiency. Across Yelp,
Goodreads, and Amazon datasets, PerFine consistently improves personalization
over PGraphRAG, with GEval gains of +7-13%, steady improvements over 3-5
refinement iterations, and scalability with increasing critic size. These
results highlight that post-hoc, profile-aware feedback offers a powerful
paradigm for personalized LLM generation that is both training-free and
model-agnostic.</details></td><td><details><summary>展开</summary>该论文提出了一种名为PerFine的训练免费框架，通过结合检索增强生成（RAG）技术和迭代式反馈优化，提升个性化文本生成的质量。PerFine利用检索到的用户档案生成初稿，并通过基于相同档案的批评模型提供结构化反馈，进而迭代优化生成结果，在多个数据集中显著提升了个性化指标。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.24427v1">SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models</a></td><td><details><summary>展开</summary>Evaluating the reasoning ability of language models (LMs) is complicated by
their extensive parametric world knowledge, where benchmark performance often
reflects factual recall rather than genuine reasoning. Existing datasets and
approaches (e.g., temporal filtering, paraphrasing, adversarial substitution)
cannot cleanly separate the two. We present SynthWorlds, a framework that
disentangles task reasoning complexity from factual knowledge. In SynthWorlds,
we construct parallel corpora representing two worlds with identical
interconnected structure: a real-mapped world, where models may exploit
parametric knowledge, and a synthetic-mapped world, where such knowledge is
meaningless. On top of these corpora, we design two mirrored tasks as case
studies: multi-hop question answering and page navigation, which maintain equal
reasoning difficulty across worlds. Experiments in parametric-only (e.g.,
closed-book QA) and knowledge-augmented (e.g., retrieval-augmented) LM settings
reveal a persistent knowledge advantage gap, defined as the performance boost
models gain from memorized parametric world knowledge. Knowledge acquisition
and integration mechanisms reduce but do not eliminate this gap, highlighting
opportunities for system improvements. Fully automatic and scalable,
SynthWorlds provides a controlled environment for evaluating LMs in ways that
were previously challenging, enabling precise and testable comparisons of
reasoning and memorization.</details></td><td><details><summary>展开</summary>这篇论文提出SynthWorlds框架，通过构建结构相同但知识背景不同的平行语料库（真实世界映射与合成世界映射），分离语言模型的任务推理能力与事实记忆能力。虽然主要研究模型推理与记忆的区分，但明确指出实验涉及检索增强（RAG）等知识增强设置，并探讨知识整合机制对性能差距的影响，因此与RAG技术相关。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.24402v1">Metadata-Driven Retrieval-Augmented Generation for Financial Question Answering</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) struggles on long, structured financial
filings where relevant evidence is sparse and cross-referenced. This paper
presents a systematic investigation of advanced metadata-driven
Retrieval-Augmented Generation (RAG) techniques, proposing and evaluating a
novel, multi-stage RAG architecture that leverages LLM-generated metadata. We
introduce a sophisticated indexing pipeline to create contextually rich
document chunks and benchmark a spectrum of enhancements, including
pre-retrieval filtering, post-retrieval reranking, and enriched embeddings,
benchmarked on the FinanceBench dataset. Our results reveal that while a
powerful reranker is essential for precision, the most significant performance
gains come from embedding chunk metadata directly with text ("contextual
chunks"). Our proposed optimal architecture combines LLM-driven pre-retrieval
optimizations with these contextual embeddings to achieve superior performance.
Additionally, we present a custom metadata reranker that offers a compelling,
cost-effective alternative to commercial solutions, highlighting a practical
trade-off between peak performance and operational efficiency. This study
provides a blueprint for building robust, metadata-aware RAG systems for
financial document analysis.</details></td><td><details><summary>展开</summary>这篇论文针对金融长文档中信息稀疏和交叉引用的问题，提出了一种基于LLM生成元数据的多阶段RAG架构，通过改进索引流程、预检索过滤、后检索重排序和增强嵌入等方法提升性能，并在FinanceBench数据集上验证了上下文分块嵌入和定制元数据重排序器的有效性，为金融文档分析提供了高效解决方案。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.24390v1">Improving LLM Reasoning via Dependency-Aware Query Decomposition and Logic-Parallel Content Expansion</a></td><td><details><summary>展开</summary>The integration of Large Language Models (LLMs) into real-time Web
applications, such as AI-powered search and conversational agents, presents a
fundamental Web infrastructure challenge: reconciling the demand for
high-quality, complex reasoning with the stringent low-latency and
high-throughput requirements of interactive services. Current LLM reasoning,
hindered by computationally inefficient sequential generation and rigid
reasoning strategies, creates a critical bottleneck for the Web services.
Existing approaches typically optimize the LLM reasoning for either efficiency
or quality but struggle to achieve both, and thus fail to meet the dual
requirements of modern Web platforms. To overcome these limitations, we propose
Orion, a novel and efficient reasoning framework that enables dependency-aware
query decomposition and logic-parallel content expansion. Concretely, Orion
decomposes a single query reasoning process into two synergistic phases: (1)
\textit{key point generation}, which distills logically structured key points
through retrieval-augmented few-shot prompting, and (2) \textit{content
parallel expansion}, which concurrently elaborates on these points based on a
dependency graph to ensure logical consistency. Furthermore, Orion introduces a
pipeline scheduling mechanism that exploits the complementary computational
characteristics of the two phases (generation imposes pressure on GPU computing
and expansion stresses on GPU memory) across multiple queries, enabling
cross-query parallelism and dramatically improving reasoning performance (\ie,
efficiency and quality). Experiments on diverse benchmarks show that Orion not
only delivers up to 4.33x higher token generation speed and 3.42x lower answer
latency over the baselines but also improves reasoning quality by up to 18.75%
through explicitly modeling inter-point dependencies.</details></td><td><details><summary>展开</summary>本文提出Orion框架，通过检索增强的少量示例提示（retrieval-augmented few-shot prompting）分解查询为逻辑关键点生成和并行内容扩展两阶段，结合依赖图实现高效推理，显著提升大语言模型在实时Web应用中的性能（速度、延迟和回答质量）。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.24303v1">Retrieval and Argumentation Enhanced Multi-Agent LLMs for Judgmental Forecasting</a></td><td><details><summary>展开</summary>Judgmental forecasting is the task of making predictions about future events
based on human judgment. This task can be seen as a form of claim verification,
where the claim corresponds to a future event and the task is to assess the
plausibility of that event. In this paper, we propose a novel multi-agent
framework for claim verification, whereby different agents may disagree on
claim veracity and bring specific evidence for and against the claims,
represented as quantitative bipolar argumentation frameworks (QBAFs). We then
instantiate the framework for supporting claim verification, with a variety of
agents realised with Large Language Models (LLMs): (1) ArgLLM agents, an
existing approach for claim verification that generates and evaluates QBAFs;
(2) RbAM agents, whereby LLM-empowered Relation-based Argument Mining (RbAM)
from external sources is used to generate QBAFs; (3) RAG-ArgLLM agents,
extending ArgLLM agents with a form of Retrieval-Augmented Generation (RAG) of
arguments from external sources. Finally, we conduct experiments with two
standard judgmental forecasting datasets, with instances of our framework with
two or three agents, empowered by six different base LLMs. We observe that
combining evidence from agents can improve forecasting accuracy, especially in
the case of three agents, while providing an explainable combination of
evidence for claim verification.</details></td><td><details><summary>展开</summary>这篇论文提出了一种基于多智能体框架的声明验证方法，其中部分智能体（如RAG-ArgLLM）通过检索增强生成（RAG）技术从外部来源获取论据，并结合定量双极论证框架（QBAFs）进行验证。实验表明，多智能体协同能提升预测准确性并提供可解释的证据组合。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.24242v1">Enabling Near-realtime Remote Sensing via Satellite-Ground Collaboration of Large Vision-Language Models</a></td><td><details><summary>展开</summary>Large vision-language models (LVLMs) have recently demonstrated great
potential in remote sensing (RS) tasks (e.g., disaster monitoring) conducted by
low Earth orbit (LEO) satellites. However, their deployment in real-world LEO
satellite systems remains largely unexplored, hindered by limited onboard
computing resources and brief satellite-ground contacts. We propose Grace, a
satellite-ground collaborative system designed for near-realtime LVLM inference
in RS tasks. Accordingly, we deploy compact LVLM on satellites for realtime
inference, but larger ones on ground stations (GSs) to guarantee end-to-end
performance. Grace is comprised of two main phases that are asynchronous
satellite-GS Retrieval-Augmented Generation (RAG), and a task dispatch
algorithm. Firstly, we still the knowledge archive of GS RAG to satellite
archive with tailored adaptive update algorithm during limited satellite-ground
data exchange period. Secondly, propose a confidence-based test algorithm that
either processes the task onboard the satellite or offloads it to the GS.
Extensive experiments based on real-world satellite orbital data show that
Grace reduces the average latency by 76-95% compared to state-of-the-art
methods, without compromising inference accuracy.</details></td><td><details><summary>展开</summary>这篇论文提出了一个名为Grace的卫星-地面协作系统，用于在遥感任务中实现近实时的视觉语言模型推理。系统通过异步的卫星-地面检索增强生成（RAG）和任务调度算法，结合卫星上的紧凑模型与地面站的大型模型，显著降低了延迟（76-95%）同时保持推理精度。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.24120v1">Graph-Guided Concept Selection for Efficient Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Graph-based RAG constructs a knowledge graph (KG) from text chunks to enhance
retrieval in Large Language Model (LLM)-based question answering. It is
especially beneficial in domains such as biomedicine, law, and political
science, where effective retrieval often involves multi-hop reasoning over
proprietary documents. However, these methods demand numerous LLM calls to
extract entities and relations from text chunks, incurring prohibitive costs at
scale. Through a carefully designed ablation study, we observe that certain
words (termed concepts) and their associated documents are more important.
Based on this insight, we propose Graph-Guided Concept Selection (G2ConS). Its
core comprises a chunk selection method and an LLM-independent concept graph.
The former selects salient document chunks to reduce KG construction costs; the
latter closes knowledge gaps introduced by chunk selection at zero cost.
Evaluations on multiple real-world datasets show that G2ConS outperforms all
baselines in construction cost, retrieval effectiveness, and answering quality.</details></td><td><details><summary>展开</summary>该论文提出了一种基于知识图谱的RAG方法（Graph-based RAG），通过从文本块构建知识图谱来增强基于大语言模型（LLM）的问答检索效果，特别适用于需要多跳推理的专业领域（如生物医学、法律等）。针对传统方法因频繁调用LLM提取实体和关系导致的高成本问题，作者提出了Graph-Guided Concept Selection（G2ConS），包含一个文档块选择方法和一个不依赖LLM的概念图谱，显著降低了知识图谱构建成本并填补了知识空白。实验表明，G2ConS在构建成本、检索效果和回答质量上均优于基线方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.24049v1">Learning from History: A Retrieval-Augmented Framework for Spatiotemporal Prediction</a></td><td><details><summary>展开</summary>Accurate and long-term spatiotemporal prediction for complex physical systems
remains a fundamental challenge in scientific computing. While deep learning
models, as powerful parametric approximators, have shown remarkable success,
they suffer from a critical limitation: the accumulation of errors during
long-term autoregressive rollouts often leads to physically implausible
artifacts. This deficiency arises from their purely parametric nature, which
struggles to capture the full constraints of a system's intrinsic dynamics. To
address this, we introduce a novel \textbf{Retrieval-Augmented Prediction
(RAP)} framework, a hybrid paradigm that synergizes the predictive power of
deep networks with the grounded truth of historical data. The core philosophy
of RAP is to leverage historical evolutionary exemplars as a non-parametric
estimate of the system's local dynamics. For any given state, RAP efficiently
retrieves the most similar historical analog from a large-scale database. The
true future evolution of this analog then serves as a \textbf{reference
target}. Critically, this target is not a hard constraint in the loss function
but rather a powerful conditional input to a specialized dual-stream
architecture. It provides strong \textbf{dynamic guidance}, steering the
model's predictions towards physically viable trajectories. In extensive
benchmarks across meteorology, turbulence, and fire simulation, RAP not only
surpasses state-of-the-art methods but also significantly outperforms a strong
\textbf{analog-only forecasting baseline}. More importantly, RAP generates
predictions that are more physically realistic by effectively suppressing error
divergence in long-term rollouts.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为“检索增强预测（RAP）”的混合框架，通过结合深度学习模型的预测能力和历史数据的真实动态，利用检索到的相似历史演化示例作为非参数估计，引导模型生成更符合物理规律的长期时空预测，显著提升了气象、湍流和火灾模拟等领域的预测准确性和物理合理性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.24003v1">META-RAG: Meta-Analysis-Inspired Evidence-Re-Ranking Method for Retrieval-Augmented Generation in Evidence-Based Medicine</a></td><td><details><summary>展开</summary>Evidence-based medicine (EBM) holds a crucial role in clinical application.
Given suitable medical articles, doctors effectively reduce the incidence of
misdiagnoses. Researchers find it efficient to use large language models (LLMs)
techniques like RAG for EBM tasks. However, the EBM maintains stringent
requirements for evidence, and RAG applications in EBM struggle to efficiently
distinguish high-quality evidence. Therefore, inspired by the meta-analysis
used in EBM, we provide a new method to re-rank and filter the medical
evidence. This method presents multiple principles to filter the best evidence
for LLMs to diagnose. We employ a combination of several EBM methods to emulate
the meta-analysis, which includes reliability analysis, heterogeneity analysis,
and extrapolation analysis. These processes allow the users to retrieve the
best medical evidence for the LLMs. Ultimately, we evaluate these high-quality
articles and show an accuracy improvement of up to 11.4% in our experiments and
results. Our method successfully enables RAG to extract higher-quality and more
reliable evidence from the PubMed dataset. This work can reduce the infusion of
incorrect knowledge into responses and help users receive more effective
replies.</details></td><td><details><summary>展开</summary>这篇论文提出了一种基于RAG技术的改进方法，通过结合元分析中的可靠性分析、异质性分析和外推分析，对医学证据进行重新排序和筛选，以提高LLMs在循证医学任务中的诊断准确性，实验结果显示准确性提升了11.4%。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.23998v1">PICOs-RAG: PICO-supported Query Rewriting for Retrieval-Augmented Generation in Evidence-Based Medicine</a></td><td><details><summary>展开</summary>Evidence-based medicine (EBM) research has always been of paramount
importance. It is important to find appropriate medical theoretical support for
the needs from physicians or patients to reduce the occurrence of medical
accidents. This process is often carried out by human querying relevant
literature databases, which lacks objectivity and efficiency. Therefore,
researchers utilize retrieval-augmented generation (RAG) to search for evidence
and generate responses automatically. However, current RAG methods struggle to
handle complex queries in real-world clinical scenarios. For example, when
queries lack certain information or use imprecise language, the model may
retrieve irrelevant evidence and generate unhelpful answers. To address this
issue, we present the PICOs-RAG to expand the user queries into a better
format. Our method can expand and normalize the queries into professional ones
and use the PICO format, a search strategy tool present in EBM, to extract the
most important information used for retrieval. This approach significantly
enhances retrieval efficiency and relevance, resulting in up to an 8.8\%
improvement compared to the baseline evaluated by our method. Thereby the
PICOs-RAG improves the performance of the large language models into a helpful
and reliable medical assistant in EBM.</details></td><td><details><summary>展开</summary>该论文提出了一种名为PICOs-RAG的方法，通过扩展和规范化用户查询（采用EBM中的PICO格式）来优化检索增强生成（RAG）在循证医学中的应用，解决了复杂临床查询下检索不精准的问题，实验显示其检索效率和相关性较基线提升8.8%，使大语言模型成为更可靠的医学助手。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.23995v1">M-Eval: A Heterogeneity-Based Framework for Multi-evidence Validation in Medical RAG Systems</a></td><td><details><summary>展开</summary>Retrieval-augmented Generation (RAG) has demonstrated potential in enhancing
medical question-answering systems through the integration of large language
models (LLMs) with external medical literature. LLMs can retrieve relevant
medical articles to generate more professional responses efficiently. However,
current RAG applications still face problems. They generate incorrect
information, such as hallucinations, and they fail to use external knowledge
correctly. To solve these issues, we propose a new method named M-Eval. This
method is inspired by the heterogeneity analysis approach used in
Evidence-Based Medicine (EBM). Our approach can check for factual errors in RAG
responses using evidence from multiple sources. First, we extract additional
medical literature from external knowledge bases. Then, we retrieve the
evidence documents generated by the RAG system. We use heterogeneity analysis
to check whether the evidence supports different viewpoints in the response. In
addition to verifying the accuracy of the response, we also assess the
reliability of the evidence provided by the RAG system. Our method shows an
improvement of up to 23.31% accuracy across various LLMs. This work can help
detect errors in current RAG-based medical systems. It also makes the
applications of LLMs more reliable and reduces diagnostic errors.</details></td><td><details><summary>展开</summary>该论文提出了一种名为M-Eval的新方法，旨在解决RAG在医疗问答系统中存在的生成错误信息（如幻觉）和未能正确使用外部知识的问题。该方法基于循证医学的异质性分析，通过从外部知识库检索额外医学文献并与RAG生成的证据文档对比，验证回答的准确性和证据的可靠性，实验显示其可将不同大语言模型的准确率提升高达23.31%。</details></td></tr></tbody></table>

### 📅 2025-10-27
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.23601v1">Alita-G: Self-Evolving Generative Agent for Agent Generation</a></td><td><details><summary>展开</summary>Large language models (LLMs) have been shown to perform better when
scaffolded into agents with memory, tools, and feedback. Beyond this,
self-evolving agents have emerged, but current work largely limits adaptation
to prompt rewriting or failure retries. Therefore, we present ALITA-G, a
self-evolution framework that transforms a general-purpose agent into a domain
expert by systematically generating, abstracting, and curating Model Context
Protocol (MCP) tools. In this framework, a generalist agent executes a curated
suite of target-domain tasks and synthesizes candidate MCPs from successful
trajectories. These are then abstracted to parameterized primitives and
consolidated into an MCP Box. At inference time, ALITA-G performs
retrieval-augmented MCP selection with the help of each tool's descriptions and
use cases, before executing an agent equipped with the MCP Executor. Across
several benchmarks GAIA, PathVQA, and Humanity's Last Exam, ALITA-G attains
strong gains while reducing computation costs. On GAIA validation, it achieves
83.03% pass@1 and 89.09% pass@3, establishing a new state-of-the-art result
while reducing mean tokens per example by approximately 15% relative to a
strong baseline agent. ALITA-G thus provides a principled pathway from
generalist capability to reusable, domain-specific competence, improving both
accuracy and efficiency on complex reasoning tasks.</details></td><td><details><summary>展开</summary>这篇文章介绍了ALITA-G框架，该框架通过生成、抽象和整理模型上下文协议（MCP）工具，将通用智能代理转化为领域专家。ALITA-G在推理时采用检索增强的MCP选择方法，结合工具描述和使用案例进行检索，从而提升任务执行的准确性和效率，并在多个基准测试中取得了显著的性能提升和计算成本降低。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.23544v1">LimRank: Less is More for Reasoning-Intensive Information Reranking</a></td><td><details><summary>展开</summary>Existing approaches typically rely on large-scale fine-tuning to adapt LLMs
for information reranking tasks, which is computationally expensive. In this
work, we demonstrate that modern LLMs can be effectively adapted using only
minimal, high-quality supervision. To enable this, we design
LIMRANK-SYNTHESIZER, a reusable and open-source pipeline for generating
diverse, challenging, and realistic reranking examples. Using this synthetic
data, we fine-tune our reranker model, LIMRANK. We evaluate LIMRANK on two
challenging benchmarks, i.e., BRIGHT for reasoning-intensive retrieval and
FollowIR for instruction-following retrieval. Our experiments demonstrate that
LIMRANK achieves competitive performance, while being trained on less than 5%
of the data typically used in prior work. Further ablation studies demonstrate
the effectiveness of LIMRANK-SYNTHESIZER and the strong generalization
capabilities of LIMRANK across downstream tasks, including scientific
literature search and retrieval-augmented generation for knowledge-intensive
problem solving.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为LIMRANK的高效信息重排模型，通过合成数据（LIMRANK-SYNTHESIZER生成）进行小规模微调，显著减少训练数据需求。研究验证了其在推理密集型检索（BRIGHT）和指令跟随检索（FollowIR）中的竞争力，并特别提到该模型在知识密集型问题解决（如科学文献搜索和检索增强生成RAG）中的下游任务泛化能力。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.23271v1">Mubeen AI: A Specialized Arabic Language Model for Heritage Preservation and User Intent Understanding</a></td><td><details><summary>展开</summary>Mubeen is a proprietary Arabic language model developed by MASARAT SA,
optimized for deep understanding of Arabic linguistics, Islamic studies, and
cultural heritage. Trained on an extensive collection of authentic Arabic
sources significantly expanded by digitizing historical manuscripts via a
proprietary Arabic OCR engine, the model incorporates seminal scholarly works
in linguistics, jurisprudence, hadith, and Quranic exegesis, alongside
thousands of academic theses and peer-reviewed research papers. Conditioned
through a deep linguistic engineering framework, Mubeen masters not just the
meaning but the eloquence of Arabic, enabling precise understanding across
classical texts, contemporary writing, and regional dialects with focus on
comprehending user intent and delivering accurate, contextually relevant
responses. Unlike other Arabic models relying on translated English data that
often fail in intent detection or retrieval-augmented generation (RAG), Mubeen
uses native Arabic sources to ensure cultural authenticity and accuracy. Its
core innovation is the Practical Closure Architecture, designed to solve the
"Utility Gap Crisis" where factually correct answers fail to resolve users'
core needs, forcing them into frustrating cycles of re-prompting. By
prioritizing clarity and decisive guidance, Mubeen transforms from an
information repository into a decisive guide, aligning with Saudi Vision 2030.
The model's architecture combines deep heritage specialization with
multi-disciplinary expert modules, enabling robust performance across both
cultural preservation and general knowledge domains.</details></td><td><details><summary>展开</summary>这篇论文介绍了Mubeen，一个专有的阿拉伯语语言模型，通过结合原生阿拉伯语资料和深度语言工程框架，解决了传统阿拉伯语模型在意图检测和检索增强生成（RAG）上的不足。其核心创新“实用闭合架构”旨在解决“效用差距危机”，确保回答不仅准确且直接满足用户需求，同时强调了文化真实性和多学科专家模块的整合。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.23070v1">Quality-Aware Translation Tagging in Multilingual RAG system</a></td><td><details><summary>展开</summary>Multilingual Retrieval-Augmented Generation (mRAG) often retrieves English
documents and translates them into the query language for low-resource
settings. However, poor translation quality degrades response generation
performance. Existing approaches either assume sufficient translation quality
or utilize the rewriting method, which introduces factual distortion and
hallucinations. To mitigate these problems, we propose Quality-Aware
Translation Tagging in mRAG (QTT-RAG), which explicitly evaluates translation
quality along three dimensions-semantic equivalence, grammatical accuracy, and
naturalness&fluency-and attach these scores as metadata without altering the
original content. We evaluate QTT-RAG against CrossRAG and DKM-RAG as baselines
in two open-domain QA benchmarks (XORQA, MKQA) using six instruction-tuned LLMs
ranging from 2.4B to 14B parameters, covering two low-resource languages
(Korean and Finnish) and one high-resource language (Chinese). QTT-RAG
outperforms the baselines by preserving factual integrity while enabling
generator models to make informed decisions based on translation reliability.
This approach allows for effective usage of cross-lingual documents in
low-resource settings with limited native language documents, offering a
practical and robust solution across multilingual domains.</details></td><td><details><summary>展开</summary>该论文提出了一种名为QTT-RAG的质量感知翻译标记方法，用于改进多语言检索增强生成（mRAG）中的翻译质量问题。通过评估翻译的语义等价性、语法准确性和流畅性，并将评分作为元数据附加到原文，该方法在低资源语言（如韩语、芬兰语）和高资源语言（如中文）的开放域问答任务中优于现有基线模型，同时保持了事实完整性并减少了幻觉问题。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.22956v1">Tagging-Augmented Generation: Assisting Language Models in Finding Intricate Knowledge In Long Contexts</a></td><td><details><summary>展开</summary>Recent investigations into effective context lengths of modern flagship large
language models (LLMs) have revealed major limitations in effective question
answering (QA) and reasoning over long and complex contexts for even the
largest and most impressive cadre of models. While approaches like
retrieval-augmented generation (RAG) and chunk-based re-ranking attempt to
mitigate this issue, they are sensitive to chunking, embedding and retrieval
strategies and models, and furthermore, rely on extensive pre-processing,
knowledge acquisition and indexing steps. In this paper, we propose
Tagging-Augmented Generation (TAG), a lightweight data augmentation strategy
that boosts LLM performance in long-context scenarios, without degrading and
altering the integrity and composition of retrieved documents. We validate our
hypothesis by augmenting two challenging and directly relevant
question-answering benchmarks -- NoLima and NovelQA -- and show that tagging
the context or even just adding tag definitions into QA prompts leads to
consistent performance gains over the baseline -- up to 17% for 32K token
contexts, and 2.9% in complex reasoning question-answering for multi-hop
queries requiring knowledge across a wide span of text. Additional details are
available at https://sites.google.com/view/tag-emnlp.</details></td><td><details><summary>展开</summary>这篇论文探讨了现代大语言模型（LLMs）在处理长且复杂上下文时的局限性，并提出了一种名为“标记增强生成（TAG）”的轻量级数据增强策略。TAG通过在上下文或提示中添加标记或标记定义，显著提升了LLMs在长上下文场景下的性能（如问答和复杂推理任务），而无需依赖RAG等传统方法所需的预处理和索引步骤。实验表明，TAG在32K标记的上下文和多跳查询任务中分别实现了高达17%和2.9%的性能提升。</details></td></tr></tbody></table>

### 📅 2025-10-26
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.22728v1">S-Chain: Structured Visual Chain-of-Thought For Medicine</a></td><td><details><summary>展开</summary>Faithful reasoning in medical vision-language models (VLMs) requires not only
accurate predictions but also transparent alignment between textual rationales
and visual evidence. While Chain-of-Thought (CoT) prompting has shown promise
in medical visual question answering (VQA), no large-scale expert-level dataset
has captured stepwise reasoning with precise visual grounding. We introduce
S-Chain, the first large-scale dataset of 12,000 expert-annotated medical
images with bounding boxes and structured visual CoT (SV-CoT), explicitly
linking visual regions to reasoning steps. The dataset further supports 16
languages, totaling over 700k VQA pairs for broad multilingual applicability.
Using S-Chain, we benchmark state-of-the-art medical VLMs (ExGra-Med,
LLaVA-Med) and general-purpose VLMs (Qwen2.5-VL, InternVL2.5), showing that
SV-CoT supervision significantly improves interpretability, grounding fidelity,
and robustness. Beyond benchmarking, we study its synergy with
retrieval-augmented generation, revealing how domain knowledge and visual
grounding interact during autoregressive reasoning. Finally, we propose a new
mechanism that strengthens the alignment between visual evidence and reasoning,
improving both reliability and efficiency. S-Chain establishes a new benchmark
for grounded medical reasoning and paves the way toward more trustworthy and
explainable medical VLMs.</details></td><td><details><summary>展开</summary>这篇论文提出了S-Chain数据集，通过结构化视觉链式推理（SV-CoT）和跨语言VQA对增强医学视觉语言模型（VLMs）的可解释性与视觉证据对齐，并探讨了其与检索增强生成（RAG）的协同作用，揭示了领域知识与视觉基础在推理中的交互机制，最终提出了一种提升视觉证据与推理对齐的新方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.22710v1">RaCoT: Plug-and-Play Contrastive Example Generation Mechanism for Enhanced LLM Reasoning Reliability</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) faces a core bottleneck with
knowledge-sparse and semantically ambiguous long-tail queries, where retrieval
noise distorts reasoning and necessitates costly post-processing. To tackle
this, we propose RaCoT (Retrieval-aware Contrastive-of-Thought), a novel
framework that shifts contrastive thinking to the pre-retrieval stage. By
automatically generating a semantically adjacent yet differently answered
contrastive question and extracting a $\Delta$-Prompt to capture their key
differences, RaCoT guides the model to proactively focus on the ``critical
details that determine answer divergence." This approach allows it to suppress
semantic interference within a single retrieval pass, overcoming the
theoretical bottleneck of single-vector queries that struggle to simultaneously
encode signals for what to attend to and what to ignore. On six authoritative
benchmarks, including PopQA and TriviaQA-unfiltered, RaCoT outperforms strong
baselines like RankRAG and Self-RAG by 0.9-2.4 percentage points. It exhibits
superior robustness, with a performance drop of only 8.6\% in adversarial
tests, far surpassing the over 15\% degradation in other methods. Furthermore,
its low latency (3.12s) and token overhead (11.54) place it on the
accuracy-efficiency Pareto frontier, while ablation studies validate the
necessity of each component. Ultimately, RaCoT reframes the RAG paradigm from
``post-hoc context cleaning" to ``a priori shaping of discriminative
reasoning", offering an efficient and robust path toward reliable AI systems
for real-time, resource-constrained deployments.</details></td><td><details><summary>展开</summary>本文提出RaCoT框架，通过预检索阶段生成对比性问题并提取关键差异提示（Δ-Prompt），解决RAG中长尾查询的语义模糊和检索噪声问题，显著提升模型性能与鲁棒性，并在效率与准确性上达到帕累托前沿。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.22694v1">Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Multimodal Retrieval-Augmented Generation (MRAG) has emerged as a promising
method to generate factual and up-to-date responses of Multimodal Large
Language Models (MLLMs) by incorporating non-parametric knowledge from external
knowledge bases. However, existing MRAG approaches suffer from static retrieval
strategies, inflexible modality selection, and suboptimal utilization of
retrieved information, leading to three critical challenges: determining when
to retrieve, what modality to incorporate, and how to utilize retrieved
information effectively. To address these challenges, we introduce Windsock, a
query-dependent module making decisions on retrieval necessity and modality
selection, effectively reducing computational overhead and improving response
quality. Additionally, we propose Dynamic Noise-Resistance (DANCE) Instruction
Tuning, an adaptive training strategy that enhances MLLMs' ability to utilize
retrieved information while maintaining robustness against noise. Moreover, we
adopt a self-assessment approach leveraging knowledge within MLLMs to convert
question-answering datasets to MRAG training datasets. Extensive experiments
demonstrate that our proposed method significantly improves the generation
quality by 17.07% while reducing 8.95% retrieval times.</details></td><td><details><summary>展开</summary>这篇论文提出了一种多模态检索增强生成（MRAG）方法Windsock，通过动态决策检索必要性和模态选择来优化生成质量，并结合自适应训练策略DANCE提升模型对检索信息的利用能力，实验证明其显著提升了生成质量并减少了检索次数。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.22689v1">Rule-Based Explanations for Retrieval-Augmented LLM Systems</a></td><td><details><summary>展开</summary>If-then rules are widely used to explain machine learning models; e.g., "if
employed = no, then loan application = rejected." We present the first proposal
to apply rules to explain the emerging class of large language models (LLMs)
with retrieval-augmented generation (RAG). Since RAG enables LLM systems to
incorporate retrieved information sources at inference time, rules linking the
presence or absence of sources can explain output provenance; e.g., "if a Times
Higher Education ranking article is retrieved, then the LLM ranks Oxford
first." To generate such rules, a brute force approach would probe the LLM with
all source combinations and check if the presence or absence of any sources
leads to the same output. We propose optimizations to speed up rule generation,
inspired by Apriori-like pruning from frequent itemset mining but redefined
within the scope of our novel problem. We conclude with qualitative and
quantitative experiments demonstrating our solutions' value and efficiency.</details></td><td><details><summary>展开</summary>该论文提出了一种利用if-then规则解释检索增强生成（RAG）模型决策的方法，通过分析检索到的信息源与输出间的因果关系生成规则（如“若检索到某排名文章，则模型输出特定结果”），并设计优化算法加速规则挖掘，最终通过实验验证了方案的有效性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.22609v1">CLIN-LLM: A Safety-Constrained Hybrid Framework for Clinical Diagnosis and Treatment Generation</a></td><td><details><summary>展开</summary>Accurate symptom-to-disease classification and clinically grounded treatment
recommendations remain challenging, particularly in heterogeneous patient
settings with high diagnostic risk. Existing large language model (LLM)-based
systems often lack medical grounding and fail to quantify uncertainty,
resulting in unsafe outputs. We propose CLIN-LLM, a safety-constrained hybrid
pipeline that integrates multimodal patient encoding, uncertainty-calibrated
disease classification, and retrieval-augmented treatment generation. The
framework fine-tunes BioBERT on 1,200 clinical cases from the Symptom2Disease
dataset and incorporates Focal Loss with Monte Carlo Dropout to enable
confidence-aware predictions from free-text symptoms and structured vitals.
Low-certainty cases (18%) are automatically flagged for expert review, ensuring
human oversight. For treatment generation, CLIN-LLM employs Biomedical
Sentence-BERT to retrieve top-k relevant dialogues from the 260,000-sample
MedDialog corpus. The retrieved evidence and patient context are fed into a
fine-tuned FLAN-T5 model for personalized treatment generation, followed by
post-processing with RxNorm for antibiotic stewardship and drug-drug
interaction (DDI) screening. CLIN-LLM achieves 98% accuracy and F1 score,
outperforming ClinicalBERT by 7.1% (p < 0.001), with 78% top-5 retrieval
precision and a clinician-rated validity of 4.2 out of 5. Unsafe antibiotic
suggestions are reduced by 67% compared to GPT-5. These results demonstrate
CLIN-LLM's robustness, interpretability, and clinical safety alignment. The
proposed system provides a deployable, human-in-the-loop decision support
framework for resource-limited healthcare environments. Future work includes
integrating imaging and lab data, multilingual extensions, and clinical trial
validation.</details></td><td><details><summary>展开</summary>这篇文章提出了CLIN-LLM，一个结合多模态患者编码、不确定性校准疾病分类和检索增强治疗生成的安全约束混合框架。其中，治疗生成部分通过Biomedical Sentence-BERT从MedDialog语料库中检索相关对话，并将检索结果与患者上下文输入FLAN-T5模型生成个性化治疗方案，体现了RAG技术的应用。该系统在准确性、临床安全性和检索性能方面表现优异，显著减少了不安全建议。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.22521v1">Open Multimodal Retrieval-Augmented Factual Image Generation</a></td><td><details><summary>展开</summary>Large Multimodal Models (LMMs) have achieved remarkable progress in
generating photorealistic and prompt-aligned images, but they often produce
outputs that contradict verifiable knowledge, especially when prompts involve
fine-grained attributes or time-sensitive events. Conventional
retrieval-augmented approaches attempt to address this issue by introducing
external information, yet they are fundamentally incapable of grounding
generation in accurate and evolving knowledge due to their reliance on static
sources and shallow evidence integration. To bridge this gap, we introduce
ORIG, an agentic open multimodal retrieval-augmented framework for Factual
Image Generation (FIG), a new task that requires both visual realism and
factual grounding. ORIG iteratively retrieves and filters multimodal evidence
from the web and incrementally integrates the refined knowledge into enriched
prompts to guide generation. To support systematic evaluation, we build
FIG-Eval, a benchmark spanning ten categories across perceptual, compositional,
and temporal dimensions. Experiments demonstrate that ORIG substantially
improves factual consistency and overall image quality over strong baselines,
highlighting the potential of open multimodal retrieval for factual image
generation.</details></td><td><details><summary>展开</summary>该论文提出了一种名为ORIG的开放式多模态检索增强框架，用于解决大型多模态模型（LMMs）生成图像时与可验证知识矛盾的问题。ORIG通过迭代检索和过滤网络中的多模态证据，并将精炼知识逐步整合到提示中以指导生成，显著提升了生成图像的事实一致性和质量。同时，作者构建了FIG-Eval基准进行系统评估，验证了该框架的有效性。</details></td></tr></tbody></table>

### 📅 2025-10-25
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.22344v1">FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>While Retrieval-Augmented Generation (RAG) mitigates hallucination and
knowledge staleness in Large Language Models (LLMs), existing frameworks often
falter on complex, multi-hop queries that require synthesizing information from
disparate sources. Current advanced RAG methods, employing iterative or
adaptive strategies, lack a robust mechanism to systematically identify and
fill evidence gaps, often propagating noise or failing to gather a
comprehensive context. We introduce FAIR-RAG, a novel agentic framework that
transforms the standard RAG pipeline into a dynamic, evidence-driven reasoning
process. At its core is an Iterative Refinement Cycle governed by a module we
term Structured Evidence Assessment (SEA). The SEA acts as an analytical gating
mechanism: it deconstructs the initial query into a checklist of required
findings and audits the aggregated evidence to identify confirmed facts and,
critically, explicit informational gaps. These gaps provide a precise signal to
an Adaptive Query Refinement agent, which generates new, targeted sub-queries
to retrieve missing information. This cycle repeats until the evidence is
verified as sufficient, ensuring a comprehensive context for a final, strictly
faithful generation. We conducted experiments on challenging multi-hop QA
benchmarks, including HotpotQA, 2WikiMultiHopQA, and MusiQue. In a unified
experimental setup, FAIR-RAG significantly outperforms strong baselines. On
HotpotQA, it achieves an F1-score of 0.453 -- an absolute improvement of 8.3
points over the strongest iterative baseline -- establishing a new
state-of-the-art for this class of methods on these benchmarks. Our work
demonstrates that a structured, evidence-driven refinement process with
explicit gap analysis is crucial for unlocking reliable and accurate reasoning
in advanced RAG systems for complex, knowledge-intensive tasks.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为FAIR-RAG的新型代理框架，通过引入结构化证据评估（SEA）和自适应查询细化机制，改进了现有RAG系统在处理复杂多跳查询时的不足。FAIR-RAG通过迭代优化证据收集过程，显著提升了在多个基准测试上的性能，特别是在HotpotQA上实现了8.3分的绝对提升。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.22272v1">From Slides to Chatbots: Enhancing Large Language Models with University Course Materials</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) have advanced rapidly in recent years. One
application of LLMs is to support student learning in educational settings.
However, prior work has shown that LLMs still struggle to answer questions
accurately within university-level computer science courses. In this work, we
investigate how incorporating university course materials can enhance LLM
performance in this setting. A key challenge lies in leveraging diverse course
materials such as lecture slides and transcripts, which differ substantially
from typical textual corpora: slides also contain visual elements like images
and formulas, while transcripts contain spoken, less structured language. We
compare two strategies, Retrieval-Augmented Generation (RAG) and Continual
Pre-Training (CPT), to extend LLMs with course-specific knowledge. For lecture
slides, we further explore a multi-modal RAG approach, where we present the
retrieved content to the generator in image form. Our experiments reveal that,
given the relatively small size of university course materials, RAG is more
effective and efficient than CPT. Moreover, incorporating slides as images in
the multi-modal setting significantly improves performance over text-only
retrieval. These findings highlight practical strategies for developing AI
assistants that better support learning and teaching, and we hope they inspire
similar efforts in other educational contexts.</details></td><td><details><summary>展开</summary>这篇论文探讨了如何通过整合大学课程材料（如讲义幻灯片和转录文本）来提升大语言模型（LLMs）在计算机科学教育中的问答性能，重点比较了检索增强生成（RAG）和持续预训练（CPT）两种策略，并提出多模态RAG方法（以图像形式处理幻灯片内容），实验表明RAG在小型课程数据集上更高效且多模态检索显著优于纯文本检索。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.22210v1">LSPRAG: LSP-Guided RAG for Language-Agnostic Real-Time Unit Test Generation</a></td><td><details><summary>展开</summary>Automated unit test generation is essential for robust software development,
yet existing approaches struggle to generalize across multiple programming
languages and operate within real-time development. While Large Language Models
(LLMs) offer a promising solution, their ability to generate high coverage test
code depends on prompting a concise context of the focal method. Current
solutions, such as Retrieval-Augmented Generation, either rely on imprecise
similarity-based searches or demand the creation of costly, language-specific
static analysis pipelines. To address this gap, we present LSPRAG, a framework
for concise-context retrieval tailored for real-time, language-agnostic unit
test generation. LSPRAG leverages off-the-shelf Language Server Protocol (LSP)
back-ends to supply LLMs with precise symbol definitions and references in real
time. By reusing mature LSP servers, LSPRAG provides an LLM with language-aware
context retrieval, requiring minimal per-language engineering effort. We
evaluated LSPRAG on open-source projects spanning Java, Go, and Python.
Compared to the best performance of baselines, LSPRAG increased line coverage
by up to 174.55% for Golang, 213.31% for Java, and 31.57% for Python.</details></td><td><details><summary>展开</summary>这篇论文提出了LSPRAG框架，通过集成语言服务器协议（LSP）实现实时、语言无关的单元测试生成，利用精确的符号检索增强大语言模型的上下文输入，显著提升了多语言测试代码的覆盖率，解决了传统RAG方法在相似性搜索或静态分析上的局限性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.22143v1">OlaMind: Towards Human-Like and Hallucination-Safe Customer Service for Retrieval-Augmented Dialogue</a></td><td><details><summary>展开</summary>Intelligent customer service (ICS) systems via retrieval-augmented generation
(RAG) have been widely adopted in Web-based domains such as social platforms
and e-commerce, achieving remarkable improvements in automation and efficiency.
However, notable limitations still remain: these systems are prone to
hallucinations and often generate rigid, mechanical responses, which can
introduce business risks and undermine user experience, especially in Web-based
customer service interactions under the RAG scenarios. In this paper, we
introduce OlaMind, a human-like and hallucination-safe customer service
framework for retrieval-augmented dialogue. Specifically, it first leverages a
Learn-to-Think stage to learn the reasoning processes and response strategies
from human experts, and then employs a Learn-to-Respond stage to perform
cold-start supervised fine-tuning (SFT) combined with reinforcement learning
(RL) for basic-to-hard self-refinement. Our method significantly enhances
human-likeness and naturalness while effectively mitigating hallucinations and
critical business risks. We have conducted large-scale online A/B experiments
in an industry-level social customer service setting, and extensive
experimental results show that OlaMind achieves significant cumulative relative
improvements with intelligent resolution rates +28.92%/+18.42% and human
takeover rate -6.08%/-7.12% in community-support/livestream-interaction
scenarios, respectively, which highlights its consistent effectiveness across
diverse real-world applications. The code and data will be publicly available.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为OlaMind的智能客服框架，基于检索增强生成（RAG）技术，旨在解决现有RAG系统在客服对话中易产生幻觉和机械回复的问题。OlaMind通过"Learn-to-Think"阶段学习人类专家的推理过程与应答策略，再通过"Learn-to-Respond"阶段结合监督微调（SFT）和强化学习（RL）进行自我优化，显著提升了回答的自然度和安全性。工业级A/B测试表明，该框架在社区支持和直播互动场景中显著提高了智能解决率并降低了人工接管率。</details></td></tr></tbody></table>

### 📅 2025-10-24
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.21933v1">A Comparison of Conversational Models and Humans in Answering Technical Questions: the Firefox Case</a></td><td><details><summary>展开</summary>The use of Large Language Models (LLMs) to support tasks in software
development has steadily increased over recent years. From assisting developers
in coding activities to providing conversational agents that answer newcomers'
questions. In collaboration with the Mozilla Foundation, this study evaluates
the effectiveness of Retrieval-Augmented Generation (RAG) in assisting
developers within the Mozilla Firefox project. We conducted an empirical
analysis comparing responses from human developers, a standard GPT model, and a
GPT model enhanced with RAG, using real queries from Mozilla's developer chat
rooms. To ensure a rigorous evaluation, Mozilla experts assessed the responses
based on helpfulness, comprehensiveness, and conciseness. The results show that
RAG-assisted responses were more comprehensive than human developers (62.50% to
54.17%) and almost as helpful (75.00% to 79.17%), suggesting RAG's potential to
enhance developer assistance. However, the RAG responses were not as concise
and often verbose. The results show the potential to apply RAG-based tools to
Open Source Software (OSS) to minimize the load to core maintainers without
losing answer quality. Toning down retrieval mechanisms and making responses
even shorter in the future would enhance developer assistance in massive
projects like Mozilla Firefox.</details></td><td><details><summary>展开</summary>这篇论文评估了检索增强生成（RAG）在协助Mozilla Firefox开发者的效果，通过对比人类开发者、标准GPT模型和RAG增强GPT模型的回答质量（如帮助性、全面性和简洁性），发现RAG在综合性和帮助性上接近或优于人类，但存在冗长问题，未来可通过优化检索机制提升效率，减轻开源项目核心维护者负担。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.21656v1">CMOMgen: Complex Multi-Ontology Alignment via Pattern-Guided In-Context Learning</a></td><td><details><summary>展开</summary>Constructing comprehensive knowledge graphs requires the use of multiple
ontologies in order to fully contextualize data into a domain. Ontology
matching finds equivalences between concepts interconnecting ontologies and
creating a cohesive semantic layer. While the simple pairwise state of the art
is well established, simple equivalence mappings cannot provide full semantic
integration of related but disjoint ontologies. Complex multi-ontology matching
(CMOM) aligns one source entity to composite logical expressions of multiple
target entities, establishing more nuanced equivalences and provenance along
the ontological hierarchy.
  We present CMOMgen, the first end-to-end CMOM strategy that generates
complete and semantically sound mappings, without establishing any restrictions
on the number of target ontologies or entities. Retrieval-Augmented Generation
selects relevant classes to compose the mapping and filters matching reference
mappings to serve as examples, enhancing In-Context Learning. The strategy was
evaluated in three biomedical tasks with partial reference alignments. CMOMgen
outperforms baselines in class selection, demonstrating the impact of having a
dedicated strategy. Our strategy also achieves a minimum of 63% in F1-score,
outperforming all baselines and ablated versions in two out of three tasks and
placing second in the third. Furthermore, a manual evaluation of non-reference
mappings showed that 46% of the mappings achieve the maximum score, further
substantiating its ability to construct semantically sound mappings.</details></td><td><details><summary>展开</summary>该论文提出了一种名为CMOMgen的端到端复杂多本体匹配策略，利用检索增强生成（RAG）技术从多个目标本体中选择相关类并过滤参考映射作为示例，以提高上下文学习效果，从而生成完整且语义合理的映射。该方法在三个生物医学任务中表现优于基线模型，验证了其有效性和语义准确性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.21538v1">InterpDetect: Interpretable Signals for Detecting Hallucinations in Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) integrates external knowledge to
mitigate hallucinations, yet models often generate outputs inconsistent with
retrieved content. Accurate hallucination detection requires disentangling the
contributions of external context and parametric knowledge, which prior methods
typically conflate. We investigate the mechanisms underlying RAG hallucinations
and find they arise when later-layer FFN modules disproportionately inject
parametric knowledge into the residual stream. To address this, we explore a
mechanistic detection approach based on external context scores and parametric
knowledge scores. Using Qwen3-0.6b, we compute these scores across layers and
attention heads and train regression-based classifiers to predict
hallucinations. Our method is evaluated against state-of-the-art LLMs (GPT-5,
GPT-4.1) and detection baselines (RAGAS, TruLens, RefChecker). Furthermore,
classifiers trained on Qwen3-0.6b signals generalize to GPT-4.1-mini responses,
demonstrating the potential of proxy-model evaluation. Our results highlight
mechanistic signals as efficient, generalizable predictors for hallucination
detection in RAG systems.</details></td><td><details><summary>展开</summary>这篇论文研究了RAG系统中的幻觉生成机制，提出了一种基于外部语境评分和参数知识评分的机制检测方法，通过分析模型层间信号训练分类器来预测幻觉，并在多种模型和基线方法上验证了其有效性和泛化能力。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.21459v1">SBASH: a Framework for Designing and Evaluating RAG vs. Prompt-Tuned LLM Honeypots</a></td><td><details><summary>展开</summary>Honeypots are decoy systems used for gathering valuable threat intelligence
or diverting attackers away from production systems. Maximising attacker
engagement is essential to their utility. However research has highlighted that
context-awareness, such as the ability to respond to new attack types, systems
and attacker agents, is necessary to increase engagement. Large Language Models
(LLMs) have been shown as one approach to increase context awareness but suffer
from several challenges including accuracy and timeliness of response time,
high operational costs and data-protection issues due to cloud deployment. We
propose the System-Based Attention Shell Honeypot (SBASH) framework which
manages data-protection issues through the use of lightweight local LLMs. We
investigate the use of Retrieval Augmented Generation (RAG) supported LLMs and
non-RAG LLMs for Linux shell commands and evaluate them using several different
metrics such as response time differences, realism from human testers, and
similarity to a real system calculated with Levenshtein distance, SBert, and
BertScore. We show that RAG improves accuracy for untuned models while models
that have been tuned via a system prompt that tells the LLM to respond like a
Linux system achieve without RAG a similar accuracy as untuned with RAG, while
having a slightly lower latency.</details></td><td><details><summary>展开</summary>这篇论文提出了一个名为SBASH的框架，通过使用轻量级本地大语言模型（LLMs）来解决数据保护问题，并研究了基于检索增强生成（RAG）和非RAG的LLMs在Linux shell命令中的应用效果，评估了响应时间、真实性和系统相似性等指标，发现RAG能提高未调优模型的准确性，而通过系统提示调优的模型即使不使用RAG也能达到类似效果但延迟略低。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.21440v1">Redefining Retrieval Evaluation in the Era of LLMs</a></td><td><details><summary>展开</summary>Traditional Information Retrieval (IR) metrics, such as nDCG, MAP, and MRR,
assume that human users sequentially examine documents with diminishing
attention to lower ranks. This assumption breaks down in Retrieval Augmented
Generation (RAG) systems, where search results are consumed by Large Language
Models (LLMs), which, unlike humans, process all retrieved documents as a whole
rather than sequentially. Additionally, traditional IR metrics do not account
for related but irrelevant documents that actively degrade generation quality,
rather than merely being ignored. Due to these two major misalignments, namely
human vs. machine position discount and human relevance vs. machine utility,
classical IR metrics do not accurately predict RAG performance. We introduce a
utility-based annotation schema that quantifies both the positive contribution
of relevant passages and the negative impact of distracting ones. Building on
this foundation, we propose UDCG (Utility and Distraction-aware Cumulative
Gain), a metric using an LLM-oriented positional discount to directly optimize
the correlation with the end-to-end answer accuracy. Experiments on five
datasets and six LLMs demonstrate that UDCG improves correlation by up to 36%
compared to traditional metrics. Our work provides a critical step toward
aligning IR evaluation with LLM consumers and enables more reliable assessment
of RAG components</details></td><td><details><summary>展开</summary>这篇论文探讨了传统信息检索（IR）指标在检索增强生成（RAG）系统中的局限性，指出这些指标因假设人类用户按顺序浏览文档而无法准确评估LLM处理检索结果的表现。作者提出了一种基于效用的标注框架（UDCG），量化相关文档的积极作用和干扰文档的负面影响，并通过实验证明UDCG与传统指标相比能显著提升与端到端答案准确性的相关性（最高达36%），为RAG系统的评估提供了更可靠的指标。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.21144v1">NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to
dynamically integrate external knowledge during inference, improving their
factual accuracy and adaptability. However, adversaries can inject poisoned
external knowledge to override the model's internal memory. While existing
attacks iteratively manipulate retrieval content or prompt structure of RAG,
they largely ignore the model's internal representation dynamics and
neuron-level sensitivities. The underlying mechanism of RAG poisoning has not
been fully studied and the effect of knowledge conflict with strong parametric
knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning,
a novel attack framework that generates adversarial external knowledge in RAG
guided by LLM internal neuron attribution and genetic optimization. Our method
first identifies a set of Poison-Responsive Neurons whose activation strongly
correlates with contextual poisoning knowledge. We then employ a genetic
algorithm to evolve adversarial passages that maximally activate these neurons.
Crucially, our framework enables massive-scale generation of effective poisoned
RAG knowledge by identifying and reusing promising but initially unsuccessful
external knowledge variants via observed attribution signals. At the same time,
Poison-Responsive Neurons guided poisoning can effectively resolves knowledge
conflict. Experimental results across models and datasets demonstrate
consistently achieving high Population Overwrite Success Rate (POSR) of over
90% while preserving fluency. Empirical evidence shows that our method
effectively resolves knowledge conflict.</details></td><td><details><summary>展开</summary>这篇论文探讨了RAG（检索增强生成）技术中外部知识被恶意注入（投毒攻击）的安全问题，提出了一种名为NeuroGenPoisoning的新型攻击框架。该框架通过分析大语言模型内部神经元激活与投毒知识的相关性，结合遗传算法生成对抗性外部知识，能够在保持文本流畅性的同时高效覆盖模型内部参数化知识（成功率超90%），并解决了知识冲突问题。实验验证了其在多模型和数据集上的有效性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.21093v1">MedAlign: A Synergistic Framework of Multimodal Preference Optimization and Federated Meta-Cognitive Reasoning</a></td><td><details><summary>展开</summary>Recently, large models have shown significant potential for smart healthcare.
However, the deployment of Large Vision-Language Models (LVLMs) for clinical
services is currently hindered by three critical challenges: a tendency to
hallucinate answers not grounded in visual evidence, the inefficiency of
fixed-depth reasoning, and the difficulty of multi-institutional collaboration.
To address these challenges, in this paper, we develop MedAlign, a novel
framework to ensure visually accurate LVLM responses for Medical Visual
Question Answering (Med-VQA). Specifically, we first propose a multimodal
Direct Preference Optimization (mDPO) objective to explicitly align preference
learning with visual context. We then design a Retrieval-Aware
Mixture-of-Experts (RA-MoE) architecture that utilizes image and text
similarity to route queries to a specialized and context-augmented LVLM (i.e.,
an expert), thereby mitigating hallucinations in LVLMs. To achieve adaptive
reasoning and facilitate multi-institutional collaboration, we propose a
federated governance mechanism, where the selected expert, fine-tuned on
clinical datasets based on mDPO, locally performs iterative Chain-of-Thought
(CoT) reasoning via the local meta-cognitive uncertainty estimator. Extensive
experiments on three representative Med-VQA datasets demonstrate that MedAlign
achieves state-of-the-art performance, outperforming strong retrieval-augmented
baselines by up to $11.85\%$ in F1-score, and simultaneously reducing the
average reasoning length by $51.60\%$ compared with fixed-depth CoT approaches.</details></td><td><details><summary>展开</summary>该论文提出MedAlign框架，通过多模态偏好优化（mDPO）和检索感知的专家混合架构（RA-MoE）增强医学视觉问答（Med-VQA）的准确性，利用检索机制选择专家模型以减少幻觉，并结合联邦治理实现自适应推理，显著提升性能并降低推理耗时。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.21068v1">Bridging Language Gaps with Adaptive RAG: Improving Indonesian Language Question Answering</a></td><td><details><summary>展开</summary>Question Answering (QA) has seen significant improvements with the
advancement of machine learning models, further studies enhanced this question
answering system by retrieving external information, called Retrieval-Augmented
Generation (RAG) to produce more accurate and informative answers. However,
these state-of-the-art-performance is predominantly in English language. To
address this gap we made an effort of bridging language gaps by incorporating
Adaptive RAG system to Indonesian language. Adaptive RAG system integrates a
classifier whose task is to distinguish the question complexity, which in turn
determines the strategy for answering the question. To overcome the limited
availability of Indonesian language dataset, our study employs machine
translation as data augmentation approach. Experiments show reliable question
complexity classifier; however, we observed significant inconsistencies in
multi-retrieval answering strategy which negatively impacted the overall
evaluation when this strategy was applied. These findings highlight both the
promise and challenges of question answering in low-resource language
suggesting directions for future improvement.</details></td><td><details><summary>展开</summary>该论文探讨了将自适应RAG系统应用于印尼语以弥补低资源语言问答系统的不足，通过问题复杂度分类器和机器翻译增强数据，但发现多检索策略存在不一致性影响整体性能，揭示了低资源语言问答的潜力与挑战。</details></td></tr></tbody></table>

### 📅 2025-10-23
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.20797v1">Simple Context Compression: Mean-Pooling and Multi-Ratio Training</a></td><td><details><summary>展开</summary>A common strategy to reduce the computational costs of using long contexts in
retrieval-augmented generation (RAG) with large language models (LLMs) is soft
context compression, where the input sequence is transformed into a shorter
continuous representation. We develop a lightweight and simple mean-pooling
approach that consistently outperforms the widely used compression-tokens
architecture, and study training the same compressor to output multiple
compression ratios. We conduct extensive experiments across in-domain and
out-of-domain QA datasets, as well as across model families, scales, and
compression ratios. Overall, our simple mean-pooling approach achieves the
strongest performance, with a relatively small drop when training for multiple
compression ratios. More broadly though, across architectures and training
regimes the trade-offs are more nuanced, illustrating the complex landscape of
compression methods.</details></td><td><details><summary>展开</summary>该论文提出了一种轻量级的均值池化方法，用于在检索增强生成（RAG）中压缩长上下文输入，以降低计算成本。通过实验验证，该方法在多种压缩比和数据集上表现优于现有技术，并探讨了多压缩比训练的权衡问题。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.20768v1">RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) has emerged as the dominant
architectural pattern to operationalize Large Language Model (LLM) usage in
Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to
poisoning attacks, and previously proposed defenses can fail for CTI contexts
as cyber threat information is often completely new for emerging attacks, and
sophisticated threat actors can mimic legitimate formats, terminology, and
stylistic conventions. To address this issue, we propose that the robustness of
modern RAG defenses can be accelerated by applying source credibility
algorithms on corpora, using PageRank as an example. In our experiments, we
demonstrate quantitatively that our algorithm applies a lower authority score
to malicious documents while promoting trusted content, using the standardized
MS MARCO dataset. We also demonstrate proof-of-concept performance of our
algorithm on CTI documents and feeds.</details></td><td><details><summary>展开</summary>这篇论文探讨了在网络安全威胁情报（CTI）系统中应用检索增强生成（RAG）技术时面临的投毒攻击风险，并提出通过基于PageRank的源可信度算法来增强RAG防御的鲁棒性，实验验证了该算法在区分恶意文档和可信内容上的有效性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.20609v1">Practical Code RAG at Scale: Task-Aware Retrieval Design Choices under Compute Budgets</a></td><td><details><summary>展开</summary>We study retrieval design for code-focused generation tasks under realistic
compute budgets. Using two complementary tasks from Long Code Arena -- code
completion and bug localization -- we systematically compare retrieval
configurations across various context window sizes along three axes: (i)
chunking strategy, (ii) similarity scoring, and (iii) splitting granularity.
(1) For PL-PL, sparse BM25 with word-level splitting is the most effective and
practical, significantly outperforming dense alternatives while being an order
of magnitude faster. (2) For NL-PL, proprietary dense encoders (Voyager-3
family) consistently beat sparse retrievers, however requiring 100x larger
latency. (3) Optimal chunk size scales with available context: 32-64 line
chunks work best at small budgets, and whole-file retrieval becomes competitive
at 16000 tokens. (4) Simple line-based chunking matches syntax-aware splitting
across budgets. (5) Retrieval latency varies by up to 200x across
configurations; BPE-based splitting is needlessly slow, and BM25 + word
splitting offers the best quality-latency trade-off. Thus, we provide
evidence-based recommendations for implementing effective code-oriented RAG
systems based on task requirements, model constraints, and computational
efficiency.</details></td><td><details><summary>展开</summary>这篇论文研究代码相关生成任务（如代码补全和缺陷定位）中的检索设计，通过比较不同检索配置（分块策略、相似性评分和分割粒度）在有限计算资源下的表现，为代码导向的RAG系统提供了基于实证的优化建议，包括稀疏检索（BM25）与密集检索的适用场景、最佳分块大小及效率权衡方案。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.20548v1">GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning</a></td><td><details><summary>展开</summary>Reinforcement learning has recently shown promise in improving
retrieval-augmented generation (RAG). Despite these advances, its effectiveness
in multi-hop question answering (QA) remains limited by two fundamental
limitations: (i) global planning absence to structure multi-step reasoning, and
(ii) unfaithful execution, which hinders effective query formulation and
consistent use of retrieved evidence. We propose GlobalRAG, a reinforcement
learning framework designed to enhance global reasoning in multi-hop QA.
GlobalRAG decomposes questions into subgoals, coordinates retrieval with
reasoning, and refines evidence iteratively. To guide this process, we
introduce Planning Quality Reward and SubGoal Completion Reward, which
encourage coherent planning and reliable subgoal execution. In addition, a
progressive weight annealing strategy balances process-oriented and
outcome-based objectives. Extensive experiments on both in-domain and
out-of-domain benchmarks demonstrate that GlobalRAG significantly outperforms
strong baselines while using only 8k training data (42% of the training data
used by strong baselines), achieving average improvements of 14.2% in both EM
and F1.</details></td><td><details><summary>展开</summary>该论文提出了一种名为GlobalRAG的强化学习框架，旨在解决多跳问答（QA）中RAG技术的两大局限——缺乏全局规划和执行不忠实的问题。通过分解问题为子目标、协调检索与推理、迭代优化证据，并结合规划质量奖励和子目标完成奖励，该框架显著提升了性能，实验显示其在少量训练数据下优于基线模型。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.20535v1">ARC-Encoder: learning compressed text representations for large language models</a></td><td><details><summary>展开</summary>Recent techniques such as retrieval-augmented generation or chain-of-thought
reasoning have led to longer contexts and increased inference costs. Context
compression techniques can reduce these costs, but the most effective
approaches require fine-tuning the target model or even modifying its
architecture. This can degrade its general abilities when not used for this
specific purpose. Here we explore an alternative approach: an encoder that
compresses the context into continuous representations which replace token
embeddings in decoder LLMs. First, we perform a systematic study of training
strategies and architecture choices for the encoder. Our findings led to the
design of an Adaptable text Representations Compressor, named ARC-Encoder,
which outputs $x$-times fewer continuous representations (typically
$x\!\in\!\{4,8\}$) than text tokens. We evaluate ARC-Encoder across a variety
of LLM usage scenarios, ranging from in-context learning to context window
extension, on both instruct and base decoders. Results show that ARC-Encoder
achieves state-of-the-art performance on several benchmarks while improving
computational efficiency at inference. Finally, we demonstrate that our models
can be adapted to multiple decoders simultaneously, allowing a single encoder
to generalize across different decoder LLMs. This makes ARC-Encoder a flexible
and efficient solution for portable encoders that work seamlessly with multiple
LLMs. We release a training code at https://github.com/kyutai-labs/ARC-Encoder
, fine-tuning dataset and pretrained models are available at
https://huggingface.co/collections/kyutai/arc-encoders-68ee18787301407d60a57047 .</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为ARC-Encoder的上下文压缩技术，旨在解决RAG和思维链推理等技术带来的长上下文和高推理成本问题。通过将上下文压缩为更少的连续表示来替换解码器LLM中的令牌嵌入，ARC-Encoder在保持性能的同时提高了计算效率，并能适配多种解码器LLM。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.20505v1">Hierarchical Sequence Iteration for Heterogeneous Question Answering</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) remains brittle on multi-step questions
and heterogeneous evidence sources, trading accuracy against latency and
token/tool budgets. This paper introducesHierarchical Sequence (HSEQ) Iteration
for Heterogeneous Question Answering, a unified framework that (i) linearize
documents, tables, and knowledge graphs into a reversible hierarchical sequence
with lightweight structural tags, and (ii) perform structure-aware iteration to
collect just-enough evidence before answer synthesis. A Head Agent provides
guidance that leads retrieval, while an Iteration Agent selects and expands
HSeq via structure-respecting actions (e.g., parent/child hops, table
row/column neighbors, KG relations); Finally the head agent composes
canonicalized evidence to genearte the final answer, with an optional
refinement loop to resolve detected contradictions. Experiments on HotpotQA
(text), HybridQA/TAT-QA (table+text), and MetaQA (KG) show consistent EM/F1
gains over strong single-pass, multi-hop, and agentic RAG baselines with high
efficiency. Besides, HSEQ exhibits three key advantages: (1) a format-agnostic
unification that enables a single policy to operate across text, tables, and
KGs without per-dataset specialization; (2) guided, budget-aware iteration that
reduces unnecessary hops, tool calls, and tokens while preserving accuracy; and
(3) evidence canonicalization for reliable QA, improving answers consistency
and auditability.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为HSEQ Iteration的检索增强生成（RAG）改进框架，通过将文档、表格和知识图谱线性化为可逆的层次序列，并结合结构感知的迭代检索策略，以高效收集足够证据并生成更准确的答案。实验表明，该方法在多种问答任务中优于现有RAG基线，具有跨格式统一性、预算感知迭代和证据规范化三大优势。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.20356v1">FreeChunker: A Cross-Granularity Chunking Framework</a></td><td><details><summary>展开</summary>Chunking strategies significantly impact the effectiveness of
Retrieval-Augmented Generation (RAG) systems. Existing methods operate within
fixed-granularity paradigms that rely on static boundary identification,
limiting their adaptability to diverse query requirements. This paper presents
FreeChunker, a Cross-Granularity Encoding Framework that fundamentally
transforms the traditional chunking paradigm: the framework treats sentences as
atomic units and shifts from static chunk segmentation to flexible retrieval
supporting arbitrary sentence combinations. This paradigm shift not only
significantly reduces the computational overhead required for semantic boundary
detection but also enhances adaptability to complex queries. Experimental
evaluation on LongBench V2 demonstrates that FreeChunker achieves superior
retrieval performance compared to traditional chunking methods, while
significantly outperforming existing approaches in computational efficiency.</details></td><td><details><summary>展开</summary>本文提出了一种名为FreeChunker的跨粒度编码框架，通过将句子作为原子单元并支持灵活的句子组合检索，改进了传统RAG系统中固定粒度的分块策略，显著降低了语义边界检测的计算开销，提升了对复杂查询的适应性，并在LongBench V2实验中表现出优于传统方法的检索性能和计算效率。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.20303v1">Citation Failure: Definition, Analysis and Efficient Mitigation</a></td><td><details><summary>展开</summary>Citations from LLM-based RAG systems are supposed to simplify response
verification. However, this does not hold for citation failure, when a model
generates a helpful response, but fails to cite complete evidence. In contrast
to previous work, we propose to disentangle this from response failure, where
the response itself is flawed, and citing complete evidence is impossible. To
address citation failure, this work follows a two-step approach: (1) We study
when citation failure occurs and (2) how it can be mitigated. For step 1, we
extend prior work by investigating how the relation between response and
evidence affects citation quality. We introduce CITECONTROL, a benchmark that
systematically varies this relation to analyze failure modes. Experiments show
that failures increase with relational complexity and suggest that combining
citation methods could improve performance, motivating step 2. To improve LLM
citation efficiently, we propose CITENTION, a framework integrating generative,
attention-based, and retrieval-based methods. Results demonstrate substantial
citation improvements on CITECONTROL and in transfer settings. We make our data
and code publicly available.</details></td><td><details><summary>展开</summary>这篇论文探讨了基于LLM的RAG系统中存在的“引用失败”问题（即模型生成有用回答但未完整引用证据），提出将其与“回答失败”区分，并通过CITECONTROL基准分析失败模式与证据-回答关系，最终提出整合生成、注意力与检索方法的CITENTION框架以提升引用效果。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.20296v1">RAG-Stack: Co-Optimizing RAG Quality and Performance From the Vector Database Perspective</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) has emerged as one of the most prominent
applications of vector databases. By integrating documents retrieved from a
database into the prompt of a large language model (LLM), RAG enables more
reliable and informative content generation. While there has been extensive
research on vector databases, many open research problems remain once they are
considered in the wider context of end-to-end RAG pipelines. One practical yet
challenging problem is how to jointly optimize both system performance and
generation quality in RAG, which is significantly more complex than it appears
due to the numerous knobs on both the algorithmic side (spanning models and
databases) and the systems side (from software to hardware). In this paper, we
present RAG-Stack, a three-pillar blueprint for quality-performance
co-optimization in RAG systems. RAG-Stack comprises: (1) RAG-IR, an
intermediate representation that serves as an abstraction layer to decouple
quality and performance aspects; (2) RAG-CM, a cost model for estimating system
performance given an RAG-IR; and (3) RAG-PE, a plan exploration algorithm that
searches for high-quality, high-performance RAG configurations. We believe this
three-pillar blueprint will become the de facto paradigm for RAG
quality-performance co-optimization in the years to come.</details></td><td><details><summary>展开</summary>这篇论文探讨了检索增强生成（RAG）技术，提出了一种名为RAG-Stack的三支柱蓝图，旨在共同优化RAG系统的性能与生成质量，包括RAG-IR中间表示、RAG-CM成本模型和RAG-PE计划探索算法，以解决现有研究中的挑战。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.20279v1">ResearchGPT: Benchmarking and Training LLMs for End-to-End Computer Science Research Workflows</a></td><td><details><summary>展开</summary>As large language models (LLMs) advance, the ultimate vision for their role
in science is emerging: we could build an AI collaborator to effectively assist
human beings throughout the entire scientific research process. We refer to
this envisioned system as ResearchGPT. Given that scientific research
progresses through multiple interdependent phases, achieving this vision
requires rigorous benchmarks that evaluate the end-to-end workflow rather than
isolated sub-tasks. To this end, we contribute CS-54k, a high-quality corpus of
scientific Q&A pairs in computer science, built from 14k CC-licensed papers. It
is constructed through a scalable, paper-grounded pipeline that combines
retrieval-augmented generation (RAG) with multi-stage quality control to ensure
factual grounding. From this unified corpus, we derive two complementary
subsets: CS-4k, a carefully curated benchmark for evaluating AI's ability to
assist scientific research, and CS-50k, a large-scale training dataset.
Extensive experiments demonstrate that CS-4k stratifies state-of-the-art LLMs
into distinct capability tiers. Open models trained on CS-50k with supervised
training and reinforcement learning demonstrate substantial improvements. Even
7B-scale models, when properly trained, outperform many larger proprietary
systems, such as GPT-4.1, GPT-4o, and Gemini 2.5 Pro. This indicates that
making AI models better research assistants relies more on domain-aligned
training with high-quality data than on pretraining scale or general benchmark
performance. We release CS-4k and CS-50k in the hope of fostering AI systems as
reliable collaborators in CS research.</details></td><td><details><summary>展开</summary>这篇论文提出ResearchGPT愿景，通过构建CS-54k科学问答语料库（含14k论文）支持AI辅助科研，其数据生成采用RAG技术确保事实性，并拆分出评估基准CS-4k与训练集CS-50k。实验表明，基于该数据训练的7B模型可超越GPT-4等商用系统，凸显领域对齐数据的重要性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.20260v1">Balancing Fine-tuning and RAG: A Hybrid Strategy for Dynamic LLM Recommendation Updates</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) empower recommendation systems through their
advanced reasoning and planning capabilities. However, the dynamic nature of
user interests and content poses a significant challenge: While initial
fine-tuning aligns LLMs with domain knowledge and user preferences, it fails to
capture such real-time changes, necessitating robust update mechanisms. This
paper investigates strategies for updating LLM-powered recommenders, focusing
on the trade-offs between ongoing fine-tuning and Retrieval-Augmented
Generation (RAG). Using an LLM-powered user interest exploration system as a
case study, we perform a comparative analysis of these methods across
dimensions like cost, agility, and knowledge incorporation. We propose a hybrid
update strategy that leverages the long-term knowledge adaptation of periodic
fine-tuning with the agility of low-cost RAG. We demonstrate through live A/B
experiments on a billion-user platform that this hybrid approach yields
statistically significant improvements in user satisfaction, offering a
practical and cost-effective framework for maintaining high-quality LLM-powered
recommender systems.</details></td><td><details><summary>展开</summary>本文探讨了在基于大语言模型（LLM）的推荐系统中，如何通过持续微调和检索增强生成（RAG）两种策略动态适应用户兴趣和内容变化，提出了一种结合定期微调与低成本RAG的混合更新方法，并通过大规模A/B实验验证其显著提升用户满意度。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.20193v1">Multimedia-Aware Question Answering: A Review of Retrieval and Cross-Modal Reasoning Architectures</a></td><td><details><summary>展开</summary>Question Answering (QA) systems have traditionally relied on structured text
data, but the rapid growth of multimedia content (images, audio, video, and
structured metadata) has introduced new challenges and opportunities for
retrieval-augmented QA. In this survey, we review recent advancements in QA
systems that integrate multimedia retrieval pipelines, focusing on
architectures that align vision, language, and audio modalities with user
queries. We categorize approaches based on retrieval methods, fusion
techniques, and answer generation strategies, and analyze benchmark datasets,
evaluation protocols, and performance tradeoffs. Furthermore, we highlight key
challenges such as cross-modal alignment, latency-accuracy tradeoffs, and
semantic grounding, and outline open problems and future research directions
for building more robust and context-aware QA systems leveraging multimedia
data.</details></td><td><details><summary>展开</summary>这篇论文综述了结合多媒体检索（如图像、音频、视频）的问答系统（QA）的最新进展，聚焦于通过跨模态对齐（视觉、语言、音频）增强检索与生成的架构，探讨了检索方法、融合技术、生成策略及挑战（如跨模态对齐、语义 grounding），属于RAG技术在多媒体领域的扩展应用。</details></td></tr></tbody></table>

### 📅 2025-10-22
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.19723v1">From Answers to Guidance: A Proactive Dialogue System for Legal Documents</a></td><td><details><summary>展开</summary>The accessibility of legal information remains a constant challenge,
particularly for laypersons seeking to understand and apply complex
institutional texts. While the European Union provides open access to
legislation, parliamentary responses, and regulatory documents, these resources
can be challenging for laypeople to explore. In this paper, we introduce
EUDial, a proactive multi-turn dialogue dataset constructed from 204 blogs
curated by the Citizens' Enquiries Unit (AskEP) of the European Parliamentary
Research Service. EUDial contains 880 dialogue turns (averaging 4.3 turns per
dialogue), where each dialogue includes initial questions, structured answers,
and follow-up questions. Beyond dataset construction, we propose the LexGuide
framework that leverages retrieval-augmented generation with hierarchical topic
organization to structure dialogue progression, ensuring both comprehensive
coverage of legal aspects and coherence across conversational turns. The
results demonstrate that proactive, structured navigation closes the gap
between the availability of legal information and citizen comprehension,
establishing EUDial and LexGuide as practical resources for advancing proactive
legal dialogue systems.</details></td><td><details><summary>展开</summary>这篇论文介绍了EUDial数据集和LexGuide框架，通过检索增强生成（RAG）结合层次化主题组织，构建主动多轮法律对话系统，帮助非专业人士理解欧盟复杂法律文本，提升信息可及性与对话连贯性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.19670v1">CoSense-LLM: Semantics at the Edge with Cost- and Uncertainty-Aware Cloud-Edge Cooperation</a></td><td><details><summary>展开</summary>We present CoSense-LLM, an edge-first framework that turns continuous
multimodal sensor streams (for example Wi-Fi CSI, IMU, audio, RFID, and
lightweight vision) into compact, verifiable semantic tokens and coordinates
with large language models under explicit latency, energy, bandwidth, and
privacy constraints. CoSense-LLM has four parts: (i) SenseFusion, a lightweight
encoder that aligns sensor embeddings with language and compresses them into
short discrete code sequences; (ii) Edge-RAG, a local hybrid retrieval layer
that grounds generation in site specific policies and notes; (iii)
PromptRouter, a cost and uncertainty aware policy that selects edge only
generation, edge plus retrieval, or compact cloud escalation; and (iv) Secure
Execution, an auditable redaction path that enforces data minimization so raw
waveforms never leave the device. The system works with modern serving
optimizations, including paged or streaming KV caches, FlashAttention style
kernels, speculative decoding, and quantized LoRA adapters, and supports on
device personalization and federated updates under non IID drift. Across home,
office, and clinic deployments, CoSense-LLM delivers grounded explanations
while meeting tight service level objectives: it sustains sub second (p95) end
to end latency on edge dominant paths, reduces inter tier token and bandwidth
costs by preferring local retrieval grounded responses, and preserves privacy
by transmitting only discrete codes and redacted metadata. Ablations show that
Edge-RAG improves factual consistency and reduces contradictions, calibrated
uncertainty enables selective abstention and controlled escalations, and KV
plus decoding accelerators lower energy per decision. The results support an
edge first design that treats semantics, privacy, and predictable latency as co
equal goals for large model deployments in interference prone environments.</details></td><td><details><summary>展开</summary>这篇文章提出了CoSense-LLM框架，通过结合轻量级传感器编码、本地混合检索层（Edge-RAG）和成本感知策略，将多模态传感器数据转化为语义标记并与大语言模型协同工作。Edge-RAG作为核心组件，通过本地检索增强生成过程，确保回答基于特定场景策略和记录，提高了事实一致性并减少矛盾，同时满足延迟、隐私和带宽约束。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.19644v1">LLavaCode: Compressed Code Representations for Retrieval-Augmented Code Generation</a></td><td><details><summary>展开</summary>Retrieval-augmented generation has emerged as one of the most effective
approaches for code completion, particularly when context from a surrounding
repository is essential. However, incorporating context significantly extends
sequence length, leading to slower inference - a critical limitation for
interactive settings such as IDEs. In this work, we introduce LlavaCode, a
framework that compresses code into compact, semantically rich representations
interpretable by code LLM, enhancing generation quality while reducing the
retrieved context to only a few compressed single-token vectors. Using a small
projector module we can significantly increase the EM and ES metrics of coding
model with negligible latency increase. Our experiments demonstrate that
compressed context enables 20-38% reduction in Time-to-First-Token (TTFT) on
line completion tasks compared to full-RAG pipelines.</details></td><td><details><summary>展开</summary>这篇论文介绍了LlavaCode框架，通过将代码压缩成紧凑的语义丰富表示来优化RAG在代码补全中的应用，减少了检索上下文的长度，从而提升生成质量并降低延迟，实验显示其能显著减少Time-to-First-Token（TTFT）。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.19331v1">Algorithmic Fairness in NLP: Persona-Infused LLMs for Human-Centric Hate Speech Detection</a></td><td><details><summary>展开</summary>In this paper, we investigate how personalising Large Language Models
(Persona-LLMs) with annotator personas affects their sensitivity to hate
speech, particularly regarding biases linked to shared or differing identities
between annotators and targets. To this end, we employ Google's Gemini and
OpenAI's GPT-4.1-mini models and two persona-prompting methods: shallow persona
prompting and a deeply contextualised persona development based on
Retrieval-Augmented Generation (RAG) to incorporate richer persona profiles. We
analyse the impact of using in-group and out-group annotator personas on the
models' detection performance and fairness across diverse social groups. This
work bridges psychological insights on group identity with advanced NLP
techniques, demonstrating that incorporating socio-demographic attributes into
LLMs can address bias in automated hate speech detection. Our results highlight
both the potential and limitations of persona-based approaches in reducing
bias, offering valuable insights for developing more equitable hate speech
detection systems.</details></td><td><details><summary>展开</summary>该论文研究了通过个性化大语言模型（Persona-LLMs）来提升对仇恨言论的敏感性，特别关注了注释者与目标群体身份异同带来的偏见。研究采用了Google的Gemini和OpenAI的GPT-4.1-mini模型，并使用了两种人物提示方法，其中一种基于检索增强生成（RAG）来整合更丰富的人物档案。论文分析了使用内群体和外群体注释者人物对模型检测性能和公平性的影响，展示了将社会人口属性融入大语言模型以减少自动仇恨言论检测中的偏见的潜力与局限性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.19171v1">Think Straight, Stop Smart: Structured Reasoning for Efficient Multi-Hop RAG</a></td><td><details><summary>展开</summary>Multi-hop retrieval-augmented generation (RAG) is a promising strategy for
complex reasoning, yet existing iterative prompting approaches remain
inefficient. They often regenerate predictable token sequences at every step
and rely on stochastic stopping, leading to excessive token usage and unstable
termination. We propose TSSS (Think Straight, Stop Smart), a structured
multi-hop RAG framework designed for efficiency. TSSS introduces (i) a
template-based reasoning that caches recurring prefixes and anchors sub-queries
to the main question, reducing token generation cost while promoting stable
reasoning, and (ii) a retriever-based terminator, which deterministically halts
reasoning once additional sub-queries collapse into repetition. This separation
of structured reasoning and termination control enables both faster inference
and more reliable answers. On HotpotQA, 2WikiMultiHop, and MuSiQue, TSSS
achieves state-of-the-art accuracy and competitive efficiency among RAG-CoT
approaches, highlighting its effectiveness in efficiency-constrained scenarios
such as on-device inference.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为TSSS（Think Straight, Stop Smart）的高效多跳检索增强生成（RAG）框架，通过模板化推理减少重复生成的令牌成本，并引入基于检索器的终止机制以稳定结束推理，在多个数据集上实现了高准确性和效率。</details></td></tr></tbody></table>

### 📅 2025-10-21
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.18821v1">Search Self-play: Pushing the Frontier of Agent Capability without Supervision</a></td><td><details><summary>展开</summary>Reinforcement learning with verifiable rewards (RLVR) has become the
mainstream technique for training LLM agents. However, RLVR highly depends on
well-crafted task queries and corresponding ground-truth answers to provide
accurate rewards, which requires massive human efforts and hinders the RL
scaling processes, especially under agentic scenarios. Although a few recent
works explore task synthesis methods, the difficulty of generated agentic tasks
can hardly be controlled to provide effective RL training advantages. To
achieve agentic RLVR with higher scalability, we explore self-play training for
deep search agents, in which the learning LLM utilizes multi-turn search engine
calling and acts simultaneously as both a task proposer and a problem solver.
The task proposer aims to generate deep search queries with well-defined
ground-truth answers and increasing task difficulty. The problem solver tries
to handle the generated search queries and output the correct answer
predictions. To ensure that each generated search query has accurate ground
truth, we collect all the searching results from the proposer's trajectory as
external knowledge, then conduct retrieval-augmentation generation (RAG) to
test whether the proposed query can be correctly answered with all necessary
search documents provided. In this search self-play (SSP) game, the proposer
and the solver co-evolve their agent capabilities through both competition and
cooperation. With substantial experimental results, we find that SSP can
significantly improve search agents' performance uniformly on various
benchmarks without any supervision under both from-scratch and continuous RL
training setups. The code is at https://github.com/Alibaba-Quark/SSP.</details></td><td><details><summary>展开</summary>本文提出了一种基于自我对弈训练（SSP）的深度搜索代理方法，通过让大语言模型同时充当任务提出者和问题解决者，生成具有明确答案和递增难度的搜索查询，并利用检索增强生成（RAG）技术验证查询的可回答性，从而在无监督情况下显著提升搜索代理的性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.18691v1">Investigating LLM Capabilities on Long Context Comprehension for Medical Question Answering</a></td><td><details><summary>展开</summary>This study is the first to investigate LLM comprehension capabilities over
long-context (LC) medical QA of clinical relevance. Our comprehensive
assessment spans a range of content-inclusion settings based on their
relevance, LLM models of varying capabilities and datasets across task
formulations, revealing insights on model size effects, limitations, underlying
memorization issues and the benefits of reasoning models. Importantly, we
examine the effect of RAG on medical LC comprehension, uncover best settings in
single versus multi-document reasoning datasets and showcase RAG strategies for
improvements over LC. We shed light into some of the evaluation aspects using a
multi-faceted approach. Our qualitative and error analyses address open
questions on when RAG is beneficial over LC, revealing common failure cases.</details></td><td><details><summary>展开</summary>该论文首次研究了大语言模型（LLM）在长上下文（LC）医学问答中的理解能力，评估了不同模型、数据集和任务设置下的表现，探讨了模型大小、记忆问题及推理模型的优势，重点分析了RAG对医学LC理解的影响，比较了单文档与多文档推理的最佳设置，并提出了改进RAG的策略，同时通过多角度评估揭示了RAG的适用场景和常见失败案例。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.18633v1">Query Decomposition for RAG: Balancing Exploration-Exploitation</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) systems address complex user requests by
decomposing them into subqueries, retrieving potentially relevant documents for
each, and then aggregating them to generate an answer. Efficiently selecting
informative documents requires balancing a key trade-off: (i) retrieving
broadly enough to capture all the relevant material, and (ii) limiting
retrieval to avoid excessive noise and computational cost. We formulate query
decomposition and document retrieval in an exploitation-exploration setting,
where retrieving one document at a time builds a belief about the utility of a
given sub-query and informs the decision to continue exploiting or exploring an
alternative. We experiment with a variety of bandit learning methods and
demonstrate their effectiveness in dynamically selecting the most informative
sub-queries. Our main finding is that estimating document relevance using rank
information and human judgments yields a 35% gain in document-level precision,
15% increase in {\alpha}-nDCG, and better performance on the downstream task of
long-form generation.</details></td><td><details><summary>展开</summary>这篇论文探讨了在RAG系统中如何通过分解用户查询、动态检索文档并平衡检索广度与噪声的问题，提出了一种基于探索-利用策略的bandit学习方法，利用排序信息和人工评估提升文档选择效率，实验表明该方法显著提高了文档检索精度和下游长文本生成任务的表现。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.18502v1">Zero-Shot Vehicle Model Recognition via Text-Based Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Vehicle make and model recognition (VMMR) is an important task in intelligent
transportation systems, but existing approaches struggle to adapt to newly
released models. Contrastive Language-Image Pretraining (CLIP) provides strong
visual-text alignment, yet its fixed pretrained weights limit performance
without costly image-specific finetuning. We propose a pipeline that integrates
vision language models (VLMs) with Retrieval-Augmented Generation (RAG) to
support zero-shot recognition through text-based reasoning. A VLM converts
vehicle images into descriptive attributes, which are compared against a
database of textual features. Relevant entries are retrieved and combined with
the description to form a prompt, and a language model (LM) infers the make and
model. This design avoids large-scale retraining and enables rapid updates by
adding textual descriptions of new vehicles. Experiments show that the proposed
method improves recognition by nearly 20% over the CLIP baseline, demonstrating
the potential of RAG-enhanced LM reasoning for scalable VMMR in smart-city
applications.</details></td><td><details><summary>展开</summary>该论文提出了一种结合视觉语言模型（VLM）和检索增强生成（RAG）的流程，用于零样本车辆品牌和型号识别（VMMR）。通过将车辆图像转换为描述性属性并与文本特征数据库比对，检索相关信息后生成提示，由语言模型推断结果。该方法避免了大规模重新训练，支持快速更新，实验显示识别率比CLIP基线提升近20%。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.18468v1">IMB: An Italian Medical Benchmark for Question Answering</a></td><td><details><summary>展开</summary>Online medical forums have long served as vital platforms where patients seek
professional healthcare advice, generating vast amounts of valuable knowledge.
However, the informal nature and linguistic complexity of forum interactions
pose significant challenges for automated question answering systems,
especially when dealing with non-English languages. We present two
comprehensive Italian medical benchmarks: \textbf{IMB-QA}, containing 782,644
patient-doctor conversations from 77 medical categories, and \textbf{IMB-MCQA},
comprising 25,862 multiple-choice questions from medical specialty
examinations. We demonstrate how Large Language Models (LLMs) can be leveraged
to improve the clarity and consistency of medical forum data while retaining
their original meaning and conversational style, and compare a variety of LLM
architectures on both open and multiple-choice question answering tasks. Our
experiments with Retrieval Augmented Generation (RAG) and domain-specific
fine-tuning reveal that specialized adaptation strategies can outperform
larger, general-purpose models in medical question answering tasks. These
findings suggest that effective medical AI systems may benefit more from domain
expertise and efficient information retrieval than from increased model scale.
We release both datasets and evaluation frameworks in our GitHub repository to
support further research on multilingual medical question answering:
https://github.com/PRAISELab-PicusLab/IMB.</details></td><td><details><summary>展开</summary>该论文介绍了两个意大利医学基准数据集（IMB-QA和IMB-MCQA），探讨了利用大语言模型（LLMs）提升医学论坛数据的清晰度与一致性的方法，并通过实验对比了RAG与领域微调在医学问答任务中的表现，发现领域适配策略优于通用大模型，最终开源了数据集和评估框架以支持多语言医学问答研究。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.18455v1">ChronoPlay: A Framework for Modeling Dual Dynamics and Authenticity in Game RAG Benchmarks</a></td><td><details><summary>展开</summary>Retrieval Augmented Generation (RAG) systems are increasingly vital in
dynamic domains like online gaming, yet the lack of a dedicated benchmark has
impeded standardized evaluation in this area. The core difficulty lies in Dual
Dynamics: the constant interplay between game content updates and the shifting
focus of the player community. Furthermore, the necessity of automating such a
benchmark introduces a critical requirement for player-centric authenticity to
ensure generated questions are realistic. To address this integrated challenge,
we introduce ChronoPlay, a novel framework for the automated and continuous
generation of game RAG benchmarks. ChronoPlay utilizes a dual-dynamic update
mechanism to track both forms of change, and a dual-source synthesis engine
that draws from official sources and player community to ensure both factual
correctness and authentic query patterns. We instantiate our framework on three
distinct games to create the first dynamic RAG benchmark for the gaming domain,
offering new insights into model performance under these complex and realistic
conditions. Code is avaliable at: https://github.com/hly1998/ChronoPlay.</details></td><td><details><summary>展开</summary>该论文提出了一种名为ChronoPlay的自动化动态RAG基准生成框架，针对在线游戏领域设计，通过双动态更新机制和双源合成引擎（官方资料与玩家社区）解决游戏内容更新与玩家关注点变化的双重挑战，并构建了首个游戏领域的动态RAG基准，用于评估模型在复杂现实条件下的性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.18355v1">KrishokBondhu: A Retrieval-Augmented Voice-Based Agricultural Advisory Call Center for Bengali Farmers</a></td><td><details><summary>展开</summary>In Bangladesh, many farmers continue to face challenges in accessing timely,
expert-level agricultural guidance. This paper presents KrishokBondhu, a
voice-enabled, call-centre-integrated advisory platform built on a
Retrieval-Augmented Generation (RAG) framework, designed specifically for
Bengali-speaking farmers. The system aggregates authoritative agricultural
handbooks, extension manuals, and NGO publications; applies Optical Character
Recognition (OCR) and document-parsing pipelines to digitize and structure the
content; and indexes this corpus in a vector database for efficient semantic
retrieval. Through a simple phone-based interface, farmers can call the system
to receive real-time, context-aware advice: speech-to-text converts the Bengali
query, the RAG module retrieves relevant content, a large language model (Gemma
3-4B) generates a context-grounded response, and text-to-speech delivers the
answer in natural spoken Bengali. In a pilot evaluation, KrishokBondhu produced
high-quality responses for 72.7% of diverse agricultural queries covering crop
management, disease control, and cultivation practices. Compared to the
KisanQRS benchmark, the system achieved a composite score of 4.53 (vs. 3.13) on
a 5-point scale, a 44.7% improvement, with especially large gains in contextual
richness (+367%) and completeness (+100.4%), while maintaining comparable
relevance and technical specificity. Semantic similarity analysis further
revealed a strong correlation between retrieved context and answer quality,
emphasizing the importance of grounding generative responses in curated
documentation. KrishokBondhu demonstrates the feasibility of integrating
call-centre accessibility, multilingual voice interaction, and modern RAG
techniques to deliver expert-level agricultural guidance to remote Bangladeshi
farmers, paving the way toward a fully AI-driven agricultural advisory
ecosystem.</details></td><td><details><summary>展开</summary>这篇论文介绍了KrishokBondhu，一个基于RAG框架的语音农业咨询平台，专为孟加拉语农民设计，通过整合权威农业资料、OCR技术和语音交互，提供实时农业建议，并在试点评估中展现出高质量回答和显著性能提升。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.18339v1">ECG-LLM-- training and evaluation of domain-specific large language models for electrocardiography</a></td><td><details><summary>展开</summary>Domain-adapted open-weight large language models (LLMs) offer promising
healthcare applications, from queryable knowledge bases to multimodal
assistants, with the crucial advantage of local deployment for privacy
preservation. However, optimal adaptation strategies, evaluation methodologies,
and performance relative to general-purpose LLMs remain poorly characterized.
We investigated these questions in electrocardiography, an important area of
cardiovascular medicine, by finetuning open-weight models on domain-specific
literature and implementing a multi-layered evaluation framework comparing
finetuned models, retrieval-augmented generation (RAG), and Claude Sonnet 3.7
as a representative general-purpose model. Finetuned Llama 3.1 70B achieved
superior performance on multiple-choice evaluations and automatic text metrics,
ranking second to Claude 3.7 in LLM-as-a-judge assessments. Human expert
evaluation favored Claude 3.7 and RAG approaches for complex queries. Finetuned
models significantly outperformed their base counterparts across nearly all
evaluation modes. Our findings reveal substantial performance heterogeneity
across evaluation methodologies, underscoring assessment complexity.
Nevertheless, domain-specific adaptation through finetuning and RAG achieves
competitive performance with proprietary models, supporting the viability of
privacy-preserving, locally deployable clinical solutions.</details></td><td><details><summary>展开</summary>这篇论文研究了在医疗领域（特别是心电图学）中，通过微调开源大语言模型（LLMs）和采用检索增强生成（RAG）技术来提升模型性能的方法。研究比较了微调模型、RAG方法和通用模型Claude Sonnet 3.7的表现，发现微调模型在多项评估中表现优异，而RAG和Claude 3.7在复杂查询中更受专家青睐。论文强调了评估方法的复杂性，并证明了领域特定适配（包括RAG）在隐私保护型本地临床解决方案中的可行性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.18297v1">From Retrieval to Generation: Unifying External and Parametric Knowledge for Medical Question Answering</a></td><td><details><summary>展开</summary>Medical question answering (QA) requires extensive access to domain-specific
knowledge. A promising direction is to enhance large language models (LLMs)
with external knowledge retrieved from medical corpora or parametric knowledge
stored in model parameters. Existing approaches typically fall into two
categories: Retrieval-Augmented Generation (RAG), which grounds model reasoning
on externally retrieved evidence, and Generation-Augmented Generation (GAG),
which depends solely on the models internal knowledge to generate contextual
documents. However, RAG often suffers from noisy or incomplete retrieval, while
GAG is vulnerable to hallucinated or inaccurate information due to
unconstrained generation. Both issues can mislead reasoning and undermine
answer reliability. To address these challenges, we propose MedRGAG, a unified
retrieval-generation augmented framework that seamlessly integrates external
and parametric knowledge for medical QA. MedRGAG comprises two key modules:
Knowledge-Guided Context Completion (KGCC), which directs the generator to
produce background documents that complement the missing knowledge revealed by
retrieval; and Knowledge-Aware Document Selection (KADS), which adaptively
selects an optimal combination of retrieved and generated documents to form
concise yet comprehensive evidence for answer generation. Extensive experiments
on five medical QA benchmarks demonstrate that MedRGAG achieves a 12.5%
improvement over MedRAG and a 4.5% gain over MedGENIE, highlighting the
effectiveness of unifying retrieval and generation for knowledge-intensive
reasoning. Our code and data are publicly available at
https://anonymous.4open.science/r/MedRGAG</details></td><td><details><summary>展开</summary>该论文提出MedRGAG框架，通过整合外部检索（RAG）和内部生成（GAG）的医学知识，结合知识引导的上下文补全（KGCC）和知识感知文档选择（KADS）模块，显著提升医疗问答的准确性和可靠性，实验证明其性能优于纯检索或纯生成方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.18204v1">RESCUE: Retrieval Augmented Secure Code Generation</a></td><td><details><summary>展开</summary>Despite recent advances, Large Language Models (LLMs) still generate
vulnerable code. Retrieval-Augmented Generation (RAG) has the potential to
enhance LLMs for secure code generation by incorporating external security
knowledge. However, the conventional RAG design struggles with the noise of raw
security-related documents, and existing retrieval methods overlook the
significant security semantics implicitly embedded in task descriptions. To
address these issues, we propose RESCUE, a new RAG framework for secure code
generation with two key innovations. First, we propose a hybrid knowledge base
construction method that combines LLM-assisted cluster-then-summarize
distillation with program slicing, producing both high-level security
guidelines and concise, security-focused code examples. Second, we design a
hierarchical multi-faceted retrieval to traverse the constructed knowledge base
from top to bottom and integrates multiple security-critical facts at each
hierarchical level, ensuring comprehensive and accurate retrieval. We evaluated
RESCUE on four benchmarks and compared it with five state-of-the-art secure
code generation methods on six LLMs. The results demonstrate that RESCUE
improves the SecurePass@1 metric by an average of 4.8 points, establishing a
new state-of-the-art performance for security. Furthermore, we performed
in-depth analysis and ablation studies to rigorously validate the effectiveness
of individual components in RESCUE.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为RESCUE的新型RAG框架，旨在通过改进检索增强生成技术来提升大语言模型生成安全代码的能力。RESCUE通过结合LLM辅助的聚类与摘要蒸馏方法以及程序切片技术构建混合知识库，并采用分层多面检索策略，有效整合高层安全指南和聚焦安全的代码示例，从而显著提高了生成代码的安全性。实验表明，RESCUE在多个基准测试中优于现有方法，平均提升SecurePass@1指标4.8分。</details></td></tr></tbody></table>

### 📅 2025-10-20
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.17795v1">Executable Knowledge Graphs for Replicating AI Research</a></td><td><details><summary>展开</summary>Replicating AI research is a crucial yet challenging task for large language
model (LLM) agents. Existing approaches often struggle to generate executable
code, primarily due to insufficient background knowledge and the limitations of
retrieval-augmented generation (RAG) methods, which fail to capture latent
technical details hidden in referenced papers. Furthermore, previous approaches
tend to overlook valuable implementation-level code signals and lack structured
knowledge representations that support multi-granular retrieval and reuse. To
overcome these challenges, we propose Executable Knowledge Graphs (xKG), a
modular and pluggable knowledge base that automatically integrates technical
insights, code snippets, and domain-specific knowledge extracted from
scientific literature. When integrated into three agent frameworks with two
different LLMs, xKG shows substantial performance gains (10.9% with o3-mini) on
PaperBench, demonstrating its effectiveness as a general and extensible
solution for automated AI research replication. Code will released at
https://github.com/zjunlp/xKG.</details></td><td><details><summary>展开</summary>本文提出了一种名为Executable Knowledge Graphs（xKG）的可插拔知识库，用于解决现有检索增强生成（RAG）方法在复制AI研究时难以生成可执行代码的问题。xKG通过整合科学文献中的技术细节、代码片段和领域知识，显著提升了大型语言模型代理在自动化AI研究复现中的性能表现。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.17733v1">Train for Truth, Keep the Skills: Binary Retrieval-Augmented Reward Mitigates Hallucinations</a></td><td><details><summary>展开</summary>Language models often generate factually incorrect information unsupported by
their training data, a phenomenon known as extrinsic hallucination. Existing
mitigation approaches often degrade performance on open-ended generation and
downstream tasks, limiting their practical utility. We propose an online
reinforcement learning method using a novel binary retrieval-augmented reward
(RAR) to address this tradeoff. Unlike continuous reward schemes, our approach
assigns a reward of one only when the model's output is entirely factually
correct, and zero otherwise. We evaluate our method on Qwen3 reasoning models
across diverse tasks. For open-ended generation, binary RAR achieves a 39.3%
reduction in hallucination rates, substantially outperforming both supervised
training and continuous-reward RL baselines. In short-form question answering,
the model learns calibrated abstention, strategically outputting "I don't know"
when faced with insufficient parametric knowledge. This yields 44.4% and 21.7%
fewer incorrect answers on PopQA and GPQA, respectively. Crucially, these
factuality gains come without performance degradation on instruction following,
math, or code, whereas continuous-reward RL, despite improving factuality,
induces quality regressions.</details></td><td><details><summary>展开</summary>该论文提出了一种基于强化学习的检索增强奖励（RAR）方法，通过二元奖励机制（输出完全正确时奖励为1，否则为0）减少语言模型的外源性幻觉问题。实验表明，该方法在开放生成任务中显著降低幻觉率，并能在问答任务中实现策略性“未知”回答，同时保持指令遵循、数学和代码等任务的性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.17590v1">MIRAGE: Agentic Framework for Multimodal Misinformation Detection with Web-Grounded Reasoning</a></td><td><details><summary>展开</summary>Misinformation spreads across web platforms through billions of daily
multimodal posts that combine text and images, overwhelming manual
fact-checking capacity. Supervised detection models require domain-specific
training data and fail to generalize across diverse manipulation tactics. We
present MIRAGE, an inference-time, model-pluggable agentic framework that
decomposes multimodal verification into four sequential modules: visual
veracity assessment detects AI-generated images, cross-modal consistency
analysis identifies out-of-context repurposing, retrieval-augmented factual
checking grounds claims in web evidence through iterative question generation,
and a calibrated judgment module integrates all signals. MIRAGE orchestrates
vision-language model reasoning with targeted web retrieval, outputs structured
and citation-linked rationales. On MMFakeBench validation set (1,000 samples),
MIRAGE with GPT-4o-mini achieves 81.65% F1 and 75.1% accuracy, outperforming
the strongest zero-shot baseline (GPT-4V with MMD-Agent at 74.0% F1) by 7.65
points while maintaining 34.3% false positive rate versus 97.3% for a
judge-only baseline. Test set results (5,000 samples) confirm generalization
with 81.44% F1 and 75.08% accuracy. Ablation studies show visual verification
contributes 5.18 F1 points and retrieval-augmented reasoning contributes 2.97
points. Our results demonstrate that decomposed agentic reasoning with web
retrieval can match supervised detector performance without domain-specific
training, enabling misinformation detection across modalities where labeled
data remains scarce.</details></td><td><details><summary>展开</summary>这篇文章提出了MIRAGE框架，通过多模态验证和检索增强的事实核查（retrieval-augmented factual checking）来检测网络上的错误信息，利用检索到的网络证据增强生成模型（如GPT-4o-mini）的推理能力，显著提升了检测准确性并生成结构化解释。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.17476v1">Disparities in Multilingual LLM-Based Healthcare Q&A</a></td><td><details><summary>展开</summary>Equitable access to reliable health information is vital when integrating AI
into healthcare. Yet, information quality varies across languages, raising
concerns about the reliability and consistency of multilingual Large Language
Models (LLMs). We systematically examine cross-lingual disparities in
pre-training source and factuality alignment in LLM answers for multilingual
healthcare Q&A across English, German, Turkish, Chinese (Mandarin), and
Italian. We (i) constructed Multilingual Wiki Health Care
(MultiWikiHealthCare), a multilingual dataset from Wikipedia; (ii) analyzed
cross-lingual healthcare coverage; (iii) assessed LLM response alignment with
these references; and (iv) conducted a case study on factual alignment through
the use of contextual information and Retrieval-Augmented Generation (RAG). Our
findings reveal substantial cross-lingual disparities in both Wikipedia
coverage and LLM factual alignment. Across LLMs, responses align more with
English Wikipedia, even when the prompts are non-English. Providing contextual
excerpts from non-English Wikipedia at inference time effectively shifts
factual alignment toward culturally relevant knowledge. These results highlight
practical pathways for building more equitable, multilingual AI systems for
healthcare.</details></td><td><details><summary>展开</summary>这篇论文研究了多语言大语言模型（LLMs）在医疗问答中的事实准确性及其与维基百科预训练数据的跨语言差异，并通过构建多语言数据集（MultiWikiHealthCare）和引入检索增强生成（RAG）技术，验证了上下文信息能有效提升非英语回答的文化相关性与事实对齐，从而推动更公平的多语言医疗AI系统发展。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.17354v1">Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for
enhancing large language models (LLMs) by retrieving relevant documents from an
external corpus. However, existing RAG systems primarily focus on unimodal text
documents, and often fall short in real-world scenarios where both queries and
documents may contain mixed modalities (such as text and images). In this
paper, we address the challenge of Universal Retrieval-Augmented Generation
(URAG), which involves retrieving and reasoning over mixed-modal information to
improve vision-language generation. To this end, we propose Nyx, a unified
mixed-modal to mixed-modal retriever tailored for URAG scenarios. To mitigate
the scarcity of realistic mixed-modal data, we introduce a four-stage automated
pipeline for generation and filtering, leveraging web documents to construct
NyxQA, a dataset comprising diverse mixed-modal question-answer pairs that
better reflect real-world information needs. Building on this high-quality
dataset, we adopt a two-stage training framework for Nyx: we first perform
pre-training on NyxQA along with a variety of open-source retrieval datasets,
followed by supervised fine-tuning using feedback from downstream
vision-language models (VLMs) to align retrieval outputs with generative
preferences. Experimental results demonstrate that Nyx not only performs
competitively on standard text-only RAG benchmarks, but also excels in the more
general and realistic URAG setting, significantly improving generation quality
in vision-language tasks.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为Nyx的统一混合模态检索器，用于解决通用检索增强生成（URAG）中多模态（如文本和图像）信息的检索与推理问题，并通过自动构建数据集和改进训练框架，显著提升了视觉语言任务的生成质量。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.17309v1">RubiSCoT: A Framework for AI-Supported Academic Assessment</a></td><td><details><summary>展开</summary>The evaluation of academic theses is a cornerstone of higher education,
ensuring rigor and integrity. Traditional methods, though effective, are
time-consuming and subject to evaluator variability. This paper presents
RubiSCoT, an AI-supported framework designed to enhance thesis evaluation from
proposal to final submission. Using advanced natural language processing
techniques, including large language models, retrieval-augmented generation,
and structured chain-of-thought prompting, RubiSCoT offers a consistent,
scalable solution. The framework includes preliminary assessments,
multidimensional assessments, content extraction, rubric-based scoring, and
detailed reporting. We present the design and implementation of RubiSCoT,
discussing its potential to optimize academic assessment processes through
consistent, scalable, and transparent evaluation.</details></td><td><details><summary>展开</summary>该论文介绍了RubiSCoT框架，利用大语言模型、检索增强生成（RAG）和思维链提示等技术，通过自然语言处理提升学术论文评估的效率和一致性，涵盖从提案到终稿的全流程自动化评分与报告功能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.17301v1">Comprehending Spatio-temporal Data via Cinematic Storytelling using Large Language Models</a></td><td><details><summary>展开</summary>Spatio-temporal data captures complex dynamics across both space and time,
yet traditional visualizations are complex, require domain expertise and often
fail to resonate with broader audiences. Here, we propose MapMuse, a
storytelling-based framework for interpreting spatio-temporal datasets,
transforming them into compelling, narrative-driven experiences. We utilize
large language models and employ retrieval augmented generation (RAG) and
agent-based techniques to generate comprehensive stories. Drawing on principles
common in cinematic storytelling, we emphasize clarity, emotional connection,
and audience-centric design. As a case study, we analyze a dataset of taxi
trajectories. Two perspectives are presented: a captivating story based on a
heat map that visualizes millions of taxi trip endpoints to uncover urban
mobility patterns; and a detailed narrative following a single long taxi
journey, enriched with city landmarks and temporal shifts. By portraying
locations as characters and movement as plot, we argue that data storytelling
drives insight, engagement, and action from spatio-temporal information. The
case study illustrates how MapMuse can bridge the gap between data complexity
and human understanding. The aim of this short paper is to provide a glimpse to
the potential of the cinematic storytelling technique as an effective
communication tool for spatio-temporal data, as well as to describe open
problems and opportunities for future research.</details></td><td><details><summary>展开</summary>这篇论文介绍了MapMuse，一个基于叙事的数据可视化框架，通过结合大型语言模型、检索增强生成（RAG）和智能体技术，将复杂的时空数据（如出租车轨迹）转化为引人入胜的故事，强调情感连接和观众理解，旨在解决传统可视化方法的局限性并提升数据传播效果。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.17098v1">Can Transformer Memory Be Corrupted? Investigating Cache-Side Vulnerabilities in Large Language Models</a></td><td><details><summary>展开</summary>Even when prompts and parameters are secured, transformer language models
remain vulnerable because their key-value (KV) cache during inference
constitutes an overlooked attack surface. This paper introduces Malicious Token
Injection (MTI), a modular framework that systematically perturbs cached key
vectors at selected layers and timesteps through controlled magnitude and
frequency, using additive Gaussian noise, zeroing, and orthogonal rotations. A
theoretical analysis quantifies how these perturbations propagate through
attention, linking logit deviations to the Frobenius norm of corruption and
softmax Lipschitz dynamics. Empirical results show that MTI significantly
alters next-token distributions and downstream task performance across GPT-2
and LLaMA-2/7B, as well as destabilizes retrieval-augmented and agentic
reasoning pipelines. These findings identify cache integrity as a critical yet
underexplored vulnerability in current LLM deployments, positioning cache
corruption as a reproducible and theoretically grounded threat model for future
robustness and security research.</details></td><td><details><summary>展开</summary>这篇论文提出了名为“恶意令牌注入”（MTI）的攻击框架，通过干扰Transformer语言模型推理过程中的键值缓存（KV cache），利用高斯噪声、归零和正交旋转等方法系统性破坏缓存数据。研究表明，MTI不仅能显著改变模型的next-token分布和任务性能，还会影响检索增强生成（RAG）和智能代理推理流程的稳定性，揭示了KV缓存完整性在当前LLM部署中的潜在安全漏洞。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.17064v1">A Brain Cell Type Resource Created by Large Language Models and a Multi-Agent AI System for Collaborative Community Annotation</a></td><td><details><summary>展开</summary>Single-cell RNA sequencing has transformed our ability to identify diverse
cell types and their transcriptomic signatures. However, annotating these
signatures-especially those involving poorly characterized genes-remains a
major challenge. Traditional methods, such as Gene Set Enrichment Analysis
(GSEA), depend on well-curated annotations and often perform poorly in these
contexts. Large Language Models (LLMs) offer a promising alternative but
struggle to represent complex biological knowledge within structured
ontologies. To address this, we present BRAINCELL-AID (BRAINCELL-AID:
https://biodataai.uth.edu/BRAINCELL-AID), a novel multi-agent AI system that
integrates free-text descriptions with ontology labels to enable more accurate
and robust gene set annotation. By incorporating retrieval-augmented generation
(RAG), we developed a robust agentic workflow that refines predictions using
relevant PubMed literature, reducing hallucinations and enhancing
interpretability. Using this workflow, we achieved correct annotations for 77%
of mouse gene sets among their top predictions. Applying this approach, we
annotated 5,322 brain cell clusters from the comprehensive mouse brain cell
atlas generated by the BRAIN Initiative Cell Census Network, enabling novel
insights into brain cell function by identifying region-specific gene
co-expression patterns and inferring functional roles of gene ensembles.
BRAINCELL-AID also identifies Basal Ganglia-related cell types with
neurologically meaningful descriptions. Hence, we create a valuable resource to
support community-driven cell type annotation.</details></td><td><details><summary>展开</summary>这篇文章介绍了BRAINCELL-AID，一个整合自由文本描述和本体标签的多智能体AI系统，利用检索增强生成（RAG）技术从PubMed文献中检索相关信息，以提高基因集注释的准确性和可解释性，并将其应用于小鼠大脑细胞图谱的注释。</details></td></tr></tbody></table>

### 📅 2025-10-19
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.16980v1">Towards Interpretable and Trustworthy Time Series Reasoning: A BlueSky Vision</a></td><td><details><summary>展开</summary>Time series reasoning is emerging as the next frontier in temporal analysis,
aiming to move beyond pattern recognition towards explicit, interpretable, and
trustworthy inference. This paper presents a BlueSky vision built on two
complementary directions. One builds robust foundations for time series
reasoning, centered on comprehensive temporal understanding, structured
multi-step reasoning, and faithful evaluation frameworks. The other advances
system-level reasoning, moving beyond language-only explanations by
incorporating multi-agent collaboration, multi-modal context, and
retrieval-augmented approaches. Together, these directions outline a flexible
and extensible framework for advancing time series reasoning, aiming to deliver
interpretable and trustworthy temporal intelligence across diverse domains.</details></td><td><details><summary>展开</summary>这篇论文提出了一个关于时间序列推理的蓝图，结合了稳健的基础构建和系统级推理两大方向，其中特别提到了通过多智能体协作、多模态上下文和检索增强方法（retrieval-augmented approaches）来增强推理能力。这表明该研究在一定程度上涉及了RAG技术。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.16724v1">A Comprehensive Survey on Reinforcement Learning-based Agentic Search: Foundations, Roles, Optimizations, Evaluations, and Applications</a></td><td><details><summary>展开</summary>The advent of large language models (LLMs) has transformed information access
and reasoning through open-ended natural language interaction. However, LLMs
remain limited by static knowledge, factual hallucinations, and the inability
to retrieve real-time or domain-specific information. Retrieval-Augmented
Generation (RAG) mitigates these issues by grounding model outputs in external
evidence, but traditional RAG pipelines are often single turn and heuristic,
lacking adaptive control over retrieval and reasoning. Recent advances in
agentic search address these limitations by enabling LLMs to plan, retrieve,
and reflect through multi-step interaction with search environments. Within
this paradigm, reinforcement learning (RL) offers a powerful mechanism for
adaptive and self-improving search behavior. This survey provides the first
comprehensive overview of \emph{RL-based agentic search}, organizing the
emerging field along three complementary dimensions: (i) What RL is for
(functional roles), (ii) How RL is used (optimization strategies), and (iii)
Where RL is applied (scope of optimization). We summarize representative
methods, evaluation protocols, and applications, and discuss open challenges
and future directions toward building reliable and scalable RL driven agentic
search systems. We hope this survey will inspire future research on the
integration of RL and agentic search. Our repository is available at
https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers.</details></td><td><details><summary>展开</summary>这篇论文探讨了基于强化学习的自主搜索（RL-based agentic search）如何优化传统的RAG框架，通过多步交互和自适应控制解决单一检索回合和启发式方法的不足，旨在提升检索与推理的动态性和可靠性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.16715v1">Right Answer at the Right Time - Temporal Retrieval-Augmented Generation via Graph Summarization</a></td><td><details><summary>展开</summary>Question answering in temporal knowledge graphs requires retrieval that is
both time-consistent and efficient. Existing RAG methods are largely semantic
and typically neglect explicit temporal constraints, which leads to
time-inconsistent answers and inflated token usage. We propose STAR-RAG, a
temporal GraphRAG framework that relies on two key ideas: building a
time-aligned rule graph and conducting propagation on this graph to narrow the
search space and prioritize semantically relevant, time-consistent evidence.
This design enforces temporal proximity during retrieval, reduces the candidate
set of retrieval results, and lowers token consumption without sacrificing
accuracy. Compared with existing temporal RAG approaches, STAR-RAG eliminates
the need for heavy model training and fine-tuning, thereby reducing
computational cost and significantly simplifying deployment.Extensive
experiments on real-world temporal KG datasets show that our method achieves
improved answer accuracy while consuming fewer tokens than strong GraphRAG
baselines.</details></td><td><details><summary>展开</summary>这篇论文提出了STAR-RAG，一种针对时序知识图谱的检索增强生成框架，通过构建时间对齐的规则图谱和在其上进行传播来优化检索过程，确保时间一致性和高效性，同时减少令牌消耗，无需复杂模型训练即可提升答案准确性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.16695v1">Resolution-Aware Retrieval Augmented Zero-Shot Forecasting</a></td><td><details><summary>展开</summary>Zero-shot forecasting aims to predict outcomes for previously unseen
conditions without direct historical data, posing a significant challenge for
traditional forecasting methods. We introduce a Resolution-Aware
Retrieval-Augmented Forecasting model that enhances predictive accuracy by
leveraging spatial correlations and temporal frequency characteristics. By
decomposing signals into different frequency components, our model employs
resolution-aware retrieval, where lower-frequency components rely on broader
spatial context, while higher-frequency components focus on local influences.
This allows the model to dynamically retrieve relevant data and adapt to new
locations with minimal historical context.
  Applied to microclimate forecasting, our model significantly outperforms
traditional forecasting methods, numerical weather prediction models, and
modern foundation time series models, achieving 71% lower MSE than HRRR and 34%
lower MSE than Chronos on the ERA5 dataset.
  Our results highlight the effectiveness of retrieval-augmented and
resolution-aware strategies, offering a scalable and data-efficient solution
for zero-shot forecasting in microclimate modeling and beyond.</details></td><td><details><summary>展开</summary>本文提出了一种基于分辨率感知的检索增强预测模型，通过分解信号频率并结合空间相关性进行动态数据检索，显著提升了零样本微气候预测的准确性，在ERA5数据集上优于传统方法和现代时序模型。</details></td></tr></tbody></table>

### 📅 2025-10-18
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.16643v1">Structured Interfaces for Automated Reasoning with 3D Scene Graphs</a></td><td><details><summary>展开</summary>In order to provide a robot with the ability to understand and react to a
user's natural language inputs, the natural language must be connected to the
robot's underlying representations of the world. Recently, large language
models (LLMs) and 3D scene graphs (3DSGs) have become a popular choice for
grounding natural language and representing the world. In this work, we address
the challenge of using LLMs with 3DSGs to ground natural language. Existing
methods encode the scene graph as serialized text within the LLM's context
window, but this encoding does not scale to large or rich 3DSGs. Instead, we
propose to use a form of Retrieval Augmented Generation to select a subset of
the 3DSG relevant to the task. We encode a 3DSG in a graph database and provide
a query language interface (Cypher) as a tool to the LLM with which it can
retrieve relevant data for language grounding. We evaluate our approach on
instruction following and scene question-answering tasks and compare against
baseline context window and code generation methods. Our results show that
using Cypher as an interface to 3D scene graphs scales significantly better to
large, rich graphs on both local and cloud-based models. This leads to large
performance improvements in grounded language tasks while also substantially
reducing the token count of the scene graph content. A video supplement is
available at https://www.youtube.com/watch?v=zY_YI9giZSA.</details></td><td><details><summary>展开</summary>该论文提出了一种基于RAG的方法，利用图数据库（Cypher查询语言）检索3D场景图中与任务相关的子集，替代传统的大语言模型上下文窗口编码方式，以提高自然语言在机器人任务中的理解能力，并在指令跟随和场景问答任务中验证了其扩展性和性能优势。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.16609v1">Prior Makes It Possible: From Sublinear Graph Algorithms to LLM Test-Time Methods</a></td><td><details><summary>展开</summary>Test-time augmentation, such as Retrieval-Augmented Generation (RAG) or tool
use, critically depends on an interplay between a model's parametric knowledge
and externally retrieved information. However, the theoretical underpinnings of
this relationship remain poorly understood. Specifically, it is not clear how
much pre-training knowledge is required to answer queries with a small number
of augmentation steps, which is a desirable property in practice. To address
this question, we formulate multi-step reasoning as an $s$-$t$ connectivity
problem on a knowledge graph. We represent a model's pre-training parametric
knowledge as a partial, potentially noisy subgraph. We view augmentation as
querying an oracle for true edges that augment the model's knowledge. Then, we
characterize the necessary and sufficient number of augmentation steps for the
model to generate an accurate answer given partial prior knowledge. One key
result shows a phase transition: if the prior knowledge graph over $n$ vertices
is disconnected into small components, then finding a path via augmentation is
inefficient and requires $\Omega(\sqrt{n})$ queries. On the other hand, once
the density of correct knowledge surpasses a threshold, forming a giant
component, we can find paths with an expected constant number of queries.</details></td><td><details><summary>展开</summary>这篇论文探讨了RAG等技术中模型预训练知识与外部检索信息之间的理论关系，将多步推理建模为知识图上的连通性问题，分析了先验知识密度和增强步骤数量对答案准确性的影响，并揭示了知识图连接性对查询效率的临界相变现象。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.16582v1">Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) based on knowledge graphs (KGs) enhances
large language models (LLMs) by providing structured and interpretable external
knowledge. However, existing KG-based RAG methods struggle to retrieve accurate
and diverse information from text-rich KGs for complex real-world queries.
Process Reward Models (PRMs) offer a way to align the retrieval process of
KG-based RAG with query-specific knowledge requirements, but they heavily rely
on process-level supervision signals that are expensive and hard to obtain on
KGs. To address this challenge, we propose GraphFlow, a framework that
efficiently retrieves accurate and diverse knowledge required for real-world
queries from text-rich KGs. GraphFlow employs a transition-based flow matching
objective to jointly optimize a retrieval policy and a flow estimator. The flow
estimator factorizes the reward of the retrieval outcome into the intermediate
retrieval states. Such reward factorization guides the retrieval policy to
retrieve candidates from KGs in proportion to their reward. This allows
GraphFlow to explore high-quality regions of KGs that yield diverse and
relevant results. We evaluate GraphFlow on the STaRK benchmark, which includes
real-world queries from multiple domains over text-rich KGs. GraphFlow
outperforms strong KG-RAG baselines, including GPT-4o, by 10% on average in hit
rate and recall. It also shows strong generalization to unseen KGs,
demonstrating its effectiveness and robustness.</details></td><td><details><summary>展开</summary>这篇文章提出了一种名为GraphFlow的框架，通过基于知识图谱（KG）的检索增强生成（RAG）方法，优化了从文本丰富的KG中检索准确且多样化知识的过程。GraphFlow利用转移流匹配目标联合优化检索策略和流估计器，将检索结果的奖励分解到中间状态，从而在STaRK基准测试中优于现有KG-RAG基线（包括GPT-4o），并在命中率和召回率上平均提升10%，同时展现出对未见KG的强泛化能力。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.16392v1">RGMem: Renormalization Group-based Memory Evolution for Language Agent User Profile</a></td><td><details><summary>展开</summary>Personalized and continuous interactions are the key to enhancing user
experience in today's large language model (LLM)-based conversational systems,
however, the finite context windows and static parametric memory make it
difficult to model the cross-session long-term user states and behavioral
consistency. Currently, the existing solutions to this predicament, such as
retrieval-augmented generation (RAG) and explicit memory systems, primarily
focus on fact-level storage and retrieval, lacking the capability to distill
latent preferences and deep traits from the multi-turn dialogues, which limits
the long-term and effective user modeling, directly leading to the personalized
interactions remaining shallow, and hindering the cross-session continuity. To
realize the long-term memory and behavioral consistency for Language Agents in
LLM era, we propose a self-evolving memory framework RGMem, inspired by the
ideology of classic renormalization group (RG) in physics, this framework
enables to organize the dialogue history in multiple scales: it first extracts
semantics and user insights from episodic fragments, then through hierarchical
coarse-graining and rescaling operations, progressively forms a
dynamically-evolved user profile. The core innovation of our work lies in
modeling memory evolution as a multi-scale process of information compression
and emergence, which accomplishes the high-level and accurate user profiles
from noisy and microscopic-level interactions.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为RGMem的自我进化记忆框架，旨在解决基于大语言模型的对话系统中长期用户状态和行为一致性的建模问题。虽然现有方法（如RAG和显式记忆系统）主要关注事实级别的存储和检索，但RGMem通过多尺度对话历史组织和分层粗粒度化操作，能够从多轮对话中提炼潜在偏好和深层特征，从而实现更高层次和准确的用户画像，提升个性化交互的深度和跨会话连续性。因此，RGMem对RAG技术进行了扩展和改进，属于RAG相关的研究。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.16302v1">DTKG: Dual-Track Knowledge Graph-Verified Reasoning Framework for Multi-Hop QA</a></td><td><details><summary>展开</summary>Multi-hop reasoning for question answering (QA) plays a critical role in
retrieval-augmented generation (RAG) for modern large language models (LLMs).
The accurate answer can be obtained through retrieving relational structure of
entities from knowledge graph (KG). Regarding the inherent relation-dependency
and reasoning pattern, multi-hop reasoning can be in general classified into
two categories: i) parallel fact-verification multi-hop reasoning question,
i.e., requiring simultaneous verifications of multiple independent
sub-questions; and ii) chained multi-hop reasoning questions, i.e., demanding
sequential multi-step inference with intermediate conclusions serving as
essential premises for subsequent reasoning. Currently, the multi-hop reasoning
approaches singly employ one of two techniques: LLM response-based fact
verification and KG path-based chain construction. Nevertheless, the former
excels at parallel fact-verification but underperforms on chained reasoning
tasks, while the latter demonstrates proficiency in chained multi-hop reasoning
but suffers from redundant path retrieval when handling parallel
fact-verification reasoning. These limitations deteriorate the efficiency and
accuracy for multi-hop QA tasks. To address this challenge, we propose a novel
dual-track KG verification and reasoning framework DTKG, which is inspired by
the Dual Process Theory in cognitive science. Specifically, DTKG comprises two
main stages: the Classification Stage and the Branch Processing Stage.</details></td><td><details><summary>展开</summary>这篇论文探讨了在多跳推理问答（QA）任务中如何通过知识图谱（KG）提升检索增强生成（RAG）的效果，提出了一种双轨框架DTKG，结合了LLM的事实验证和KG路径构建，以优化并行事实验证和链式多跳推理的效率与准确性。</details></td></tr></tbody></table>

### 📅 2025-10-17
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.15828v1">GENESIS: A Generative Model of Episodic-Semantic Interaction</a></td><td><details><summary>展开</summary>A central challenge in cognitive neuroscience is to explain how semantic and
episodic memory, two major forms of declarative memory, typically associated
with cortical and hippocampal processing, interact to support learning, recall,
and imagination. Despite significant advances, we still lack a unified
computational framework that jointly accounts for core empirical phenomena
across both semantic and episodic processing domains. Here, we introduce the
Generative Episodic-Semantic Integration System (GENESIS), a computational
model that formalizes memory as the interaction between two limited-capacity
generative systems: a Cortical-VAE, supporting semantic learning and
generalization, and a Hippocampal-VAE, supporting episodic encoding and
retrieval within a retrieval-augmented generation (RAG) architecture. GENESIS
reproduces hallmark behavioral findings, including generalization in semantic
memory, recognition, serial recall effects and gist-based distortions in
episodic memory, and constructive episodic simulation, while capturing their
dynamic interactions. The model elucidates how capacity constraints shape the
fidelity and memorability of experiences, how semantic processing introduces
systematic distortions in episodic recall, and how episodic replay can
recombine previous experiences. Together, these results provide a principled
account of memory as an active, constructive, and resource-bounded process.
GENESIS thus advances a unified theoretical framework that bridges semantic and
episodic memory, offering new insights into the generative foundations of human
cognition.</details></td><td><details><summary>展开</summary>这篇文章介绍了GENESIS模型，该模型通过结合语义记忆（Cortical-VAE）和情景记忆（Hippocampal-VAE）的计算框架，利用检索增强生成（RAG）架构支持记忆的编码、检索和想象，揭示了记忆的生成性、主动性和资源限制特性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.15782v1">Demo: Guide-RAG: Evidence-Driven Corpus Curation for Retrieval-Augmented Generation in Long COVID</a></td><td><details><summary>展开</summary>As AI chatbots gain adoption in clinical medicine, developing effective
frameworks for complex, emerging diseases presents significant challenges. We
developed and evaluated six Retrieval-Augmented Generation (RAG) corpus
configurations for Long COVID (LC) clinical question answering, ranging from
expert-curated sources to large-scale literature databases. Our evaluation
employed an LLM-as-a-judge framework across faithfulness, relevance, and
comprehensiveness metrics using LongCOVID-CQ, a novel dataset of
expert-generated clinical questions. Our RAG corpus configuration combining
clinical guidelines with high-quality systematic reviews consistently
outperformed both narrow single-guideline approaches and large-scale literature
databases. Our findings suggest that for emerging diseases, retrieval grounded
in curated secondary reviews provides an optimal balance between narrow
consensus documents and unfiltered primary literature, supporting clinical
decision-making while avoiding information overload and oversimplified
guidance. We propose Guide-RAG, a chatbot system and accompanying evaluation
framework that integrates both curated expert knowledge and comprehensive
literature databases to effectively answer LC clinical questions.</details></td><td><details><summary>展开</summary>该论文研究和评估了六种针对长期新冠肺炎（Long COVID）临床问答的RAG语料库配置，提出结合临床指南与高质量系统评价的配置效果最佳，并开发了名为Guide-RAG的聊天机器人系统，整合专家知识库与文献数据库以优化临床决策支持。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.15722v1">The 3rd Place Solution of CCIR CUP 2025: A Framework for Retrieval-Augmented Generation in Multi-Turn Legal Conversation</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation has made significant progress in the field of
natural language processing. By combining the advantages of information
retrieval and large language models, RAG can generate relevant and contextually
appropriate responses based on items retrieved from reliable sources. This
technology has demonstrated outstanding performance across multiple domains,
but its application in the legal field remains in its exploratory phase. In
this paper, we introduce our approach for "Legal Knowledge Retrieval and
Generation" in CCIR CUP 2025, which leverages large language models and
information retrieval systems to provide responses based on laws in response to
user questions.</details></td><td><details><summary>展开</summary>本文探讨了RAG技术在法律领域的应用，提出了一种结合大语言模型和信息检索系统的方法，用于基于法律条文生成回答用户问题的响应，并介绍了在CCIR CUP 2025中的“Legal Knowledge Retrieval and Generation”方案。目前该技术在法律领域的应用仍处于探索阶段。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.15719v1">Cost-Aware Retrieval-Augmentation Reasoning Models with Adaptive Retrieval Depth</a></td><td><details><summary>展开</summary>Reasoning models have gained significant attention due to their strong
performance, particularly when enhanced with retrieval augmentation. However,
these models often incur high computational costs, as both retrieval and
reasoning tokens contribute substantially to the overall resource usage. In
this work, we make the following contributions: (1) we propose a
retrieval-augmented reasoning model that dynamically adjusts the length of the
retrieved document list based on the query and retrieval results; (2) we
develop a cost-aware advantage function for training of efficient
retrieval-augmented reasoning models through reinforcement learning; and (3) we
explore both memory- and latency-bound implementations of the proposed
cost-aware framework for both proximal and group relative policy optimization
algorithms. We evaluate our approach on seven public question answering
datasets and demonstrate significant efficiency gains, without compromising
effectiveness. In fact, we observed that the model latency decreases by ~16-20%
across datasets, while its effectiveness increases by ~5% on average, in terms
of exact match.</details></td><td><details><summary>展开</summary>这篇论文提出了一种动态调整检索文档长度的检索增强推理模型，通过强化学习训练成本感知优势函数以提高效率，并在多个问答数据集上验证了其在降低延迟16-20%的同时提升准确率约5%的效果，未牺牲模型性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.15682v1">SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>We present SQuAI (https://squai.scads.ai/), a scalable and trustworthy
multi-agent retrieval-augmented generation (RAG) framework for scientific
question answering (QA) with large language models (LLMs). SQuAI addresses key
limitations of existing RAG systems in the scholarly domain, where complex,
open-domain questions demand accurate answers, explicit claims with citations,
and retrieval across millions of scientific documents. Built on over 2.3
million full-text papers from arXiv.org, SQuAI employs four collaborative
agents to decompose complex questions into sub-questions, retrieve targeted
evidence via hybrid sparse-dense retrieval, and adaptively filter documents to
improve contextual relevance. To ensure faithfulness and traceability, SQuAI
integrates in-line citations for each generated claim and provides supporting
sentences from the source documents. Our system improves faithfulness, answer
relevance, and contextual relevance by up to +0.088 (12%) over a strong RAG
baseline. We further release a benchmark of 1,000 scientific
question-answer-evidence triplets to support reproducibility. With transparent
reasoning, verifiable citations, and domain-wide scalability, SQuAI
demonstrates how multi-agent RAG enables more trustworthy scientific QA with
LLMs.</details></td><td><details><summary>展开</summary>本文介绍了一种名为SQuAI的可扩展且可信的多智能体检索增强生成（RAG）框架，用于基于大语言模型（LLMs）的科学问答（QA）。SQuAI针对学术领域现有RAG系统的局限性，通过分解复杂问题、混合稀疏-密集检索、自适应过滤文档等技术提高回答的准确性和相关性，并为每个生成的主张提供内联引用和来源支持。实验表明，SQuAI在忠实性、答案相关性和上下文相关性上比基线RAG系统提升了12%，同时发布了包含1000个科学问答证据三元组的基准数据集以支持可复现性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.15681v1">ProofBridge: Auto-Formalization of Natural Language Proofs in Lean via Joint Embeddings</a></td><td><details><summary>展开</summary>Translating human-written mathematical theorems and proofs from natural
language (NL) into formal languages (FLs) like Lean 4 has long been a
significant challenge for AI. Most state-of-the-art methods address this
separately, first translating theorems and then generating proofs, creating a
fundamental disconnect vis-a-vis true proof auto-formalization. This two-step
process and its limitations were evident even in AlphaProof's silver-medal
performance at the 2024 IMO, where problem statements needed manual translation
before automated proof synthesis.
  We present ProofBridge, a unified framework for automatically translating
entire NL theorems and proofs into Lean 4. At its core is a joint embedding
model that aligns NL and FL (NL-FL) theorem-proof pairs in a shared semantic
space, enabling cross-modal retrieval of semantically relevant FL examples to
guide translation. Our training ensures that NL-FL theorems (and their proofs)
are mapped close together in this space if and only if the NL-FL pairs are
semantically equivalent. ProofBridge integrates retrieval-augmented fine-tuning
with iterative proof repair, leveraging Lean's type checker and semantic
equivalence feedback to ensure both syntactic correctness and semantic
fidelity. Experiments show substantial improvements in proof auto-formalization
over strong baselines (including GPT-5, Gemini-2.5, Kimina-Prover,
DeepSeek-Prover), with our retrieval-augmented approach yielding significant
gains in semantic correctness (SC, via proving bi-directional equivalence) and
type correctness (TC, via type-checking theorem+proof) across pass@k metrics on
miniF2F-Test-PF, a dataset we curated. In particular, ProofBridge improves
cross-modal retrieval quality by up to 3.28x Recall@1 over all-MiniLM-L6-v2,
and achieves +31.14% SC and +1.64% TC (pass@32) compared to the baseline
Kimina-Prover-RL-1.7B.</details></td><td><details><summary>展开</summary>这篇论文介绍了ProofBridge，一个将自然语言数学定理和证明自动翻译为Lean 4的统一框架。它通过联合嵌入模型对齐自然语言和形式语言的语义空间，利用检索增强的微调和迭代证明修复来提高翻译的准确性和语义保真度，实验显示在多个指标上显著优于现有基线方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.15620v1">GRATING: Low-Latency and Memory-Efficient Semantic Selection on Device</a></td><td><details><summary>展开</summary>Semantic top-K selection with cross-encoder rerankers underpins of on-device
AI services, such as retrieval-augmented generation, agent memory, and
personalized recommendation. However, its latency and memory demands dominate
end-to-end budgets on edge hardware. Revisiting the objective of top-K
selection, we reveal that only relative rankings matter, not exact
per-candidate scores. We further observe sequence-level sparsity: relative
rankings stabilize early in intermediate layers, allowing pruning opportunities
prior to completing full inference.
  Building on this insight, we propose monolithic forwarding and develop a
training-free inference system, GRATING. By maintaining a global view of all
candidates, it reduces latency through progressive cluster pruning. It also
bounds peak memory usage by strategically overlapping I/O with computation via
dual-layer sliding window and chunked execution. We evaluate GRATING against
state-of-the-art baselines on rerankers from 0.6B to 8B parameters across Apple
M2 and RTX 5070. GRATING consistently reduces latency by up to 89.0% and peak
memory by up to 94.9% in microbenchmarks, without any loss in precision. Across
three real-world on-device AI applications, GRATING lowers latency by
11.6%-51.0% and peak memory by 18.6%-77.8%, demonstrating substantial
improvements in efficiency and deployability.</details></td><td><details><summary>展开</summary>本文提出了一种名为GRATING的高效推理系统，专注于优化交叉编码器重排序器（cross-encoder rerankers）在语义Top-K选择中的计算效率，特别是在边缘设备上的延迟和内存占用问题。该方法通过利用序列级稀疏性和相对排名的早期稳定性，实现了无需训练即可显著降低延迟（高达89.0%）和峰值内存（高达94.9%）的技术，并直接应用于检索增强生成（RAG）等设备端AI服务场景。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.15552v1">Think Parallax: Solving Multi-Hop Problems via Multi-View Knowledge-Graph-Based Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Large language models (LLMs) excel at language understanding but often
hallucinate and struggle with multi-hop reasoning. Knowledge-graph-based
retrieval-augmented generation (KG-RAG) offers grounding, yet most methods rely
on flat embeddings and noisy path exploration. We propose ParallaxRAG, a
framework that symmetrically decouples queries and graph triples into
multi-view spaces, enabling a robust retrieval architecture that explicitly
enforces head diversity while constraining weakly related paths. Central to our
approach is the observation that different attention heads specialize in
semantic relations at distinct reasoning stages, contributing to different hops
of the reasoning chain. This specialization allows ParallaxRAG to construct
cleaner subgraphs and guide LLMs through grounded, step-wise reasoning.
Experiments on WebQSP and CWQ, under our unified, reproducible setup (BGE-M3 +
Llama3.1-8B), demonstrate competitive retrieval and QA performance, alongside
reduced hallucination and good generalization. Our results highlight multi-view
head specialization as a principled direction for knowledge-grounded multi-hop
reasoning. Our implementation will be released as soon as the paper is
accepted.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为ParallaxRAG的框架，它通过多视角空间对称解耦查询和知识图谱三元组，增强了基于知识图谱的检索增强生成（KG-RAG）。该方法利用不同注意力头在推理链不同阶段的语义关系 specialization，构建更清晰的子图并指导大语言模型（LLM）进行逐步推理，从而减少幻觉并提升多跳推理性能。实验在WebQSP和CWQ数据集上验证了其在检索和问答任务中的竞争力。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.15428v1">Fault Cause Identification across Manufacturing Lines through Ontology-Guided and Process-Aware FMEA Graph Learning with LLMs</a></td><td><details><summary>展开</summary>Fault cause identification in automated manufacturing lines is challenging
due to the system's complexity, frequent reconfigurations, and the limited
reusability of existing Failure Mode and Effects Analysis (FMEA) knowledge.
Although FMEA worksheets contain valuable expert insights, their reuse across
heterogeneous lines is hindered by natural language variability, inconsistent
terminology, and process differences. To address these limitations, this study
proposes a process-aware framework that enhances FMEA reusability by combining
manufacturing-domain conceptualization with graph neural network (GNN)
reasoning. First, FMEA worksheets from multiple manufacturing lines are
transformed into a unified knowledge graph through ontology-guided large
language model (LLM) extraction, capturing domain concepts such as actions,
states, components, and parameters. Second, a Relational Graph Convolutional
Network (RGCN) with the process-aware scoring function learns embeddings that
respect both semantic relationships and sequential process flows. Finally, link
prediction is employed to infer and rank candidate fault causes consistent with
the target line's process flow.
  A case study on automotive pressure sensor assembly lines demonstrates that
the proposed method outperforms a state-of-the-art retrieval-augmented
generation (RAG) baseline (F1@20 = 0.267) and an RGCN approach (0.400),
achieving the best performance (0.523) in fault cause identification. Ablation
studies confirm the contributions of both LLM-driven domain conceptualization
and process-aware learning. These results indicate that the proposed framework
significantly improves the transferability of FMEA knowledge across
heterogeneous lines, thereby supporting operators in diagnosing failures more
reliably and paving the way for future domain-adaptive LLM applications in
smart manufacturing.</details></td><td><details><summary>展开</summary>这篇文章提出了一种结合制造业领域概念化和图神经网络（GNN）推理的流程感知框架，用于提升故障模式与效应分析（FMEA）知识的可重用性，并通过知识图谱构建和链接预测实现故障原因识别。研究显示该方法优于检索增强生成（RAG）基线和其他模型，显著提升了FMEA知识在异构生产线间的迁移能力。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.15418v1">Fine-Tuning MedGemma for Clinical Captioning to Enhance Multimodal RAG over Malaysia CPGs</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation systems are essential for providing fact-based
guidance from Malaysian Clinical Practice Guidelines. However, their
effectiveness with image-based queries is limited, as general Vision-Language
Model captions often lack clinical specificity and factual grounding. This
study proposes and validates a framework to specialize the MedGemma model for
generating high-fidelity captions that serve as superior queries. To overcome
data scarcity, we employ a knowledge distillation pipeline to create a
synthetic dataset across dermatology, fundus, and chest radiography domains,
and fine-tune MedGemma using the parameter-efficient QLoRA method. Performance
was rigorously assessed through a dual framework measuring both classification
accuracy and, via a novel application of the RAGAS framework, caption
faithfulness, relevancy, and correctness. The fine-tuned model demonstrated
substantial improvements in classification performance, while RAGAS evaluation
confirmed significant gains in caption faithfulness and correctness, validating
the models ability to produce reliable, factually grounded descriptions. This
work establishes a robust pipeline for specializing medical VLMs and validates
the resulting model as a high-quality query generator, laying the groundwork
for enhancing multimodal RAG systems in evidence-based clinical decision
support.</details></td><td><details><summary>展开</summary>这篇论文提出并验证了一个专为医学图像生成高保真字幕的框架，旨在提升基于图像查询的多模态RAG系统性能。通过知识蒸馏创建合成数据集并微调MedGemma模型，研究显著改进了字幕的准确性和临床相关性，为循证临床决策支持中的RAG系统增强奠定了基础。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.15339v1">AutoGraph-R1: End-to-End Reinforcement Learning for Knowledge Graph Construction</a></td><td><details><summary>展开</summary>Building effective knowledge graphs (KGs) for Retrieval-Augmented Generation
(RAG) is pivotal for advancing question answering (QA) systems. However, its
effectiveness is hindered by a fundamental disconnect: the knowledge graph (KG)
construction process is decoupled from its downstream application, yielding
suboptimal graph structures. To bridge this gap, we introduce AutoGraph-R1, the
first framework to directly optimize KG construction for task performance using
Reinforcement Learning (RL). AutoGraph-R1 trains an LLM constructor by framing
graph generation as a policy learning problem, where the reward is derived from
the graph's functional utility in a RAG pipeline. We design two novel,
task-aware reward functions, one for graphs as knowledge carriers and another
as knowledge indices. Across multiple QA benchmarks, AutoGraph-R1 consistently
enables graph RAG methods to achieve significant performance gains over using
task-agnostic baseline graphs. Our work shows it is possible to close the loop
between construction and application, shifting the paradigm from building
intrinsically ``good'' graphs to building demonstrably ``useful'' ones.</details></td><td><details><summary>展开</summary>该论文提出了AutoGraph-R1框架，通过强化学习优化知识图谱（KG）构建过程，使其在RAG流程中更有效地支持问答系统，设计了两种任务感知的奖励函数以提升图谱作为知识载体和索引的性能，并在多个问答基准测试中验证了其优越性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.15261v1">AUGUSTUS: An LLM-Driven Multimodal Agent System with Contextualized User Memory</a></td><td><details><summary>展开</summary>Riding on the success of LLMs with retrieval-augmented generation (RAG),
there has been a growing interest in augmenting agent systems with external
memory databases. However, the existing systems focus on storing text
information in their memory, ignoring the importance of multimodal signals.
Motivated by the multimodal nature of human memory, we present AUGUSTUS, a
multimodal agent system aligned with the ideas of human memory in cognitive
science. Technically, our system consists of 4 stages connected in a loop: (i)
encode: understanding the inputs; (ii) store in memory: saving important
information; (iii) retrieve: searching for relevant context from memory; and
(iv) act: perform the task. Unlike existing systems that use vector databases,
we propose conceptualizing information into semantic tags and associating the
tags with their context to store them in a graph-structured multimodal
contextual memory for efficient concept-driven retrieval. Our system
outperforms the traditional multimodal RAG approach while being 3.5 times
faster for ImageNet classification and outperforming MemGPT on the MSC
benchmark.</details></td><td><details><summary>展开</summary>这篇论文介绍了AUGUSTUS，一种多模态代理系统，受到人类记忆启发，提出了一个包含编码、存储、检索和执行四个阶段的循环框架。与传统的基于向量数据库的系统不同，它采用图结构的多模态上下文记忆进行概念驱动检索，在ImageNet分类和MSC基准上表现优于传统多模态RAG方法和MemGPT。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.15253v1">Scaling Beyond Context: A Survey of Multimodal Retrieval-Augmented Generation for Document Understanding</a></td><td><details><summary>展开</summary>Document understanding is critical for applications from financial analysis
to scientific discovery. Current approaches, whether OCR-based pipelines
feeding Large Language Models (LLMs) or native Multimodal LLMs (MLLMs), face
key limitations: the former loses structural detail, while the latter struggles
with context modeling. Retrieval-Augmented Generation (RAG) helps ground models
in external data, but documents' multimodal nature, i.e., combining text,
tables, charts, and layout, demands a more advanced paradigm: Multimodal RAG.
This approach enables holistic retrieval and reasoning across all modalities,
unlocking comprehensive document intelligence. Recognizing its importance, this
paper presents a systematic survey of Multimodal RAG for document
understanding. We propose a taxonomy based on domain, retrieval modality, and
granularity, and review advances involving graph structures and agentic
frameworks. We also summarize key datasets, benchmarks, and applications, and
highlight open challenges in efficiency, fine-grained representation, and
robustness, providing a roadmap for future progress in document AI.</details></td><td><details><summary>展开</summary>这篇论文是关于多模态RAG（检索增强生成）在文档理解中的应用，系统性地综述了该领域的进展、提出分类法，总结了关键数据集和应用，并指出未来研究方向如效率和细粒度表示等挑战。</details></td></tr></tbody></table>

### 📅 2025-10-16
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.14944v1">MetaBench: A Multi-task Benchmark for Assessing LLMs in Metabolomics</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) have demonstrated remarkable capabilities on
general text; however, their proficiency in specialized scientific domains that
require deep, interconnected knowledge remains largely uncharacterized.
Metabolomics presents unique challenges with its complex biochemical pathways,
heterogeneous identifier systems, and fragmented databases. To systematically
evaluate LLM capabilities in this domain, we introduce MetaBench, the first
benchmark for metabolomics assessment. Curated from authoritative public
resources, MetaBench evaluates five capabilities essential for metabolomics
research: knowledge, understanding, grounding, reasoning, and research. Our
evaluation of 25 open- and closed-source LLMs reveals distinct performance
patterns across metabolomics tasks: while models perform well on text
generation tasks, cross-database identifier grounding remains challenging even
with retrieval augmentation. Model performance also decreases on long-tail
metabolites with sparse annotations. With MetaBench, we provide essential
infrastructure for developing and evaluating metabolomics AI systems, enabling
systematic progress toward reliable computational tools for metabolomics
research.</details></td><td><details><summary>展开</summary>这篇论文提出了MetaBench，首个用于评估大型语言模型（LLMs）在代谢组学领域能力的基准测试。研究发现，尽管模型在文本生成任务上表现良好，但即使使用了检索增强技术（RAG），跨数据库标识符的匹配仍具挑战性，尤其对注释稀疏的长尾代谢物性能下降，强调了RAG在专业科学领域的局限性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.14915v1">Harmonizing Diverse Models: A Layer-wise Merging Strategy for Consistent Generation</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) systems leverage Large Language Models
(LLMs) to generate accurate and reliable responses that are grounded in
retrieved context. However, LLMs often generate inconsistent outputs for
semantically equivalent inputs, a problem compounded by the scarcity of
consistency-focused training data and the limitations of current fine-tuning
techniques in enhancing output consistency. We propose a new approach combining
systematic synthetic data generation, triplet loss for better embeddings, and a
novel layer-wise model merging approach. Using consistency-aware weights
derived from intermediate layer activations, our method effectively integrates
knowledge from specialized models. Experimental results how that our merged
model significantly enhances output consistency, achieving a ~47.5\%
improvement in response similarity over the baseline, thus offering a practical
solution for increasing the reliability of an industrial RAG system.</details></td><td><details><summary>展开</summary>本文针对RAG系统中大语言模型（LLM）对语义相同输入产生不一致输出的问题，提出了一种结合合成数据生成、三元组损失改进嵌入及分层模型融合的新方法，通过一致性感知权重整合专业模型知识，实验显示合并后的模型使响应相似性提升约47.5%，显著增强了工业级RAG系统的可靠性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.14900v1">Mapping Smarter, Not Harder: A Test-Time Reinforcement Learning Agent That Improves Without Labels or Model Updates</a></td><td><details><summary>展开</summary>The Enterprise Intelligence Platform must integrate logs from numerous
third-party vendors in order to perform various downstream tasks. However,
vendor documentation is often unavailable at test time. It is either misplaced,
mismatched, poorly formatted, or incomplete, which makes schema mapping
challenging. We introduce a reinforcement learning agent that can self-improve
without labeled examples or model weight updates. During inference, the agent:
1) Identifies ambiguous field-mapping attempts. 2) Generates targeted
web-search queries to gather external evidence. 3) Applies a confidence-based
reward to iteratively refine its mappings. To demonstrate this concept, we
converted Microsoft Defender for Endpoint logs into a common schema. Our method
increased mapping accuracy from 56.4\%(LLM-only) to 72.73\%(RAG) to 93.94\%
over 100 iterations using GPT-4o. At the same time, it reduced the number of
low-confidence mappings requiring expert review by 85\%. This new approach
provides an evidence-driven, transparent method for solving future industry
problems, paving the way for more robust, accountable, scalable, efficient,
flexible, adaptable, and collaborative solutions.</details></td><td><details><summary>展开</summary>该论文提出了一种基于强化学习的智能代理，通过实时检索外部证据（如网络搜索）解决企业日志模式映射中的模糊性问题，并迭代优化映射结果。实验表明，结合检索增强生成（RAG）方法显著提升了GPT-4o的映射准确率（从56.4%提升至93.94%），同时减少了85%需专家审核的低置信度映射，凸显了RAG在提升模型证据驱动决策和透明度方面的价值。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.14629v1">MR.Rec: Synergizing Memory and Reasoning for Personalized Recommendation Assistant with LLMs</a></td><td><details><summary>展开</summary>The application of Large Language Models (LLMs) in recommender systems faces
key challenges in delivering deep personalization and intelligent reasoning,
especially for interactive scenarios. Current methods are often constrained by
limited context windows and single-turn reasoning, hindering their ability to
capture dynamic user preferences and proactively reason over recommendation
contexts. To address these limitations, we propose MR.Rec, a novel framework
that synergizes memory and reasoning for LLM-based recommendations. To achieve
personalization, we develop a comprehensive Retrieval-Augmented Generation
(RAG) system that efficiently indexes and retrieves relevant external memory to
enhance LLM personalization capabilities. Furthermore, to enable the synergy
between memory and reasoning, our RAG system goes beyond conventional
query-based retrieval by integrating reasoning enhanced memory retrieval.
Finally, we design a reinforcement learning framework that trains the LLM to
autonomously learn effective strategies for both memory utilization and
reasoning refinement. By combining dynamic memory retrieval with adaptive
reasoning, this approach ensures more accurate, context-aware, and highly
personalized recommendations. Extensive experiments demonstrate that MR.Rec
significantly outperforms state-of-the-art baselines across multiple metrics,
validating its efficacy in delivering intelligent and personalized
recommendations. We will release code and data upon paper notification.</details></td><td><details><summary>展开</summary>该论文提出了MR.Rec框架，通过结合检索增强生成（RAG）系统和强化学习，增强大语言模型（LLMs）在推荐系统中的记忆检索与推理能力，以实现更精准、个性化的推荐。RAG用于高效索引和检索外部记忆数据，同时引入推理增强的检索机制，最终通过实验验证其优于现有基线方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.14605v1">Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering</a></td><td><details><summary>展开</summary>Knowledge-based visual question answering (KB-VQA) requires visual language
models (VLMs) to integrate visual understanding with external knowledge
retrieval. Although retrieval-augmented generation (RAG) achieves significant
advances in this task by combining knowledge-base querying, it still struggles
with the quality of multimodal queries and the relevance of retrieved results.
To overcome these challenges, we propose a novel three-stage method, termed
Wiki-PRF, including Processing, Retrieval and Filtering stages. The processing
stage dynamically invokes visual tools to extract precise multimodal
information for retrieval. The retrieval stage integrates visual and text
features to achieve multimodal knowledge retrieval. The filtering stage
performs relevance filtering and concentration on retrieval results. To this
end, we introduce a visual language model trained with answer accuracy and
format consistency as reward signals via a reinforcement learning manner. This
enhances the model's reasoning, tool invocation for accurate queries, and
filtering of irrelevant content. Experiments on benchmark datasets (E-VQA and
InfoSeek) show significant improvements~(36.0 and 42.8) in answer quality,
achieving state-of-the-art performance. Code is available at
https://github.com/cqu-student/Wiki-PRF</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为Wiki-PRF的三阶段方法（处理、检索、过滤），通过动态调用视觉工具提取多模态信息，增强知识检索与过滤能力，结合强化学习训练视觉语言模型，显著提升了基于知识的视觉问答（KB-VQA）任务中的答案质量，在E-VQA和InfoSeek数据集上达到最优性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.14592v1">Multimodal RAG for Unstructured Data:Leveraging Modality-Aware Knowledge Graphs with Hybrid Retrieval</a></td><td><details><summary>展开</summary>Current Retrieval-Augmented Generation (RAG) systems primarily operate on
unimodal textual data, limiting their effectiveness on unstructured multimodal
documents. Such documents often combine text, images, tables, equations, and
graphs, each contributing unique information. In this work, we present a
Modality-Aware Hybrid retrieval Architecture (MAHA), designed specifically for
multimodal question answering with reasoning through a modality-aware knowledge
graph. MAHA integrates dense vector retrieval with structured graph traversal,
where the knowledge graph encodes cross-modal semantics and relationships. This
design enables both semantically rich and context-aware retrieval across
diverse modalities. Evaluations on multiple benchmark datasets demonstrate that
MAHA substantially outperforms baseline methods, achieving a ROUGE-L score of
0.486, providing complete modality coverage. These results highlight MAHA's
ability to combine embeddings with explicit document structure, enabling
effective multimodal retrieval. Our work establishes a scalable and
interpretable retrieval framework that advances RAG systems by enabling
modality-aware reasoning over unstructured multimodal data.</details></td><td><details><summary>展开</summary>本文提出了一种名为MAHA的模态感知混合检索架构，旨在解决当前RAG系统在处理多模态文档（如文本、图像、表格等）时的局限性。通过结合密集向量检索和结构化知识图谱遍历，MAHA能够实现跨模态的语义丰富和上下文感知检索，显著提升了多模态问答任务中的性能，并在多个基准测试中优于基线方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.14400v1">MedTrust-RAG: Evidence Verification and Trust Alignment for Biomedical Question Answering</a></td><td><details><summary>展开</summary>Biomedical question answering (QA) requires accurate interpretation of
complex medical knowledge. Large language models (LLMs) have shown promising
capabilities in this domain, with retrieval-augmented generation (RAG) systems
enhancing performance by incorporating external medical literature. However,
RAG-based approaches in biomedical QA suffer from hallucinations due to
post-retrieval noise and insufficient verification of retrieved evidence,
undermining response reliability. We propose MedTrust-Guided Iterative RAG, a
framework designed to enhance factual consistency and mitigate hallucinations
in medical QA. Our method introduces three key innovations. First, it enforces
citation-aware reasoning by requiring all generated content to be explicitly
grounded in retrieved medical documents, with structured Negative Knowledge
Assertions used when evidence is insufficient. Second, it employs an iterative
retrieval-verification process, where a verification agent assesses evidence
adequacy and refines queries through Medical Gap Analysis until reliable
information is obtained. Third, it integrates the MedTrust-Align Module (MTAM)
that combines verified positive examples with hallucination-aware negative
samples, leveraging Direct Preference Optimization to reinforce
citation-grounded reasoning while penalizing hallucination-prone response
patterns. Experiments on MedMCQA, MedQA, and MMLU-Med demonstrate that our
approach consistently outperforms competitive baselines across multiple model
architectures, achieving the best average accuracy with gains of 2.7% for
LLaMA3.1-8B-Instruct and 2.4% for Qwen3-8B.</details></td><td><details><summary>展开</summary>该论文提出了一种名为MedTrust-Guided Iterative RAG的框架，旨在通过引用感知推理、迭代检索验证和MedTrust-Align模块来减少医学问答中的幻觉问题，提高事实一致性，并在多个数据集上验证了其优于现有基线方法的性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.14377v1">PluriHop: Exhaustive, Recall-Sensitive QA over Distractor-Rich Corpora</a></td><td><details><summary>展开</summary>Recent advances in large language models (LLMs) and retrieval-augmented
generation (RAG) have enabled progress on question answering (QA) when relevant
evidence is in one (single-hop) or multiple (multi-hop) passages. Yet many
realistic questions about recurring report data - medical records, compliance
filings, maintenance logs - require aggregation across all documents, with no
clear stopping point for retrieval and high sensitivity to even one missed
passage. We term these pluri-hop questions and formalize them by three
criteria: recall sensitivity, exhaustiveness, and exactness. To study this
setting, we introduce PluriHopWIND, a diagnostic multilingual dataset of 48
pluri-hop questions built from 191 real-world wind industry reports in German
and English. We show that PluriHopWIND is 8-40% more repetitive than other
common datasets and thus has higher density of distractor documents, better
reflecting practical challenges of recurring report corpora. We test a
traditional RAG pipeline as well as graph-based and multimodal variants, and
find that none of the tested approaches exceed 40% in statement-wise F1 score.
Motivated by this, we propose PluriHopRAG, a RAG architecture that follows a
"check all documents individually, filter cheaply" approach: it (i) decomposes
queries into document-level subquestions and (ii) uses a cross-encoder filter
to discard irrelevant documents before costly LLM reasoning. We find that
PluriHopRAG achieves relative F1 score improvements of 18-52% depending on base
LLM. Despite its modest size, PluriHopWIND exposes the limitations of current
QA systems on repetitive, distractor-rich corpora. PluriHopRAG's performance
highlights the value of exhaustive retrieval and early filtering as a powerful
alternative to top-k methods.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为"pluri-hop"的新型问答任务，其特点是需要在大量重复性文档中进行全面检索和精确聚合。作者构建了一个多语言诊断数据集PluriHopWIND，并测试了多种RAG方法，发现现有方法表现不佳。为此，他们提出了PluriHopRAG架构，通过查询分解和交叉编码器过滤策略显著提升了性能，论证了全面检索和早期过滤在重复性文档问答中的优势。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.14337v1">Stop-RAG: Value-Based Retrieval Control for Iterative RAG</a></td><td><details><summary>展开</summary>Iterative retrieval-augmented generation (RAG) enables large language models
to answer complex multi-hop questions, but each additional loop increases
latency, costs, and the risk of introducing distracting evidence, motivating
the need for an efficient stopping strategy. Existing methods either use a
predetermined number of iterations or rely on confidence proxies that poorly
reflect whether more retrieval will actually help. We cast iterative RAG as a
finite-horizon Markov decision process and introduce Stop-RAG, a value-based
controller that adaptively decides when to stop retrieving. Trained with
full-width forward-view Q($\lambda$) targets from complete trajectories,
Stop-RAG learns effective stopping policies while remaining compatible with
black-box APIs and existing pipelines. On multi-hop question-answering
benchmarks, Stop-RAG consistently outperforms both fixed-iteration baselines
and prompting-based stopping with LLMs. These results highlight adaptive
stopping as a key missing component in current agentic systems, and demonstrate
that value-based control can improve the accuracy of RAG systems.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为Stop-RAG的自适应停止策略，用于优化迭代式检索增强生成（RAG）的效率。通过将迭代RAG建模为有限范围马尔可夫决策过程，Stop-RAG基于价值控制动态决定何时停止检索，从而减少延迟、成本和无关证据的干扰。实验表明，该方法在多跳问答任务中优于固定迭代次数和基于提示的停止策略，验证了自适应控制在提升RAG系统准确性中的重要性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.14271v1">Less is More: Denoising Knowledge Graphs For Retrieval Augmented Generation</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) systems enable large language models
(LLMs) instant access to relevant information for the generative process,
demonstrating their superior performance in addressing common LLM challenges
such as hallucination, factual inaccuracy, and the knowledge cutoff.
Graph-based RAG further extends this paradigm by incorporating knowledge graphs
(KGs) to leverage rich, structured connections for more precise and inferential
responses. A critical challenge, however, is that most Graph-based RAG systems
rely on LLMs for automated KG construction, often yielding noisy KGs with
redundant entities and unreliable relationships. This noise degrades retrieval
and generation performance while also increasing computational cost. Crucially,
current research does not comprehensively address the denoising problem for
LLM-generated KGs. In this paper, we introduce DEnoised knowledge Graphs for
Retrieval Augmented Generation (DEG-RAG), a framework that addresses these
challenges through: (1) entity resolution, which eliminates redundant entities,
and (2) triple reflection, which removes erroneous relations. Together, these
techniques yield more compact, higher-quality KGs that significantly outperform
their unprocessed counterparts. Beyond the methods, we conduct a systematic
evaluation of entity resolution for LLM-generated KGs, examining different
blocking strategies, embedding choices, similarity metrics, and entity merging
techniques. To the best of our knowledge, this is the first comprehensive
exploration of entity resolution in LLM-generated KGs. Our experiments
demonstrate that this straightforward approach not only drastically reduces
graph size but also consistently improves question answering performance across
diverse popular Graph-based RAG variants.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为DEG-RAG的框架，通过实体解析和三元组反射技术解决基于知识图谱的RAG系统中因大语言模型自动构建知识图谱而产生的噪声问题，从而提高知识图谱的质量和检索生成性能，并在实验中验证了其有效性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.14252v1">MoM: Mixtures of Scenario-Aware Document Memories for Retrieval-Augmented Generation Systems</a></td><td><details><summary>展开</summary>The traditional RAG paradigm, which typically engages in the comprehension of
relevant text chunks in response to received queries, inherently restricts both
the depth of knowledge internalization and reasoning capabilities. To address
this limitation, our research transforms the text processing in RAG from
passive chunking to proactive understanding, defining this process as document
memory extraction with the objective of simulating human cognitive processes
during reading. Building upon this, we propose the Mixtures of scenario-aware
document Memories (MoM) framework, engineered to efficiently handle documents
from multiple domains and train small language models (SLMs) to acquire the
ability to proactively explore and construct document memories. The MoM
initially instructs large language models (LLMs) to simulate domain experts in
generating document logical outlines, thereby directing structured chunking and
core content extraction. It employs a multi-path sampling and multi-perspective
evaluation mechanism, specifically designing comprehensive metrics that
represent chunk clarity and extraction completeness to select the optimal
document memories. Additionally, to infuse deeper human-like reading abilities
during the training of SLMs, we incorporate a reverse reasoning strategy, which
deduces refined expert thinking paths from high-quality outcomes. Finally,
leveraging diverse forms of content generated by MoM, we develop a three-layer
document memory retrieval mechanism, which is grounded in our theoretical proof
from the perspective of probabilistic modeling. Extensive experimental results
across three distinct domains demonstrate that the MoM framework not only
resolves text chunking challenges in existing RAG systems, providing LLMs with
semantically complete document memories, but also paves the way for SLMs to
achieve human-centric intelligent text processing.</details></td><td><details><summary>展开</summary>该论文提出了一种改进传统RAG框架的方法——MoM（Mixtures of scenario-aware document Memories），通过主动理解文档（模拟人类认知）替代被动分块处理，利用LLM生成逻辑大纲指导结构化分块和核心内容提取，并结合多路径采样、多视角评估及反向推理策略，提升小型语言模型（SLMs）的主动探索与记忆构建能力。实验证明MoM能解决现有RAG的文本分块问题，提供语义完整的文档记忆，并推动SLMs实现更人性化的文本处理。</details></td></tr></tbody></table>

### 📅 2025-10-15
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.13799v1">BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning</a></td><td><details><summary>展开</summary>As retrieval-augmented generation (RAG) tackles complex tasks, increasingly
expanded contexts offer richer information, but at the cost of higher latency
and increased cognitive load on the model. To mitigate this bottleneck,
especially for intricate multi-hop questions, we introduce BRIEF-Pro. It is a
universal, lightweight compressor that distills relevant evidence for a given
query from retrieved documents into a concise summary for seamless integration
into in-context RAG. Using seed data consisting of relatively short contexts
(fewer than 1k words), BRIEF-Pro is trained to perform abstractive compression
of extended contexts exceeding 10k words across a wide range of scenarios.
Furthermore, BRIEF-Pro offers flexible user control over summary length by
allowing users to specify the desired number of sentences. Experiments on four
open-domain multi-hop question-answering datasets show that BRIEF-Pro generates
more concise and relevant summaries, enhancing performance across small, large,
and proprietary language models. With the 70B reader model, 32x compression by
BRIEF-Pro improves QA performance by 4.67% on average over LongLLMLingua's 9x,
while requiring only 23% of its computational overhead.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为BRIEF-Pro的轻量级通用压缩器，旨在解决RAG系统中因检索上下文过长导致的延迟和模型认知负荷问题。BRIEF-Pro通过抽象压缩技术从检索文档中提取关键信息生成简洁摘要，支持用户自定义摘要长度，并在多跳问答任务中显著提升模型性能（如70B参数模型上压缩32倍时QA性能平均提升4.67%），同时降低计算开销。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.13750v1">Confidence-Based Response Abstinence: Improving LLM Trustworthiness via Activation-Based Uncertainty Estimation</a></td><td><details><summary>展开</summary>We propose a method for confidence estimation in retrieval-augmented
generation (RAG) systems that aligns closely with the correctness of large
language model (LLM) outputs. Confidence estimation is especially critical in
high-stakes domains such as finance and healthcare, where the cost of an
incorrect answer outweighs that of not answering the question. Our approach
extends prior uncertainty quantification methods by leveraging raw feed-forward
network (FFN) activations as auto-regressive signals, avoiding the information
loss inherent in token logits and probabilities after projection and softmax
normalization. We model confidence prediction as a sequence classification
task, and regularize training with a Huber loss term to improve robustness
against noisy supervision. Applied in a real-world financial industry
customer-support setting with complex knowledge bases, our method outperforms
strong baselines and maintains high accuracy under strict latency constraints.
Experiments on Llama 3.1 8B model show that using activations from only the
16th layer preserves accuracy while reducing response latency. Our results
demonstrate that activation-based confidence modeling offers a scalable,
architecture-aware path toward trustworthy RAG deployment.</details></td><td><details><summary>展开</summary>该论文提出了一种用于检索增强生成（RAG）系统的置信度估计方法，通过利用前馈网络（FFN）的原始激活信号来更准确地预测模型输出的正确性，适用于金融等高风险领域，并在实际应用中展示了优越性能和低延迟。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.13590v1">RAG Meets Temporal Graphs: Time-Sensitive Modeling and Retrieval for Evolving Knowledge</a></td><td><details><summary>展开</summary>Knowledge is inherently time-sensitive and continuously evolves over time.
Although current Retrieval-Augmented Generation (RAG) systems enrich LLMs with
external knowledge, they largely ignore this temporal nature. This raises two
challenges for RAG. First, current RAG methods lack effective time-aware
representations. Same facts of different time are difficult to distinguish with
vector embeddings or conventional knowledge graphs. Second, most RAG
evaluations assume a static corpus, leaving a blind spot regarding update costs
and retrieval stability as knowledge evolves. To make RAG time-aware, we
propose Temporal GraphRAG (TG-RAG), which models external corpora as a bi-level
temporal graph consisting of a temporal knowledge graph with timestamped
relations and a hierarchical time graph. Multi-granularity temporal summaries
are generated for each time node to capture both key events and broader trends
at that time. The design supports incremental updates by extracting new
temporal facts from the incoming corpus and merging them into the existing
graph. The temporal graph explicitly represents identical facts at different
times as distinct edges to avoid ambiguity, and the time hierarchy graph allows
only generating reports for new leaf time nodes and their ancestors, ensuring
effective and efficient updates. During inference, TG-RAG dynamically retrieves
a subgraph within the temporal and semantic scope of the query, enabling
precise evidence gathering. Moreover, we introduce ECT-QA, a time-sensitive
question-answering dataset featuring both specific and abstract queries, along
with a comprehensive evaluation protocol designed to assess incremental update
capabilities of RAG systems. Extensive experiments show that TG-RAG
significantly outperforms existing baselines, demonstrating the effectiveness
of our method in handling temporal knowledge and incremental updates.</details></td><td><details><summary>展开</summary>这篇论文提出了Temporal GraphRAG (TG-RAG)，一种改进的RAG系统，通过构建双层时序图（包含带时间戳的关系图和时间层次图）来显式建模知识的时间敏感性，支持动态查询和增量更新，并引入ECT-QA数据集验证其在处理时序知识和更新效率上的优越性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.13366v1">Document Intelligence in the Era of Large Language Models: A Survey</a></td><td><details><summary>展开</summary>Document AI (DAI) has emerged as a vital application area, and is
significantly transformed by the advent of large language models (LLMs). While
earlier approaches relied on encoder-decoder architectures, decoder-only LLMs
have revolutionized DAI, bringing remarkable advancements in understanding and
generation. This survey provides a comprehensive overview of DAI's evolution,
highlighting current research attempts and future prospects of LLMs in this
field. We explore key advancements and challenges in multimodal, multilingual,
and retrieval-augmented DAI, while also suggesting future research directions,
including agent-based approaches and document-specific foundation models. This
paper aims to provide a structured analysis of the state-of-the-art in DAI and
its implications for both academic and practical applications.</details></td><td><details><summary>展开</summary>这篇论文探讨了大型语言模型（LLMs）对文档人工智能（DAI）的变革性影响，特别关注了包括检索增强生成（RAG）在内的多模态、多语言技术进展，并提出了未来研究方向如基于代理的框架和文档专用基础模型。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.13363v1">D-SMART: Enhancing LLM Dialogue Consistency via Dynamic Structured Memory And Reasoning Tree</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) often exhibit factual inconsistencies and
logical decay in extended, multi-turn dialogues, a challenge stemming from
their reliance on static, pre-trained knowledge and an inability to reason
adaptively over the dialogue history. Prevailing mitigation strategies, such as
Retrieval-Augmented Generation (RAG) and agentic working memories, improve
information recall but still engage with fundamentally static knowledge sources
and follow pre-defined single reasoning path. This hinders their ability to
preserve factual and logical consistency of their responses in multi-turn
dialogues while the context evolves over time. To address this issue, we
propose D-SMART, a model-agnostic framework designed to maintain multi-turn
dialogue consistency by enabling LLMs to build and reason over a dynamic,
structured representation of the conversational context. This is achieved via
two synergistic components: (1) a Dynamic Structured Memory (DSM), which
incrementally constructs and maintains an authoritative, OWL-compliant
knowledge graph of the conversation; and (2) a Reasoning Tree (RT), which
executes inferences as an explicit and traceable multi-step search over the
graph. As the popular-used quality score (judged by GPT-4) can overlook logical
flaws, we introduce new NLI-based metrics to better measure multi-turn dialogue
consistency. Comprehensive experiments on the MT-Bench-101 benchmark show that
D-SMART significantly outperforms state-of-the-art baselines, elevating the
dialogue consistency score by over 48\% for both proprietary and open-source
models, and notably improves the quality score of the latter by up to 10.1\%.</details></td><td><details><summary>展开</summary>本文针对大语言模型（LLMs）在多轮对话中存在的逻辑衰减和事实不一致问题，提出了一种名为D-SMART的模型无关框架。该框架通过动态结构化记忆（DSM）和推理树（RT）来增强LLMs对对话上下文的动态理解和推理能力，显著提升了多轮对话的一致性。文章提及了现有方法（如RAG）的局限性，并引入了新的基于NLI的评估指标，实验结果表明D-SMART在一致性方面优于现有方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.13329v1">Embedding-Based Context-Aware Reranker</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) systems rely on retrieving relevant
evidence from a corpus to support downstream generation. The common practice of
splitting a long document into multiple shorter passages enables finer-grained
and targeted information retrieval. However, it also introduces challenges when
a correct retrieval would require inference across passages, such as resolving
coreference, disambiguating entities, and aggregating evidence scattered across
multiple sources. Many state-of-the-art (SOTA) reranking methods, despite
utilizing powerful large pretrained language models with potentially high
inference costs, still neglect the aforementioned challenges. Therefore, we
propose Embedding-Based Context-Aware Reranker (EBCAR), a lightweight reranking
framework operating directly on embeddings of retrieved passages with enhanced
cross-passage understandings through the structural information of the passages
and a hybrid attention mechanism, which captures both high-level interactions
across documents and low-level relationships within each document. We evaluate
EBCAR against SOTA rerankers on the ConTEB benchmark, demonstrating its
effectiveness for information retrieval requiring cross-passage inference and
its advantages in both accuracy and efficiency.</details></td><td><details><summary>展开</summary>本文提出了一种名为EBCAR的轻量级重排序框架，旨在解决RAG系统中跨段落推理的挑战（如共指消解和证据聚合），通过结合嵌入信息和混合注意力机制提升检索性能，并在ConTEB基准测试中验证了其高效性和准确性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.13312v1">ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering</a></td><td><details><summary>展开</summary>We present ChatR1, a reasoning framework based on reinforcement learning (RL)
for conversational question answering (CQA). Reasoning plays an important role
in CQA, where user intent evolves across dialogue turns, and utterances are
often underspecified, requiring contextual interpretation, query reformulation,
and dynamic coordination between retrieval and generation. Unlike static
`rewrite, retrieve, and generate' pipelines, ChatR1 interleaves search and
reasoning across turns, enabling exploratory and adaptive behaviors learned
through RL. To address the challenge of sparse and delayed rewards in RL, we
propose an intent-aware reward that provides turn-level feedback by aligning
retrieval and reasoning with evolving user goals. Our proposed ChatR1
demonstrates strong performance on both 3B and 7B model backbones,
outperforming competitive models on five CQA datasets, measured by different
metrics (F1, BERTScore, and LLM-as-judge). We include a diverse set of CQA
datasets to cover topic shifts, evolving intents, mixed-initiative dialogues,
and multi-document grounding, testing ChatR1's performance from various
aspects. Ablation studies confirm the effectiveness of the intent-aware reward.
Our analyses further reveal diverse reasoning trajectories and effective use of
the search tool. ChatR1 also generalizes robustly across domains, demonstrating
that RL-based reasoning enables more flexible and context-sensitive behavior
than static CQA pipelines.</details></td><td><details><summary>展开</summary>这篇论文提出了ChatR1，一个基于强化学习（RL）的对话式问答（CQA）推理框架，通过动态交织搜索和推理来适应用户意图的演变，利用意图感知奖励优化检索与生成的协同，显著提升了多任务场景下的性能表现。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.13272v1">Beyond Correctness: Rewarding Faithful Reasoning in Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Inspired by the success of reinforcement learning (RL) in Large Language
Model (LLM) training for domains like math and code, recent works have begun
exploring how to train LLMs to use search engines more effectively as tools for
retrieval-augmented generation. Although these methods achieve performance
improvement across QA benchmarks, many prioritize final answer correctness
while overlooking the quality of intermediate reasoning steps, which may lead
to chain-of-thought unfaithfulness. In this paper, we first introduce a
comprehensive evaluation framework for evaluating RL-based search agents,
covering three distinct faithfulness metrics: information-think faithfulness,
think-answer faithfulness, and think-search faithfulness. Our evaluations
reveal that a prototypical RL-based search agent, Search-R1, has significant
room for improvement in this regard. To foster faithful reasoning, we introduce
VERITAS (Verifying Entailed Reasoning through Intermediate Traceability in
Agentic Search), a novel framework that integrates fine-grained faithfulness
rewards into the reinforcement learning process. Our experiments show that
models trained with VERITAS not only significantly improve reasoning
faithfulness, but also achieve comparable task performance across seven QA
benchmarks.</details></td><td><details><summary>展开</summary>这篇论文探讨如何通过强化学习（RL）改进大语言模型（LLM）使用搜索引擎进行检索增强生成（RAG）时的推理忠实性问题，提出评估框架和VERITAS方法以提升中间推理步骤的质量，最终在多个QA基准测试中实现更高忠实性和任务性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.13193v1">ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG</a></td><td><details><summary>展开</summary>Knowledge graphs (KGs), with their structured representation capabilities,
offer promising avenue for enhancing Retrieval Augmented Generation (RAG)
systems, leading to the development of KG-RAG systems. Nevertheless, existing
methods often struggle to achieve effective synergy between system
effectiveness and cost efficiency, leading to neither unsatisfying performance
nor excessive LLM prompt tokens and inference time. To this end, this paper
proposes REMINDRAG, which employs an LLM-guided graph traversal featuring node
exploration, node exploitation, and, most notably, memory replay, to improve
both system effectiveness and cost efficiency. Specifically, REMINDRAG
memorizes traversal experience within KG edge embeddings, mirroring the way
LLMs "memorize" world knowledge within their parameters, but in a train-free
manner. We theoretically and experimentally confirm the effectiveness of
REMINDRAG, demonstrating its superiority over existing baselines across various
benchmark datasets and LLM backbones. Our code is available at
https://github.com/kilgrims/ReMindRAG.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为REMINDRAG的KG-RAG系统，通过LLM引导的图遍历（包含节点探索、节点利用和记忆回放）来提升知识图谱（KG）与RAG系统的协同效果，同时优化系统性能和成本效率。其核心创新是将遍历经验嵌入KG边表示中（无需训练），实验证明该方法在多个基准数据集和LLM主干模型上优于现有基线。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.13191v1">Grounding Long-Context Reasoning with Contextual Normalization for Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) has become an essential approach for
extending the reasoning and knowledge capacity of large language models (LLMs).
While prior research has primarily focused on retrieval quality and prompting
strategies, the influence of how the retrieved documents are framed, i.e.,
context format, remains underexplored. We show that seemingly superficial
choices, such as delimiters or structural markers in key-value extraction, can
induce substantial shifts in accuracy and stability, even when semantic content
is identical. To systematically investigate this effect, we design controlled
experiments that vary context density, delimiter styles, and positional
placement, revealing the underlying factors that govern performance
differences. Building on these insights, we introduce Contextual Normalization,
a lightweight strategy that adaptively standardizes context representations
before generation. Extensive experiments on both controlled and real-world RAG
benchmarks across diverse settings demonstrate that the proposed strategy
consistently improves robustness to order variation and strengthens
long-context utilization. These findings underscore that reliable RAG depends
not only on retrieving the right content, but also on how that content is
presented, offering both new empirical evidence and a practical technique for
better long-context reasoning.</details></td><td><details><summary>展开</summary>这篇论文探讨了检索增强生成（RAG）中上下文格式（如分隔符和结构标记）对模型性能和稳定性的影响，提出了一种称为“上下文归一化”的轻量级策略来标准化上下文表示，实验证明该策略可提升模型对顺序变化的鲁棒性和长上下文利用能力。</details></td></tr></tbody></table>

### 📅 2025-10-14
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.12801v1">DeepMMSearch-R1: Empowering Multimodal LLMs in Multimodal Web Search</a></td><td><details><summary>展开</summary>Multimodal Large Language Models (MLLMs) in real-world applications require
access to external knowledge sources and must remain responsive to the dynamic
and ever-changing real-world information in order to address
information-seeking and knowledge-intensive user queries. Existing approaches,
such as retrieval augmented generation (RAG) methods, search agents, and search
equipped MLLMs, often suffer from rigid pipelines, excessive search calls, and
poorly constructed search queries, which result in inefficiencies and
suboptimal outcomes. To address these limitations, we present DeepMMSearch-R1,
the first multimodal LLM capable of performing on-demand, multi-turn web
searches and dynamically crafting queries for both image and text search tools.
Specifically, DeepMMSearch-R1 can initiate web searches based on relevant crops
of the input image making the image search more effective, and can iteratively
adapt text search queries based on retrieved information, thereby enabling
self-reflection and self-correction. Our approach relies on a two-stage
training pipeline: a cold start supervised finetuning phase followed by an
online reinforcement learning optimization. For training, we introduce
DeepMMSearchVQA, a novel multimodal VQA dataset created through an automated
pipeline intermixed with real-world information from web search tools. This
dataset contains diverse, multi-hop queries that integrate textual and visual
information, teaching the model when to search, what to search for, which
search tool to use and how to reason over the retrieved information. We conduct
extensive experiments across a range of knowledge-intensive benchmarks to
demonstrate the superiority of our approach. Finally, we analyze the results
and provide insights that are valuable for advancing multimodal web-search.</details></td><td><details><summary>展开</summary>本文介绍了DeepMMSearch-R1，这是一种多模态大语言模型（MLLM），旨在通过动态生成图像和文本搜索查询、多轮迭代检索及自反思优化，解决现有检索增强生成（RAG）方法在管道僵化、搜索效率低下和查询质量不足等问题。文章提出两阶段训练方法（监督微调与强化学习优化）并构建了新型多模态VQA数据集DeepMMSearchVQA，通过实验验证了其在知识密集型任务中的优越性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.12668v1">The Role of Parametric Injection-A Systematic Study of Parametric Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) enhances large language models (LLMs) by
retrieving external documents. As an emerging form of RAG, parametric
retrieval-augmented generation (PRAG) encodes documents as model parameters
(i.e., LoRA modules) and injects these representations into the model during
inference, enabling interaction between the LLM and documents at parametric
level. Compared with directly placing documents in the input context, PRAG is
more efficient and has the potential to offer deeper model-document
interaction. Despite its growing attention, the mechanism underlying parametric
injection remains poorly understood. In this work, we present a systematic
study of PRAG to clarify the role of parametric injection, showing that
parameterized documents capture only partial semantic information of documents,
and relying on them alone yields inferior performance compared to interaction
at text level. However, these parametric representations encode high-level
document information that can enhance the model's understanding of documents
within the input context. When combined parameterized documents with textual
documents, the model can leverage relevant information more effectively and
become more robust to noisy inputs, achieving better performance than either
source alone. We recommend jointly using parameterized and textual documents
and advocate for increasing the information content of parametric
representations to advance PRAG.</details></td><td><details><summary>展开</summary>这篇论文探讨了一种新兴的RAG形式——参数化检索增强生成（PRAG），它通过将文档编码为模型参数（如LoRA模块）并在推理时注入这些表示，实现了LLM与文档在参数层面的交互。研究指出，PRAG虽比直接将文档放入输入上下文更高效，但单独使用时性能不如文本级别交互；然而，参数化表示能捕获高阶文档信息，若与文本结合可提升模型性能并增强抗噪能力。作者建议联合使用两种表示并提升参数化信息量以优化PRAG。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.12460v1">Probing Latent Knowledge Conflict for Faithful Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm to
enhance the factuality of Large Language Models (LLMs). However, existing RAG
systems often suffer from an unfaithfulness issue, where the model's response
contradicts evidence from the retrieved context. Existing approaches to
improving contextual faithfulness largely rely on external interventions, such
as prompt engineering, decoding constraints, or reward-based fine-tuning. These
works treat the LLM as a black box and overlook a crucial question: how does
the LLM internally integrate retrieved evidence with its parametric memory,
particularly under knowledge conflicts? To address this gap, we conduct a
probing-based analysis of hidden-state representations in LLMs and observe
three findings: knowledge integration occurs hierarchically, conflicts manifest
as latent signals at the sentence level, and irrelevant context is often
amplified when aligned with parametric knowledge. Building on these findings,
we propose CLEAR (Conflict-Localized and Enhanced Attention for RAG), a
framework that (i) decomposes context into fine-grained sentence-level
knowledge, (ii) employs hidden-state probing to localize conflicting knowledge,
and (iii) introduces conflict-aware fine-tuning to guide the model to
accurately integrate retrieved evidence. Extensive experiments across three
benchmarks demonstrate that CLEAR substantially improves both accuracy and
contextual faithfulness, consistently outperforming strong baselines under
diverse conflict conditions. The related resources are available at
https://github.com/LinfengGao/CLEAR.</details></td><td><details><summary>展开</summary>这篇论文探讨了现有检索增强生成（RAG）系统中存在的“不忠实”问题（模型响应与检索证据矛盾），提出通过分析大语言模型（LLM）内部隐藏状态表示来研究知识整合机制，并开发了CLEAR框架，通过句子级知识分解、冲突定位和冲突感知微调来提高RAG的准确性和上下文忠实性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.12434v1">PRoH: Dynamic Planning and Reasoning over Knowledge Hypergraphs for Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Knowledge Hypergraphs (KHs) have recently emerged as a knowledge
representation for retrieval-augmented generation (RAG), offering a paradigm to
model multi-entity relations into a structured form. However, existing KH-based
RAG methods suffer from three major limitations: static retrieval planning,
non-adaptive retrieval execution, and superficial use of KH structure and
semantics, which constrain their ability to perform effective multi-hop
question answering. To overcome these limitations, we propose PRoH, a dynamic
Planning and Reasoning over Knowledge Hypergraphs framework. PRoH incorporates
three core innovations: (i) a context-aware planning module that sketches the
local KH neighborhood to guide structurally grounded reasoning plan generation;
(ii) a structured question decomposition process that organizes subquestions as
a dynamically evolving Directed Acyclic Graph (DAG) to enable adaptive,
multi-trajectory exploration; and (iii) an Entity-Weighted Overlap (EWO)-guided
reasoning path retrieval algorithm that prioritizes semantically coherent
hyperedge traversals. Experiments across multiple domains demonstrate that PRoH
achieves state-of-the-art performance, surpassing the prior SOTA model
HyperGraphRAG by an average of 19.73% in F1 and 8.41% in Generation Evaluation
(G-E) score, while maintaining strong robustness in long-range multi-hop
reasoning tasks.</details></td><td><details><summary>展开</summary>本文提出PRoH框架，通过动态规划和知识超图推理改进现有基于知识超图的RAG方法，解决了静态检索规划、非适应性执行及结构语义利用不足三大问题，实验表明其在多跳问答任务中性能显著优于先前最优模型HyperGraphRAG。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.12323v1">RAG-Anything: All-in-One RAG Framework</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) has emerged as a fundamental paradigm
for expanding Large Language Models beyond their static training limitations.
However, a critical misalignment exists between current RAG capabilities and
real-world information environments. Modern knowledge repositories are
inherently multimodal, containing rich combinations of textual content, visual
elements, structured tables, and mathematical expressions. Yet existing RAG
frameworks are limited to textual content, creating fundamental gaps when
processing multimodal documents. We present RAG-Anything, a unified framework
that enables comprehensive knowledge retrieval across all modalities. Our
approach reconceptualizes multimodal content as interconnected knowledge
entities rather than isolated data types. The framework introduces dual-graph
construction to capture both cross-modal relationships and textual semantics
within a unified representation. We develop cross-modal hybrid retrieval that
combines structural knowledge navigation with semantic matching. This enables
effective reasoning over heterogeneous content where relevant evidence spans
multiple modalities. RAG-Anything demonstrates superior performance on
challenging multimodal benchmarks, achieving significant improvements over
state-of-the-art methods. Performance gains become particularly pronounced on
long documents where traditional approaches fail. Our framework establishes a
new paradigm for multimodal knowledge access, eliminating the architectural
fragmentation that constrains current systems. Our framework is open-sourced
at: https://github.com/HKUDS/RAG-Anything.</details></td><td><details><summary>展开</summary>这篇论文提出了名为"RAG-Anything"的统一框架，解决了现有RAG技术局限于文本检索的问题，通过将多模态内容重构为相互关联的知识实体，采用双图结构捕获跨模态关系和文本语义，实现跨模态混合检索，显著提升了多模态长文档的处理性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.12316v1">Beating Harmful Stereotypes Through Facts: RAG-based Counter-speech Generation</a></td><td><details><summary>展开</summary>Counter-speech generation is at the core of many expert activities, such as
fact-checking and hate speech, to counter harmful content. Yet, existing work
treats counter-speech generation as pure text generation task, mainly based on
Large Language Models or NGO experts. These approaches show severe drawbacks
due to the limited reliability and coherence in the generated countering text,
and in scalability, respectively. To close this gap, we introduce a novel
framework to model counter-speech generation as knowledge-wise text generation
process. Our framework integrates advanced Retrieval-Augmented Generation (RAG)
pipelines to ensure the generation of trustworthy counter-speech for 8 main
target groups identified in the hate speech literature, including women, people
of colour, persons with disabilities, migrants, Muslims, Jews, LGBT persons,
and other. We built a knowledge base over the United Nations Digital Library,
EUR-Lex and the EU Agency for Fundamental Rights, comprising a total of 32,792
texts. We use the MultiTarget-CONAN dataset to empirically assess the quality
of the generated counter-speech, both through standard metrics (i.e., JudgeLM)
and a human evaluation. Results show that our framework outperforms standard
LLM baselines and competitive approach, on both assessments. The resulting
framework and the knowledge base pave the way for studying trustworthy and
sound counter-speech generation, in hate speech and beyond.</details></td><td><details><summary>展开</summary>这篇文章提出了一种新颖的反仇恨言论生成框架，将其建模为基于知识的文本生成过程，通过集成检索增强生成（RAG）技术，从联合国数字图书馆等构建的知识库中检索信息，以确保生成可信的反仇恨言论。实验结果表明，该框架在标准评估和人工评估中均优于传统大语言模型基准。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.12171v1">MatSciBench: Benchmarking the Reasoning Ability of Large Language Models in Materials Science</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) have demonstrated remarkable abilities in
scientific reasoning, yet their reasoning capabilities in materials science
remain underexplored. To fill this gap, we introduce MatSciBench, a
comprehensive college-level benchmark comprising 1,340 problems that span the
essential subdisciplines of materials science. MatSciBench features a
structured and fine-grained taxonomy that categorizes materials science
questions into 6 primary fields and 31 sub-fields, and includes a three-tier
difficulty classification based on the reasoning length required to solve each
question. MatSciBench provides detailed reference solutions enabling precise
error analysis and incorporates multimodal reasoning through visual contexts in
numerous questions. Evaluations of leading models reveal that even the
highest-performing model, Gemini-2.5-Pro, achieves under 80% accuracy on
college-level materials science questions, highlighting the complexity of
MatSciBench. Our systematic analysis of different reasoning strategie--basic
chain-of-thought, tool augmentation, and self-correction--demonstrates that no
single method consistently excels across all scenarios. We further analyze
performance by difficulty level, examine trade-offs between efficiency and
accuracy, highlight the challenges inherent in multimodal reasoning tasks,
analyze failure modes across LLMs and reasoning methods, and evaluate the
influence of retrieval-augmented generation. MatSciBench thus establishes a
comprehensive and solid benchmark for assessing and driving improvements in the
scientific reasoning capabilities of LLMs within the materials science domain.</details></td><td><details><summary>展开</summary>这篇文章介绍了MatSciBench，一个针对材料科学领域的全面大学级基准测试，包含1,340个问题，覆盖多个子领域和难度层级。文章评估了领先大语言模型的性能，分析了不同推理策略（包括检索增强生成）的效果，并指出该基准在提升材料科学推理能力方面的价值。</details></td></tr></tbody></table>

### 📅 2025-10-13
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.11654v1">FinVet: A Collaborative Framework of RAG and External Fact-Checking Agents for Financial Misinformation Detection</a></td><td><details><summary>展开</summary>Financial markets face growing threats from misinformation that can trigger
billions in losses in minutes. Most existing approaches lack transparency in
their decision-making and provide limited attribution to credible sources. We
introduce FinVet, a novel multi-agent framework that integrates two
Retrieval-Augmented Generation (RAG) pipelines with external fact-checking
through a confidence-weighted voting mechanism. FinVet employs adaptive
three-tier processing that dynamically adjusts verification strategies based on
retrieval confidence, from direct metadata extraction to hybrid reasoning to
full model-based analysis. Unlike existing methods, FinVet provides
evidence-backed verdicts, source attribution, confidence scores, and explicit
uncertainty flags when evidence is insufficient. Experimental evaluation on the
FinFact dataset shows that FinVet achieves an F1 score of 0.85, which is a
10.4% improvement over the best individual pipeline (fact-check pipeline) and
37% improvement over standalone RAG approaches.</details></td><td><details><summary>展开</summary>该论文提出了一种名为FinVet的多智能体框架，通过整合两条RAG管道和外部事实核查机制，结合置信度加权投票进行金融信息验证，实现动态三层级处理策略，提供证据支持、来源追溯及不确定性标注，在FinFact数据集上的实验表明其F1分数显著优于独立RAG方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.11541v1">Query-Specific GNN: A Comprehensive Graph Representation Learning Method for Retrieval Augmented Generation</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) has demonstrated its ability to enhance
Large Language Models (LLMs) by integrating external knowledge sources.
However, multi-hop questions, which require the identification of multiple
knowledge targets to form a synthesized answer, raise new challenges for RAG
systems. Under the multi-hop settings, existing methods often struggle to fully
understand the questions with complex semantic structures and are susceptible
to irrelevant noise during the retrieval of multiple information targets. To
address these limitations, we propose a novel graph representation learning
framework for multi-hop question retrieval. We first introduce a
Multi-information Level Knowledge Graph (Multi-L KG) to model various
information levels for a more comprehensive understanding of multi-hop
questions. Based on this, we design a Query-Specific Graph Neural Network
(QSGNN) for representation learning on the Multi-L KG. QSGNN employs
intra/inter-level message passing mechanisms, and in each message passing the
information aggregation is guided by the query, which not only facilitates
multi-granular information aggregation but also significantly reduces the
impact of noise. To enhance its ability to learn robust representations, we
further propose two synthesized data generation strategies for pre-training the
QSGNN. Extensive experimental results demonstrate the effectiveness of our
framework in multi-hop scenarios, especially in high-hop questions the
improvement can reach 33.8\%. The code is available at:
https://github.com/Jerry2398/QSGNN.</details></td><td><details><summary>展开</summary>本文针对RAG系统在多跳问题（需检索多目标知识合成答案）中的局限性，提出了一种新的图表示学习框架。通过构建多层次知识图谱（Multi-L KG）和设计基于查询的图神经网络（QSGNN），利用跨层级信息传递和噪声抑制机制提升多跳检索性能，并结合合成数据预训练策略，实验表明其在多跳场景（尤其是高跳问题）中效果显著提升达33.8%。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.11483v1">Uncertainty Quantification for Retrieval-Augmented Reasoning</a></td><td><details><summary>展开</summary>Retrieval-augmented reasoning (RAR) is a recent evolution of
retrieval-augmented generation (RAG) that employs multiple reasoning steps for
retrieval and generation. While effective for some complex queries, RAR remains
vulnerable to errors and misleading outputs. Uncertainty quantification (UQ)
offers methods to estimate the confidence of systems' outputs. These methods,
however, often handle simple queries with no retrieval or single-step
retrieval, without properly handling RAR setup. Accurate estimation of UQ for
RAR requires accounting for all sources of uncertainty, including those arising
from retrieval and generation. In this paper, we account for all these sources
and introduce Retrieval-Augmented Reasoning Consistency (R2C)--a novel UQ
method for RAR. The core idea of R2C is to perturb the multi-step reasoning
process by applying various actions to reasoning steps. These perturbations
alter the retriever's input, which shifts its output and consequently modifies
the generator's input at the next step. Through this iterative feedback loop,
the retriever and generator continuously reshape one another's inputs, enabling
us to capture uncertainty arising from both components. Experiments on five
popular RAR systems across diverse QA datasets show that R2C improves AUROC by
over 5% on average compared to the state-of-the-art UQ baselines. Extrinsic
evaluations using R2C as an external signal further confirm its effectiveness
for two downstream tasks: in Abstention, it achieves ~5% gains in both
F1Abstain and AccAbstain; in Model Selection, it improves the exact match by
~7% over single models and ~3% over selection methods.</details></td><td><details><summary>展开</summary>本文提出了一种针对检索增强推理（RAR，RAG的多步推理扩展）的新型不确定性量化方法R2C，通过扰动多步推理过程并迭代反馈检索与生成组件的输入差异，显著提升了复杂查询下的不确定性估计性能，实验证明其在多项下游任务中优于现有基线。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.11394v1">VeriCite: Towards Reliable Citations in Retrieval-Augmented Generation via Rigorous Verification</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) has emerged as a crucial approach for
enhancing the responses of large language models (LLMs) with external knowledge
sources. Despite the impressive performance in complex question-answering
tasks, RAG still struggles with hallucinations. Attributing RAG-generated
content through in-line citations has demonstrated potential in reducing
hallucinations and facilitating human verification. Existing citation
generation methods primarily rely on either fine-tuning the generator or
employing post-processing approaches for citation matching. However, the former
approach demands substantial annotated data and computational resources, while
the latter often encounters difficulties in managing multiple citations and
frequently produces suboptimal results. In this paper, we introduce a novel
framework, called VeriCite, designed to rigorously validate supporting evidence
and enhance answer attribution. Specifically, VeriCite breaks down into a
three-stage generation: 1) The initial answer generation first generates a
response based on all available contexts and has its claims verified through
the NLI model; 2) the supporting evidence selection assesses the utility of
each document and extracts useful supporting evidences; 3) the final answer
refinement integrates the initial response and collected evidences to produce
the final, refined answer.We conduct experiments across five open-source LLMs
and four datasets, demonstrating that VeriCite can significantly improve
citation quality while maintaining the correctness of the answers.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为VeriCite的新框架，旨在解决RAG技术在生成内容时可能出现幻觉的问题。该框架通过三阶段生成过程（初始答案生成、支持证据选择、最终答案精炼）来验证支持证据并增强答案的可追溯性，实验表明VeriCite能显著提高引用质量并保持答案准确性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.11358v1">LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) enhances large language models (LLMs) by
incorporating external knowledge. While traditional retrieval focuses on
relevance, RAG's effectiveness depends on the utility of retrieved passages,
i.e., the usefulness in facilitating the generation of an accurate and
comprehensive answer. Existing studies often treat utility as a generic
attribute, ignoring the fact that different LLMs may benefit differently from
the same passage due to variations in internal knowledge and comprehension
ability. In this work, we introduce and systematically investigate the notion
of LLM-specific utility. Through large-scale experiments across multiple
datasets and LLMs, we demonstrate that human-annotated passages are not optimal
for LLMs and that ground-truth utilitarian passages are not transferable across
different LLMs. These findings highlight the necessity of adopting the
LLM-specific utility in RAG research. Our findings indicate that some
human-annotated passages are not ground-truth utilitarian passages for specific
LLMs, partially due to the varying readability of queries and passages for
LLMs, a tendency for which perplexity is a key metric. Based on these findings,
we propose a benchmarking procedure for LLM-specific utility judgments. We
evaluate existing utility judgment methods on six datasets and find that while
verbalized methods using pseudo-answers perform robustly, LLMs struggle to
assess utility effectively-failing to reject all passages for known queries and
to select truly useful ones for unknown queries.</details></td><td><details><summary>展开</summary>这篇论文探讨了检索增强生成（RAG）中检索内容的效用问题，提出并研究了LLM特定的效用概念。通过大规模实验，作者发现人类标注的段落对不同LLM并非最优，且效用段落在不同LLM间不可迁移，强调了在RAG研究中考虑LLM特定效用的必要性。论文还提出了基于LLM特定效用的基准测试流程，并评估了现有效用判断方法的性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.11217v1">Domain-Specific Data Generation Framework for RAG Adaptation</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) combines the language understanding and
reasoning power of large language models (LLMs) with external retrieval to
enable domain-grounded responses. Effectively adapting RAG systems to
domain-specific settings requires specialized, context-rich training data
beyond general-purpose question-answering. Here, we propose RAGen, a scalable
and modular framework for generating domain-grounded question-answer-context
(QAC) triples tailored to diverse RAG adaptation approaches. RAGen produces
these QAC triples by identifying key concepts in documents, generating diverse
questions guided by Bloom's Taxonomy-inspired principles, and pairing them with
precise answers extracted from relevant contexts. RAGen supports multiple RAG
adaptation strategies, including the optimization of key components such as the
LLM, retriever, and embedding model, etc. Its modular pipeline features
semantic chunking, hierarchical concept extraction, and multi-chunk retrieval,
along with the introduction of curated distractor contexts to promote robust
reasoning. Designed for scalability, RAGen efficiently handles large and
evolving document corpora without redundant processing, making it especially
suitable for dynamic evolving domains such as scientific research and
enterprise knowledge bases.</details></td><td><details><summary>展开</summary>这篇论文提出了RAGen，一个可扩展且模块化的框架，用于生成针对特定领域定制的问答上下文三元组（QAC），以支持不同RAG系统的适应性优化，包括改进大语言模型、检索器和嵌入模型等关键组件，并适用于动态演进的领域如科研和企业知识库。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.11195v1">RAG-Pull: Imperceptible Attacks on RAG Systems for Code Generation</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) increases the reliability and
trustworthiness of the LLM response and reduces hallucination by eliminating
the need for model retraining. It does so by adding external data into the
LLM's context. We develop a new class of black-box attack, RAG-Pull, that
inserts hidden UTF characters into queries or external code repositories,
redirecting retrieval toward malicious code, thereby breaking the models'
safety alignment. We observe that query and code perturbations alone can shift
retrieval toward attacker-controlled snippets, while combined query-and-target
perturbations achieve near-perfect success. Once retrieved, these snippets
introduce exploitable vulnerabilities such as remote code execution and SQL
injection. RAG-Pull's minimal perturbations can alter the model's safety
alignment and increase preference towards unsafe code, therefore opening up a
new class of attacks on LLMs.</details></td><td><details><summary>展开</summary>本文提出了一种针对RAG的新攻击方法RAG-Pull，通过在查询或外部代码库中插入隐藏的UTF字符，将检索结果导向恶意代码，从而破坏模型的安全对齐性，导致远程代码执行和SQL注入等漏洞被利用。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.11122v1">DyKnow-RAG: Dynamic Knowledge Utilization Reinforcement Framework for Noisy Retrieval-Augmented Generation in E-commerce Search Relevance</a></td><td><details><summary>展开</summary>Accurately modeling query-item relevance drives e-commerce ranking, yet
long-tail, knowledge-heavy, and fast-evolving queries exceed parametric LLM
coverage. External context (reviews, attribute encyclopedias, UGC) can help but
is noisy, and single-pass latency and cost forbid any clean-then-summarize
step. The model must, per query, judge relevance and decide whether to use,
partially use, or ignore the context. DyKnow-RAG is a dynamic noisy-RAG
framework built on Group Relative Policy Optimization. It trains two rollout
groups (no external context vs a single retrieved chunk) and applies
posterior-driven inter-group advantage scaling that adaptively reweights their
contributions by the per-query correctness gap. This teaches when to trust
retrieval versus fall back to parametric knowledge, without process labels,
value networks, or extra inference passes, preserving single-pass, single-chunk
deployment under production latency. Training combines: (1) supervised
initialization with a structured rationale that explicitly records the
context-usage decision; (2) an RL pool prioritized by SFT uncertainty to focus
where context choice is most consequential; and (3) an optional lightweight DPO
warm start to stabilize with-context calibration. Under a unified
retrieval/index and fixed latency budget, DyKnow-RAG outperforms SFT, DPO, and
vanilla GRPO in offline tests, and delivers consistent lifts on GSB, Query
Goodrate, and Item Goodrate in Taobao A/B testing. It is deployed in Taobao's
production relevance system, serving live traffic. To our knowledge, it is
among the first single-pass RAG solutions for e-commerce relevance, turning
noisy external signals into reliable gains without added online complexity.</details></td><td><details><summary>展开</summary>这篇论文提出了DyKnow-RAG框架，通过动态噪声检索增强生成（Dynamic Noisy-RAG）技术优化电商排序中的查询-商品相关性建模。该系统基于Group Relative Policy Optimization（GRPO），结合监督学习和强化学习，动态决定何时利用检索的外部上下文（如评论、百科等），解决长尾、知识密集及快速变化查询的覆盖问题。DyKnow-RAG在训练中通过自适应权重调整和单次推理部署，实现了检索可信度判断与参数化知识的互补，最终在淘宝生产环境中显著提升了相关性指标。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.10931v1">PoU: Proof-of-Use to Counter Tool-Call Hacking in DeepResearch Agents</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) agents, such as recent
DeepResearch-style systems, extend large language models (LLMs) with autonomous
information-seeking capabilities through external tools. While reinforcement
learning (RL) has enabled impressive multi-step reasoning, we identify a
previously overlooked failure mode, Tool-Call Hacking, where agents inflate
reward signals by issuing superficially correct tool calls without genuinely
leveraging the retrieved evidence. This results in (i) mode collapse into
repetitive reliance on a single source and (ii) spurious grounding, where
answers are only weakly supported by cited content.
  To address this, we propose Proof-of-Use (PoU), an evidence-grounded RL
framework that enforces verifiable causal links between retrieved evidence,
reasoning traces, and final answers. PoU operationalizes this through a unified
step-wise contract combining syntactic citation validation, perturbation-based
sensitivity rewards, and answer-evidence alignment objectives, ensuring that
tool usage remains both interpretable and functionally grounded.
  Across seven QA benchmarks spanning in-domain, out-of-domain, and
out-of-tool-distribution settings, PoU consistently outperforms strong
DeepResearch baselines in factual accuracy, evidence faithfulness, and
tool-routing balance. These findings highlight the necessity of grounding
RL-trained agents not merely in task outcomes but in the causal use of
retrieved information, offering a principled path toward trustworthy
retrieval-augmented reasoning.</details></td><td><details><summary>展开</summary>该论文探讨了检索增强生成（RAG）智能体中的"工具调用黑客"问题，即模型通过表面正确的工具调用而非实际利用检索证据来提升奖励信号，导致模式崩溃和虚假引用，并提出了一种名为"使用证明"（PoU）的新型强化学习框架，通过结合语法引用验证、基于扰动的敏感性奖励和答案-证据对齐目标，确保工具使用的可解释性和功能性基础，在多个问答基准测试中展现出优于现有方法的性能。</details></td></tr></tbody></table>

### 📅 2025-10-12
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.10828v1">VeritasFi: An Adaptable, Multi-tiered RAG Framework for Multi-modal Financial Question Answering</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) is becoming increasingly essential for
Question Answering (QA) in the financial sector, where accurate and
contextually grounded insights from complex public disclosures are crucial.
However, existing financial RAG systems face two significant challenges: (1)
they struggle to process heterogeneous data formats, such as text, tables, and
figures; and (2) they encounter difficulties in balancing general-domain
applicability with company-specific adaptation. To overcome these challenges,
we present VeritasFi, an innovative hybrid RAG framework that incorporates a
multi-modal preprocessing pipeline alongside a cutting-edge two-stage training
strategy for its re-ranking component. VeritasFi enhances financial QA through
three key innovations: (1) A multi-modal preprocessing pipeline that seamlessly
transforms heterogeneous data into a coherent, machine-readable format. (2) A
tripartite hybrid retrieval engine that operates in parallel, combining deep
multi-path retrieval over a semantically indexed document corpus, real-time
data acquisition through tool utilization, and an expert-curated memory bank
for high-frequency questions, ensuring comprehensive scope, accuracy, and
efficiency. (3) A two-stage training strategy for the document re-ranker, which
initially constructs a general, domain-specific model using anonymized data,
followed by rapid fine-tuning on company-specific data for targeted
applications. By integrating our proposed designs, VeritasFi presents a
groundbreaking framework that greatly enhances the adaptability and robustness
of financial RAG systems, providing a scalable solution for both general-domain
and company-specific QA tasks. Code accompanying this work is available at
https://github.com/simplew4y/VeritasFi.git.</details></td><td><details><summary>展开</summary>这篇论文介绍了一种名为VeritasFi的混合RAG框架，针对金融领域问答系统中的异构数据处理和通用性与公司特定适应性平衡问题，提出了多模态预处理流水线、三重混合检索引擎及两阶段文档重排序训练策略，以提升金融RAG系统的准确性和适应性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.10824v1">Agentic RAG for Software Testing with Hybrid Vector-Graph and Multi-Agent Orchestration</a></td><td><details><summary>展开</summary>We present an approach to software testing automation using Agentic
Retrieval-Augmented Generation (RAG) systems for Quality Engineering (QE)
artifact creation. We combine autonomous AI agents with hybrid vector-graph
knowledge systems to automate test plan, case, and QE metric generation. Our
approach addresses traditional software testing limitations by leveraging LLMs
such as Gemini and Mistral, multi-agent orchestration, and enhanced
contextualization. The system achieves remarkable accuracy improvements from
65% to 94.8% while ensuring comprehensive document traceability throughout the
quality engineering lifecycle. Experimental validation of enterprise Corporate
Systems Engineering and SAP migration projects demonstrates an 85% reduction in
testing timeline, an 85% improvement in test suite efficiency, and projected
35% cost savings, resulting in a 2-month acceleration of go-live.</details></td><td><details><summary>展开</summary>该论文提出了一种利用基于代理的检索增强生成（Agentic RAG）系统自动化生成质量工程（QE）工件的方法，结合自主AI代理与混合向量-图知识系统，显著提升了测试计划、用例和指标的生成准确性（65%→94.8%），并在企业级项目中验证了85%的测试时间缩减和35%成本节约效果。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.10815v1">DRIFT: Decompose, Retrieve, Illustrate, then Formalize Theorems</a></td><td><details><summary>展开</summary>Automating the formalization of mathematical statements for theorem proving
remains a major challenge for Large Language Models (LLMs). LLMs struggle to
identify and utilize the prerequisite mathematical knowledge and its
corresponding formal representation in languages like Lean. Current
retrieval-augmented autoformalization methods query external libraries using
the informal statement directly, but overlook a fundamental limitation:
informal mathematical statements are often complex and offer limited context on
the underlying math concepts. To address this, we introduce DRIFT, a novel
framework that enables LLMs to decompose informal mathematical statements into
smaller, more tractable ''sub-components''. This facilitates targeted retrieval
of premises from mathematical libraries such as Mathlib. Additionally, DRIFT
retrieves illustrative theorems to help models use premises more effectively in
formalization tasks. We evaluate DRIFT across diverse benchmarks (ProofNet,
ConNF, and MiniF2F-test) and find that it consistently improves premise
retrieval, nearly doubling the F1 score compared to the DPR baseline on
ProofNet. Notably, DRIFT demonstrates strong performance on the
out-of-distribution ConNF benchmark, with BEq+@10 improvements of 37.14% and
42.25% using GPT-4.1 and DeepSeek-V3.1, respectively. Our analysis shows that
retrieval effectiveness in mathematical autoformalization depends heavily on
model-specific knowledge boundaries, highlighting the need for adaptive
retrieval strategies aligned with each model's capabilities.</details></td><td><details><summary>展开</summary>这篇论文提出了DRIFT框架，通过将非正式数学陈述分解为更小的子组件来改进检索增强的自动形式化方法，并利用针对性检索数学库中的前提和示例定理，显著提升了LLMs在数学定理证明中的形式化能力。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.10806v1">Is Implicit Knowledge Enough for LLMs? A RAG Approach for Tree-based Structures</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) are adept at generating responses based on
information within their context. While this ability is useful for interacting
with structured data like code files, another popular method,
Retrieval-Augmented Generation (RAG), retrieves relevant documents to augment
the model's in-context learning. However, it is not well-explored how to best
represent this retrieved knowledge for generating responses on structured data,
particularly hierarchical structures like trees. In this work, we propose a
novel bottom-up method to linearize knowledge from tree-like structures (like a
GitHub repository) by generating implicit, aggregated summaries at each
hierarchical level. This approach enables the knowledge to be stored in a
knowledge base and used directly with RAG. We then compare our method to using
RAG on raw, unstructured code, evaluating the accuracy and quality of the
generated responses. Our results show that while response quality is comparable
across both methods, our approach generates over 68% fewer documents in the
retriever, a significant gain in efficiency. This finding suggests that
leveraging implicit, linearized knowledge may be a highly effective and
scalable strategy for handling complex, hierarchical data structures.</details></td><td><details><summary>展开</summary>这篇论文探讨了如何优化RAG在处理结构化数据（如树形结构的GitHub仓库）时的知识表示方法，提出了一种自底向上生成隐式聚合摘要的新方法，显著减少了检索文档数量，同时保持了回答质量。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.10787v1">Review of Inference-Time Scaling Strategies: Reasoning, Search and RAG</a></td><td><details><summary>展开</summary>The performance gains of LLMs have historically been driven by scaling up
model size and training data. However, the rapidly diminishing availability of
high-quality training data is introducing a fundamental bottleneck, shifting
the focus of research toward inference-time scaling. This paradigm uses
additional computation at the time of deployment to substantially improve LLM
performance on downstream tasks without costly model re-training. This review
systematically surveys the diverse techniques contributing to this new era of
inference-time scaling, organizing the rapidly evolving field into two
comprehensive perspectives: Output-focused and Input-focused methods.
Output-focused techniques encompass complex, multi-step generation strategies,
including reasoning (e.g., CoT, ToT, ReAct), various search and decoding
methods (e.g., MCTS, beam search), training for long CoT (e.g., RLVR, GRPO),
and model ensemble methods. Input-focused techniques are primarily categorized
by few-shot and RAG, with RAG as the central focus. The RAG section is further
detailed through a structured examination of query expansion, data, retrieval
and reranker, LLM generation methods, and multi-modal RAG.</details></td><td><details><summary>展开</summary>这篇论文系统地调研了推理时扩展（inference-time scaling）的多样化技术，重点关注输入导向方法中的检索增强生成（RAG），并详细探讨了RAG的查询扩展、数据检索、重排序、生成方法及多模态应用，将其视为提升大语言模型性能的关键方向之一。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.10549v1">ELAIPBench: A Benchmark for Expert-Level Artificial Intelligence Paper Understanding</a></td><td><details><summary>展开</summary>While large language models (LLMs) excel at many domain-specific tasks, their
ability to deeply comprehend and reason about full-length academic papers
remains underexplored. Existing benchmarks often fall short of capturing such
depth, either due to surface-level question design or unreliable evaluation
metrics. To address this gap, we introduce ELAIPBench, a benchmark curated by
domain experts to evaluate LLMs' comprehension of artificial intelligence (AI)
research papers. Developed through an incentive-driven, adversarial annotation
process, ELAIPBench features 403 multiple-choice questions from 137 papers. It
spans three difficulty levels and emphasizes non-trivial reasoning rather than
shallow retrieval. Our experiments show that the best-performing LLM achieves
an accuracy of only 39.95%, far below human performance. Moreover, we observe
that frontier LLMs equipped with a thinking mode or a retrieval-augmented
generation (RAG) system fail to improve final results-even harming accuracy due
to overthinking or noisy retrieval. These findings underscore the significant
gap between current LLM capabilities and genuine comprehension of academic
papers.</details></td><td><details><summary>展开</summary>这篇论文介绍了ELAIPBench，一个用于评估大语言模型（LLMs）对人工智能研究论文理解能力的专家标注基准，研究发现即便配备检索增强生成（RAG）系统的先进LLMs也无法提升性能，甚至因检索噪声而降低准确性，揭示了LLMs在深度理解学术论文方面的不足。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.10480v1">Latent Retrieval Augmented Generation of Cross-Domain Protein Binders</a></td><td><details><summary>展开</summary>Designing protein binders targeting specific sites, which requires to
generate realistic and functional interaction patterns, is a fundamental
challenge in drug discovery. Current structure-based generative models are
limited in generating nterfaces with sufficient rationality and
interpretability. In this paper, we propose Retrieval-Augmented Diffusion for
Aligned interface (RADiAnce), a new framework that leverages known interfaces
to guide the design of novel binders. By unifying retrieval and generation in a
shared contrastive latent space, our model efficiently identifies relevant
interfaces for a given binding site and seamlessly integrates them through a
conditional latent diffusion generator, enabling cross-domain interface
transfer. Extensive exeriments show that RADiAnce significantly outperforms
baseline models across multiple metrics, including binding affinity and
recovery of geometries and interactions. Additional experimental results
validate cross-domain generalization, demonstrating that retrieving interfaces
from diverse domains, such as peptides, antibodies, and protein fragments,
enhances the generation performance of binders for other domains. Our work
establishes a new paradigm for protein binder design that successfully bridges
retrieval-based knowledge and generative AI, opening new possibilities for drug
discovery.</details></td><td><details><summary>展开</summary>这篇论文提出了一个名为RADiAnce的框架，通过结合检索已知蛋白质接口和生成新绑定剂的方法，利用共享对比潜空间和条件潜扩散生成器，实现了跨领域接口转移，显著提升了蛋白质绑定剂设计的性能，为药物发现提供了新的可能性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.10452v1">Steering Over-refusals Towards Safety in Retrieval Augmented Generation</a></td><td><details><summary>展开</summary>Safety alignment in large language models (LLMs) induces over-refusals --
where LLMs decline benign requests due to aggressive safety filters. We analyze
this phenomenon in retrieval-augmented generation (RAG), where both the query
intent and retrieved context properties influence refusal behavior. We
construct RagRefuse, a domain-stratified benchmark spanning medical, chemical,
and open domains, pairing benign and harmful queries with controlled context
contamination patterns and sizes. Our analysis shows that context arrangement /
contamination, domain of query and context, and harmful-text density trigger
refusals even on benign queries, with effects depending on model-specific
alignment choices. To mitigate over-refusals, we introduce
\textsc{SafeRAG-Steering}, a model-centric embedding intervention that steers
the embedding regions towards the confirmed safe, non-refusing output regions
at inference time. This reduces over-refusals in contaminated RAG pipelines
while preserving legitimate refusals.</details></td><td><details><summary>展开</summary>该论文探讨了大型语言模型（LLMs）在检索增强生成（RAG）中因安全对齐导致的过度拒绝问题，构建了多领域基准测试RagRefuse分析影响因素，并提出了SafeRAG-Steering方法，通过嵌入干预减少无害查询的误拒，同时保持对有害请求的合理拒绝。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.10448v1">RECON: Reasoning with Condensation for Efficient Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) systems trained using reinforcement
learning (RL) with reasoning are hampered by inefficient context management,
where long, noisy retrieved documents increase costs and degrade performance.
We introduce RECON (REasoning with CONdensation), a framework that integrates
an explicit summarization module to compress evidence within the reasoning
loop. Our summarizer is trained via a two-stage process: relevance pretraining
on QA datasets, followed by multi-aspect distillation from proprietary LLMs to
ensure factuality and clarity. Integrated into the Search-R1 pipeline, RECON
reduces total context length by 35\%, leading to improved training speed and
inference latency, while simultaneously improving RAG performance on downstream
QA benchmarks. Notably, it boosts the average EM score of the 3B model by
14.5\% and the 7B model by 3.0\%, showing particular strength in multi-hop QA.
RECON demonstrates that learned context compression is essential for building
practical, scalable, and performant RAG systems. Our code implementation is
made available at https://github.com/allfornancy/RECON.</details></td><td><details><summary>展开</summary>本文提出了一种名为RECON的框架，通过引入显式摘要模块在推理循环中压缩检索证据，解决了RAG系统中长且噪声文档导致的效率低下问题，并结合强化学习和多阶段训练方法，显著减少了上下文长度并提升了QA任务性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.10426v1">Taming a Retrieval Framework to Read Images in Humanlike Manner for Augmenting Generation of MLLMs</a></td><td><details><summary>展开</summary>Multimodal large language models (MLLMs) often fail in fine-grained visual
question answering, producing hallucinations about object identities,
positions, and relations because textual queries are not explicitly anchored to
visual referents. Retrieval-augmented generation (RAG) alleviates some errors,
but it fails to align with human-like processing at both the retrieval and
augmentation levels. Specifically, it focuses only on global-level image
information but lacks local detail and limits reasoning about fine-grained
interactions. To overcome this limitation, we present Human-Like
Retrieval-Augmented Generation (HuLiRAG), a framework that stages multimodal
reasoning as a ``what--where--reweight'' cascade. Queries are first anchored to
candidate referents via open-vocabulary detection (what), then spatially
resolved with SAM-derived masks to recover fine-grained precision (where), and
adaptively prioritized through the trade-off between local and global alignment
(reweight). Mask-guided fine-tuning further injects spatial evidence into the
generation process, transforming grounding from a passive bias into an explicit
constraint on answer formulation. Extensive experiments demonstrate that this
human-like cascade improves grounding fidelity and factual consistency while
reducing hallucinations, advancing multimodal question answering toward
trustworthy reasoning.</details></td><td><details><summary>展开</summary>本文提出了一种名为HuLiRAG的新型框架，通过"What-Where-Reweight"多级推理机制改进传统RAG在细粒度视觉问答中的不足。该框架结合开放词汇检测（定位物体）、SAM分割（精确定位空间关系）和自适应权重调整，并引入掩码导向微调，显著减少了多模态大语言模型在物体识别、位置和关系推理中的幻觉问题，提升了回答的可靠性和事实一致性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.10390v1">RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models</a></td><td><details><summary>展开</summary>The ability of language models in RAG systems to selectively refuse to answer
based on flawed context is critical for safety, yet remains a significant
failure point. Our large-scale study reveals that even frontier models struggle
in this setting, with refusal accuracy dropping below 50% on multi-document
tasks, while exhibiting either dangerous overconfidence or overcaution. Static
benchmarks fail to reliably evaluate this capability, as models exploit
dataset-specific artifacts and memorize test instances. We introduce
RefusalBench, a generative methodology that programmatically creates diagnostic
test cases through controlled linguistic perturbation. Our framework employs
176 distinct perturbation strategies across six categories of informational
uncertainty and three intensity levels. Evaluation of over 30 models uncovers
systematic failure patterns: refusal comprises separable detection and
categorization skills, and neither scale nor extended reasoning improves
performance. We find that selective refusal is a trainable, alignment-sensitive
capability, offering a clear path for improvement. We release two benchmarks --
RefusalBench-NQ (single document) and RefusalBench-GaRAGe (multi-document) --
and our complete generation framework to enable continued, dynamic evaluation
of this critical capability.</details></td><td><details><summary>展开</summary>这篇论文研究了RAG系统中语言模型基于有缺陷的上下文选择拒绝回答的能力，发现前沿模型在该任务上表现不佳，并提出RefusalBench这一通过程序化生成诊断测试案例的方法论框架，同时发布了两个基准测试集以改进模型的关键拒绝能力。</details></td></tr></tbody></table>

### 📅 2025-10-11
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.10129v1">CacheClip: Accelerating RAG with Effective KV Cache Reuse</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) systems suffer from severe
time-to-first-token (TTFT) bottlenecks due to long input sequences. Existing KV
cache reuse methods face a fundamental trade-off: prefix caching requires
identical prefixes that rarely occur in RAG scenarios, while direct
precomputation sacrifices quality due to missing inter-chunk attention and
repeated attention sinks. Recent methods like APE and CacheBlend partially
address these issues but remain inadequate for robust RAG applications. This
paper presents CacheClip, a novel framework that achieves both fast TTFT and
high generation quality. Our key insight is that small auxiliary LLMs exhibit
similar last-layer attention distributions to primary LLMs (the target model
for generation), enabling efficient identification of tokens critical for
restoring inter-chunk attention, thereby significantly improving response
quality on cross-chunk reasoning tasks. CacheClip integrates three techniques:
(1) auxiliary-model-guided token selection for selective KV cache
recomputation, where the auxiliary model is finetuned to improve selection
accuracy, (2) shared prefixes to eliminate redundant attention sinks, and (3)
grouping strategy to maintain local coherence during partial KV cache updates.
Experiments show CacheClip retains up to 94.8% and 85.0% of full-attention
performance on NIAH and LongBench, outperforming APE and CacheBlend by 25.2%
and 35.1% on NIAH (with reomp% = 20%). Meanwhile, CacheClip accelerates LLM
inference by up to 1.92x in prefill time, providing a practical solution to the
efficiency-quality trade-off in RAG systems.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为CacheClip的新框架，旨在解决RAG系统中因长输入序列导致的首个令牌生成时间（TTFT）瓶颈问题。通过利用辅助小型LLM识别关键令牌以恢复跨块注意力，并结合选择性KV缓存重计算、共享前缀和分组策略，CacheClip在保证生成质量的同时显著提升了推理效率，实验表明其在多项任务上优于现有方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.10114v1">LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) is widely used to mitigate
hallucinations of Large Language Models (LLMs) by leveraging external
knowledge. While effective for simple queries, traditional RAG systems struggle
with large-scale, unstructured corpora where information is fragmented. Recent
advances incorporate knowledge graphs to capture relational structures,
enabling more comprehensive retrieval for complex, multi-hop reasoning tasks.
However, existing graph-based RAG (GraphRAG) methods rely on unstable and
costly relation extraction for graph construction, often producing noisy graphs
with incorrect or inconsistent relations that degrade retrieval quality. In
this paper, we revisit the pipeline of existing GraphRAG systems and propose
LinearRAG (Linear Graph-based Retrieval-Augmented Generation), an efficient
framework that enables reliable graph construction and precise passage
retrieval. Specifically, LinearRAG constructs a relation-free hierarchical
graph, termed Tri-Graph, using only lightweight entity extraction and semantic
linking, avoiding unstable relation modeling. This new paradigm of graph
construction scales linearly with corpus size and incurs no extra token
consumption, providing an economical and reliable indexing of the original
passages. For retrieval, LinearRAG adopts a two-stage strategy: (i) relevant
entity activation via local semantic bridging, followed by (ii) passage
retrieval through global importance aggregation. Extensive experiments on four
datasets demonstrate that LinearRAG significantly outperforms baseline models.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为LinearRAG的高效框架，通过构建无层级关系的关系无关层次图（Tri-Graph）来解决传统基于知识图谱的RAG（GraphRAG）方法中关系抽取不稳定和高成本的问题，从而实现可靠的图构建和精确的段落检索，显著提升了大规模非结构化语料库上的检索效果。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.10008v1">RIPRAG: Hack a Black-box Retrieval-Augmented Generation Question-Answering System with Reinforcement Learning</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) systems based on Large Language Models
(LLMs) have become a core technology for tasks such as question-answering (QA)
and content generation. However, by injecting poisoned documents into the
database of RAG systems, attackers can manipulate LLMs to generate text that
aligns with their intended preferences. Existing research has primarily focused
on white-box attacks against simplified RAG architectures. In this paper, we
investigate a more complex and realistic scenario: the attacker lacks knowledge
of the RAG system's internal composition and implementation details, and the
RAG system comprises components beyond a mere retriever. Specifically, we
propose the RIPRAG attack framework, an end-to-end attack pipeline that treats
the target RAG system as a black box, where the only information accessible to
the attacker is whether the poisoning succeeds. Our method leverages
Reinforcement Learning (RL) to optimize the generation model for poisoned
documents, ensuring that the generated poisoned document aligns with the target
RAG system's preferences. Experimental results demonstrate that this method can
effectively execute poisoning attacks against most complex RAG systems,
achieving an attack success rate (ASR) improvement of up to 0.72 compared to
baseline methods. This highlights prevalent deficiencies in current defensive
methods and provides critical insights for LLM security research.</details></td><td><details><summary>展开</summary>本文研究了RAG系统在缺乏内部知识情况下的黑盒攻击场景，提出RIPRAG攻击框架，利用强化学习生成优化后的投毒文档以操纵RAG系统输出，实验显示该方法显著提升攻击成功率，揭示了当前防御机制的缺陷。</details></td></tr></tbody></table>

### 📅 2025-10-10
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.09355v1">NL2GenSym: Natural Language to Generative Symbolic Rules for SOAR Cognitive Architecture via Large Language Models</a></td><td><details><summary>展开</summary>SOAR, a classic symbol-based cognitive architecture, has been fostering the
development of general, human-like intelligent agents. Nevertheless, its
practical adoption is hindered by the laborious manual rule coding. Emerging
Large Language Models (LLMs) present the immense potential for efficient rules
generation. However, there is a critical gap that current research
predominantly focuses on conceptual frameworks and lacks robust experimental
validation. To bridge this gap, we propose \textit{N}atural \textit{L}anguage
to \textit{Gen}erative \textit{Sym}bolic Rules (NL2GenSym), a novel framework
that integrates LLMs with SOAR to autonomously produce generative symbolic
rules from natural language. Specifically, our framework introduces a novel
Execution-Grounded Generator-Critic mechanism. The LLM-based Generator, guided
by a Retrieval-Augmented Generation-accessed self-evolving domain knowledge
base, proposes rules from natural language. Subsequently, these rules are
immediately executed within the SOAR environment to rigorously validate their
correctness. Based on this execution-grounded feedback, a reflective LLM-based
Critic drives the iterative refinement of these rules. Experiments on our
specialized Water Jug Problem (WJP) dataset, utilizing both Gemini and Qwen
series models, validate the efficacy of our framework. It achieves a success
rate over 86\% in generating rules from natural language. Crucially, the
framework also generates novel heuristic rules, reducing average decision
cycles for solving the WJP to 1.98 times the optimal solution and 1/1000 of
baseline methods. Additionally, our initial experiments show that NL2GenSym
enables smaller-parameter models to achieve better performance than larger
counterparts.</details></td><td><details><summary>展开</summary>该论文提出了一种名为NL2GenSym的新框架，将大语言模型（LLMs）与SOAR认知架构结合，通过检索增强生成（RAG）访问自演化的领域知识库，从自然语言中自动生成符号规则，并通过执行验证和迭代优化机制提升规则的正确性和效率。实验验证了该框架在生成规则和启发式规则方面的有效性，显著提升了问题解决性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.09266v1">CFVBench: A Comprehensive Video Benchmark for Fine-grained Multimodal Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Multimodal Retrieval-Augmented Generation (MRAG) enables Multimodal Large
Language Models (MLLMs) to generate responses with external multimodal
evidence, and numerous video-based MRAG benchmarks have been proposed to
evaluate model capabilities across retrieval and generation stages. However,
existing benchmarks remain limited in modality coverage and format diversity,
often focusing on single- or limited-modality tasks, or coarse-grained scene
understanding. To address these gaps, we introduce CFVBench, a large-scale,
manually verified benchmark constructed from 599 publicly available videos,
yielding 5,360 open-ended QA pairs. CFVBench spans high-density formats and
domains such as chart-heavy reports, news broadcasts, and software tutorials,
requiring models to retrieve and reason over long temporal video spans while
maintaining fine-grained multimodal information. Using CFVBench, we
systematically evaluate 7 retrieval methods and 14 widely-used MLLMs, revealing
a critical bottleneck: current models (even GPT5 or Gemini) struggle to capture
transient yet essential fine-grained multimodal details. To mitigate this, we
propose Adaptive Visual Refinement (AVR), a simple yet effective framework that
adaptively increases frame sampling density and selectively invokes external
tools when necessary. Experiments show that AVR consistently enhances
fine-grained multimodal comprehension and improves performance across all
evaluated MLLMs</details></td><td><details><summary>展开</summary>这篇论文提出了一个名为CFVBench的多模态检索增强生成（MRAG）基准测试，用于评估多模态大语言模型（MLLMs）在检索和生成阶段的能力。该基准基于599个公开视频构建，包含5,360个开放式问答对，覆盖高密度格式和多样领域，要求模型在长时视频中检索并推理细粒度多模态信息。研究发现当前模型在捕捉关键细节方面存在瓶颈，并提出了自适应视觉优化（AVR）框架以增强细粒度多模态理解，实验证明AVR能有效提升模型性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.09156v1">Agentic-KGR: Co-evolutionary Knowledge Graph Construction through Multi-Agent Reinforcement Learning</a></td><td><details><summary>展开</summary>Current knowledge-enhanced large language models (LLMs) rely on static,
pre-constructed knowledge bases that suffer from coverage gaps and temporal
obsolescence, limiting their effectiveness in dynamic information environments.
We present Agentic-KGR, a novel framework enabling co-evolution between LLMs
and knowledge graphs (KGs) through multi-round reinforcement learning (RL). Our
approach introduces three key innovations: (1) a dynamic schema expansion
mechanism that systematically extends graph ontologies beyond pre-defined
boundaries during training; (2) a retrieval-augmented memory system enabling
synergistic co-evolution between model parameters and knowledge structures
through continuous optimization; (3) a learnable multi-scale prompt compression
approach that preserves critical information while reducing computational
complexity through adaptive sequence optimization. Experimental results
demonstrate substantial improvements over supervised baselines and single-round
RL approaches in knowledge extraction tasks. When integrated with GraphRAG, our
method achieves superior performance in downstream QA tasks, with significant
gains in both accuracy and knowledge coverage compared to existing methods.</details></td><td><details><summary>展开</summary>该论文提出了一种名为Agentic-KGR的新框架，通过多轮强化学习实现大语言模型与知识图谱的协同进化，包含动态模式扩展、检索增强记忆系统和可学习的多尺度提示压缩等创新，显著提升了知识提取和问答任务的性能，并与GraphRAG结合展示了优越性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.09106v1">When Retrieval Succeeds and Fails: Rethinking Retrieval-Augmented Generation for LLMs</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) have enabled a wide range of applications
through their powerful capabilities in language understanding and generation.
However, as LLMs are trained on static corpora, they face difficulties in
addressing rapidly evolving information or domain-specific queries.
Retrieval-Augmented Generation (RAG) was developed to overcome this limitation
by integrating LLMs with external retrieval mechanisms, allowing them to access
up-to-date and contextually relevant knowledge. However, as LLMs themselves
continue to advance in scale and capability, the relative advantages of
traditional RAG frameworks have become less pronounced and necessary. Here, we
present a comprehensive review of RAG, beginning with its overarching
objectives and core components. We then analyze the key challenges within RAG,
highlighting critical weakness that may limit its effectiveness. Finally, we
showcase applications where LLMs alone perform inadequately, but where RAG,
when combined with LLMs, can substantially enhance their effectiveness. We hope
this work will encourage researchers to reconsider the role of RAG and inspire
the development of next-generation RAG systems.</details></td><td><details><summary>展开</summary>这篇论文是一篇关于检索增强生成（RAG）的综述性文章，探讨了RAG如何通过结合外部检索机制弥补大语言模型（LLMs）在动态信息和领域特定查询上的不足，分析了RAG的核心组件、关键挑战及其局限性，并展示了RAG与LLMs结合的应用场景，旨在推动下一代RAG系统的发展。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.09093v1">Exploiting Web Search Tools of AI Agents for Data Exfiltration</a></td><td><details><summary>展开</summary>Large language models (LLMs) are now routinely used to autonomously execute
complex tasks, from natural language processing to dynamic workflows like web
searches. The usage of tool-calling and Retrieval Augmented Generation (RAG)
allows LLMs to process and retrieve sensitive corporate data, amplifying both
their functionality and vulnerability to abuse. As LLMs increasingly interact
with external data sources, indirect prompt injection emerges as a critical and
evolving attack vector, enabling adversaries to exploit models through
manipulated inputs. Through a systematic evaluation of indirect prompt
injection attacks across diverse models, we analyze how susceptible current
LLMs are to such attacks, which parameters, including model size and
manufacturer, specific implementations, shape their vulnerability, and which
attack methods remain most effective. Our results reveal that even well-known
attack patterns continue to succeed, exposing persistent weaknesses in model
defenses. To address these vulnerabilities, we emphasize the need for
strengthened training procedures to enhance inherent resilience, a centralized
database of known attack vectors to enable proactive defense, and a unified
testing framework to ensure continuous security validation. These steps are
essential to push developers toward integrating security into the core design
of LLMs, as our findings show that current models still fail to mitigate
long-standing threats.</details></td><td><details><summary>展开</summary>这篇论文探讨了大型语言模型（LLMs）在结合工具调用和检索增强生成（RAG）技术处理敏感数据时面临的间接提示注入攻击风险，通过系统评估不同模型的脆弱性、影响因素及攻击方法，揭示了当前防御的不足，并提出了加强训练、建立攻击向量数据库和统一测试框架等安全改进措施。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.08981v1">SEER: Sustainability Enhanced Engineering of Software Requirements</a></td><td><details><summary>展开</summary>The rapid expansion of software development has significant environmental,
technical, social, and economic impacts. Achieving the United Nations
Sustainable Development Goals by 2030 compels developers to adopt sustainable
practices. Existing methods mostly offer high-level guidelines, which are
time-consuming to implement and rely on team adaptability. Moreover, they focus
on design or implementation, while sustainability assessment should start at
the requirements engineering phase. In this paper, we introduce SEER, a
framework which addresses sustainability concerns in the early software
development phase. The framework operates in three stages: (i) it identifies
sustainability requirements (SRs) relevant to a specific software product from
a general taxonomy; (ii) it evaluates how sustainable system requirements are
based on the identified SRs; and (iii) it optimizes system requirements that
fail to satisfy any SR. The framework is implemented using the reasoning
capabilities of large language models and the agentic RAG (Retrieval Augmented
Generation) approach. SEER has been experimented on four software projects from
different domains. Results generated using Gemini 2.5 reasoning model
demonstrate the effectiveness of the proposed approach in accurately
identifying a broad range of sustainability concerns across diverse domains.</details></td><td><details><summary>展开</summary>这篇论文介绍了SEER框架，该框架利用大语言模型的推理能力和RAG技术，在软件开发早期阶段识别、评估和优化可持续性需求，以应对不同领域的可持续性问题。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.08976v1">Hierarchical Scheduling for Multi-Vector Image Retrieval</a></td><td><details><summary>展开</summary>To effectively leverage user-specific data, retrieval augmented generation
(RAG) is employed in multimodal large language model (MLLM) applications.
However, conventional retrieval approaches often suffer from limited retrieval
accuracy. Recent advances in multi-vector retrieval (MVR) improve accuracy by
decomposing queries and matching against segmented images. They still suffer
from sub-optimal accuracy and efficiency, overlooking alignment between the
query and varying image objects and redundant fine-grained image segments. In
this work, we present an efficient scheduling framework for image retrieval -
HiMIR. First, we introduce a novel hierarchical paradigm, employing multiple
intermediate granularities for varying image objects to enhance alignment.
Second, we minimize redundancy in retrieval by leveraging cross-hierarchy
similarity consistency and hierarchy sparsity to minimize unnecessary matching
computation. Furthermore, we configure parameters for each dataset
automatically for practicality across diverse scenarios. Our empirical study
shows that, HiMIR not only achieves substantial accuracy improvements but also
reduces computation by up to 3.5 times over the existing MVR system.</details></td><td><details><summary>展开</summary>该论文提出了一种名为HiMIR的高效图像检索调度框架，针对多模态大语言模型（MLLM）中检索增强生成（RAG）的局限性进行优化。通过分层多粒度对齐策略减少冗余计算，提升检索准确性和效率，实验显示其性能优于现有多向量检索（MVR）系统。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.08958v1">EcphoryRAG: Re-Imagining Knowledge-Graph RAG via Human Associative Memory</a></td><td><details><summary>展开</summary>Cognitive neuroscience research indicates that humans leverage cues to
activate entity-centered memory traces (engrams) for complex, multi-hop
recollection. Inspired by this mechanism, we introduce EcphoryRAG, an
entity-centric knowledge graph RAG framework. During indexing, EcphoryRAG
extracts and stores only core entities with corresponding metadata, a
lightweight approach that reduces token consumption by up to 94\% compared to
other structured RAG systems. For retrieval, the system first extracts cue
entities from queries, then performs a scalable multi-hop associative search
across the knowledge graph. Crucially, EcphoryRAG dynamically infers implicit
relations between entities to populate context, enabling deep reasoning without
exhaustive pre-enumeration of relationships. Extensive evaluations on the
2WikiMultiHop, HotpotQA, and MuSiQue benchmarks demonstrate that EcphoryRAG
sets a new state-of-the-art, improving the average Exact Match (EM) score from
0.392 to 0.474 over strong KG-RAG methods like HippoRAG. These results validate
the efficacy of the entity-cue-multi-hop retrieval paradigm for complex
question answering.</details></td><td><details><summary>展开</summary>这篇论文提出了EcphoryRAG，一种基于实体中心知识图谱的RAG框架，通过提取和存储核心实体及元数据减少token消耗，并利用多跳关联检索和动态关系推理提升复杂问答性能，在多个基准测试中表现优于现有方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.08945v1">FATHOMS-RAG: A Framework for the Assessment of Thinking and Observation in Multimodal Systems that use Retrieval Augmented Generation</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) has emerged as a promising paradigm for
improving factual accuracy in large language models (LLMs). We introduce a
benchmark designed to evaluate RAG pipelines as a whole, evaluating a
pipeline's ability to ingest, retrieve, and reason about several modalities of
information, differentiating it from existing benchmarks that focus on
particular aspects such as retrieval. We present (1) a small, human-created
dataset of 93 questions designed to evaluate a pipeline's ability to ingest
textual data, tables, images, and data spread across these modalities in one or
more documents; (2) a phrase-level recall metric for correctness; (3) a
nearest-neighbor embedding classifier to identify potential pipeline
hallucinations; (4) a comparative evaluation of 2 pipelines built with
open-source retrieval mechanisms and 4 closed-source foundation models; and (5)
a third-party human evaluation of the alignment of our correctness and
hallucination metrics. We find that closed-source pipelines significantly
outperform open-source pipelines in both correctness and hallucination metrics,
with wider performance gaps in questions relying on multimodal and
cross-document information. Human evaluation of our metrics showed average
agreement of 4.62 for correctness and 4.53 for hallucination detection on a 1-5
Likert scale (5 indicating "strongly agree").</details></td><td><details><summary>展开</summary>这篇论文介绍了一个用于评估多模态RAG（检索增强生成）管道的综合基准测试，包括创建包含文本、表格和图像的多模态数据集、提出新的评估指标（短语级准确率和最近邻嵌入分类器用于检测幻觉），并对开源与闭源RAG管道进行性能比较，结果显示闭源模型在多模态和跨文档任务上表现更优。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.08935v1">Personalize Before Retrieve: LLM-based Personalized Query Expansion for User-Centric Retrieval</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) critically depends on effective query
expansion to retrieve relevant information. However, existing expansion methods
adopt uniform strategies that overlook user-specific semantics, ignoring
individual expression styles, preferences, and historical context. In practice,
identical queries in text can express vastly different intentions across users.
This representational rigidity limits the ability of current RAG systems to
generalize effectively in personalized settings. Specifically, we identify two
core challenges for personalization: 1) user expression styles are inherently
diverse, making it difficult for standard expansions to preserve personalized
intent. 2) user corpora induce heterogeneous semantic structures-varying in
topical focus and lexical organization-which hinders the effective anchoring of
expanded queries within the user's corpora space. To address these challenges,
we propose Personalize Before Retrieve (PBR), a framework that incorporates
user-specific signals into query expansion prior to retrieval. PBR consists of
two components: P-PRF, which generates stylistically aligned pseudo feedback
using user history for simulating user expression style, and P-Anchor, which
performs graph-based structure alignment over user corpora to capture its
structure. Together, they produce personalized query representations tailored
for retrieval. Experiments on two personalized benchmarks show that PBR
consistently outperforms strong baselines, with up to 10% gains on PersonaBench
across retrievers. Our findings demonstrate the value of modeling
personalization before retrieval to close the semantic gap in user-adaptive RAG
systems. Our code is available at https://github.com/Zhang-Yingyi/PBR-code.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为PBR（Personalize Before Retrieve）的框架，旨在解决RAG系统中查询扩展忽略用户个性化语义（如表达风格、偏好和历史上下文）的问题。PBR通过整合用户特定信号（P-PRF模拟用户表达风格，P-Anchor对齐用户语料结构）生成个性化查询表示，实验表明其在个性化基准测试中性能显著优于基线方法，提升了10%的检索效果。</details></td></tr></tbody></table>

### 📅 2025-10-09
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.08383v1">QAgent: A modular Search Agent with Interactive Query Understanding</a></td><td><details><summary>展开</summary>Large language models (LLMs) excel at natural language tasks but are limited
by their static parametric knowledge, especially in knowledge-intensive task.
Retrieval-augmented generation (RAG) mitigates this by integrating external
information. However, (1) traditional RAG struggles with complex query
understanding, and (2) even search agents trained with reinforcement learning
(RL), despite their promise, still face generalization and deployment
challenges. To address these limitations, we propose QAgent, a unified agentic
RAG framework that employs a search agent for adaptive retrieval. This agent
optimizes its understanding of the query through interactive reasoning and
retrieval. To facilitate real-world application, we focus on modular search
agent for query understanding that are plug-and-play in complex systems.
Secifically, the agent follows a multi-step decision process trained with RL to
maximize retrieval quality and support accurate downstream answers. We further
analyze the strengths and weaknesses of end-to-end RL and propose a strategy
that focuses on effective retrieval, thereby enhancing generalization in LLM
applications. Experiments show QAgent excels at QA and serves as a
plug-and-play module for real-world deployment.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为QAgent的新型检索增强生成（RAG）框架，通过强化学习训练搜索代理以优化复杂查询的理解和自适应检索，解决传统RAG在查询理解和泛化能力上的不足，并实现即插即用的模块化部署，实验表明其在问答任务中表现优异。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.08149v1">AI Knowledge Assist: An Automated Approach for the Creation of Knowledge Bases for Conversational AI Agents</a></td><td><details><summary>展开</summary>The utilization of conversational AI systems by leveraging Retrieval
Augmented Generation (RAG) techniques to solve customer problems has been on
the rise with the rapid progress of Large Language Models (LLMs). However, the
absence of a company-specific dedicated knowledge base is a major barrier to
the integration of conversational AI systems in contact centers. To this end,
we introduce AI Knowledge Assist, a system that extracts knowledge in the form
of question-answer (QA) pairs from historical customer-agent conversations to
automatically build a knowledge base. Fine-tuning a lightweight LLM on internal
data demonstrates state-of-the-art performance, outperforming larger
closed-source LLMs. More specifically, empirical evaluation on 20 companies
demonstrates that the proposed AI Knowledge Assist system that leverages the
LLaMA-3.1-8B model eliminates the cold-start gap in contact centers by
achieving above 90% accuracy in answering information-seeking questions. This
enables immediate deployment of RAG-powered chatbots.</details></td><td><details><summary>展开</summary>该论文介绍了AI Knowledge Assist系统，通过从历史客户-代理对话中提取问答对自动构建知识库，并利用轻量级LLM（如LLaMA-3.1-8B）微调内部数据，在解决联系中心冷启动问题中实现超过90%的准确率，支持基于RAG技术的聊天机器人快速部署。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.08109v1">VersionRAG: Version-Aware Retrieval-Augmented Generation for Evolving Documents</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) systems fail when documents evolve
through versioning-a ubiquitous characteristic of technical documentation.
Existing approaches achieve only 58-64% accuracy on version-sensitive
questions, retrieving semantically similar content without temporal validity
checks. We present VersionRAG, a version-aware RAG framework that explicitly
models document evolution through a hierarchical graph structure capturing
version sequences, content boundaries, and changes between document states.
During retrieval, VersionRAG routes queries through specialized paths based on
intent classification, enabling precise version-aware filtering and change
tracking. On our VersionQA benchmark-100 manually curated questions across 34
versioned technical documents-VersionRAG achieves 90% accuracy, outperforming
naive RAG (58%) and GraphRAG (64%). VersionRAG reaches 60% accuracy on implicit
change detection where baselines fail (0-10%), demonstrating its ability to
track undocumented modifications. Additionally, VersionRAG requires 97% fewer
tokens during indexing than GraphRAG, making it practical for large-scale
deployment. Our work establishes versioned document QA as a distinct task and
provides both a solution and benchmark for future research.</details></td><td><details><summary>展开</summary>这篇论文提出了VersionRAG，一个针对版本化文档的检索增强生成框架，通过分层图结构显式建模文档演变过程，解决了传统RAG在文档版本更新时准确性不足的问题。VersionRAG在版本敏感问题上达到90%的准确率，显著优于基线方法，并大幅降低索引开销，为版本化文档问答任务提供了解决方案和基准。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.07925v1">Enabling Personalized Long-term Interactions in LLM-based Agents through Persistent Memory and User Profiles</a></td><td><details><summary>展开</summary>Large language models (LLMs) increasingly serve as the central control unit
of AI agents, yet current approaches remain limited in their ability to deliver
personalized interactions. While Retrieval Augmented Generation enhances LLM
capabilities by improving context-awareness, it lacks mechanisms to combine
contextual information with user-specific data. Although personalization has
been studied in fields such as human-computer interaction or cognitive science,
existing perspectives largely remain conceptual, with limited focus on
technical implementation. To address these gaps, we build on a unified
definition of personalization as a conceptual foundation to derive technical
requirements for adaptive, user-centered LLM-based agents. Combined with
established agentic AI patterns such as multi-agent collaboration or
multi-source retrieval, we present a framework that integrates persistent
memory, dynamic coordination, self-validation, and evolving user profiles to
enable personalized long-term interactions. We evaluate our approach on three
public datasets using metrics such as retrieval accuracy, response correctness,
or BertScore. We complement these results with a five-day pilot user study
providing initial insights into user feedback on perceived personalization. The
study provides early indications that guide future work and highlights the
potential of integrating persistent memory and user profiles to improve the
adaptivity and perceived personalization of LLM-based agents.</details></td><td><details><summary>展开</summary>这篇论文探讨了如何通过结合检索增强生成（RAG）与用户特定数据，提升基于大语言模型（LLM）的AI代理的个性化交互能力。作者提出了一个框架，整合持久记忆、动态协调、自我验证和动态用户画像等技术，以实现长期个性化互动，并通过实验和用户研究验证了其效果。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.07923v1">STEPER: Step-wise Knowledge Distillation for Enhancing Reasoning Ability in Multi-Step Retrieval-Augmented Language Models</a></td><td><details><summary>展开</summary>Answering complex real-world questions requires step-by-step retrieval and
integration of relevant information to generate well-grounded responses.
However, existing knowledge distillation methods overlook the need for
different reasoning abilities at different steps, hindering transfer in
multi-step retrieval-augmented frameworks. To address this, we propose Stepwise
Knowledge Distillation for Enhancing Reasoning Ability in Multi-Step
Retrieval-Augmented Language Models (StepER). StepER employs step-wise
supervision to align with evolving information and reasoning demands across
stages. Additionally, it incorporates difficulty-aware training to
progressively optimize learning by prioritizing suitable steps. Our method is
adaptable to various multi-step retrieval-augmented language models, including
those that use retrieval queries for reasoning paths or decomposed questions.
Extensive experiments show that StepER outperforms prior methods on multi-hop
QA benchmarks, with an 8B model achieving performance comparable to a 70B
teacher model.</details></td><td><details><summary>展开</summary>该论文提出了一种名为StepER的逐步知识蒸馏方法，旨在提升多步检索增强语言模型（RAG）的推理能力，通过分步监督和难度感知训练优化不同步骤的信息整合与推理需求，实验表明其在多跳问答任务中显著优于现有方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.07920v1">Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents</a></td><td><details><summary>展开</summary>LLM-based financial agents have attracted widespread excitement for their
ability to trade like human experts. However, most systems exhibit a "profit
mirage": dazzling back-tested returns evaporate once the model's knowledge
window ends, because of the inherent information leakage in LLMs. In this
paper, we systematically quantify this leakage issue across four dimensions and
release FinLake-Bench, a leakage-robust evaluation benchmark. Furthermore, to
mitigate this issue, we introduce FactFin, a framework that applies
counterfactual perturbations to compel LLM-based agents to learn causal drivers
instead of memorized outcomes. FactFin integrates four core components:
Strategy Code Generator, Retrieval-Augmented Generation, Monte Carlo Tree
Search, and Counterfactual Simulator. Extensive experiments show that our
method surpasses all baselines in out-of-sample generalization, delivering
superior risk-adjusted performance.</details></td><td><details><summary>展开</summary>这篇论文探讨了基于LLM的金融代理因信息泄露导致的"利润幻象"问题，提出了泄漏鲁棒性基准FinLake-Bench和解决方案框架FactFin。FactFin通过反事实扰动使模型学习因果驱动而非记忆结果，其核心包含检索增强生成（RAG）等组件，实验表明该方法在样本外泛化中优于基线并提升风险调整后表现。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.07794v1">HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation</a></td><td><details><summary>展开</summary>Agentic RAG is a powerful technique for incorporating external information
that LLMs lack, enabling better problem solving and question answering.
However, suboptimal search behaviors exist widely, such as over-search
(retrieving information already known) and under-search (failing to search when
necessary), which leads to unnecessary overhead and unreliable outputs. Current
training methods, which typically rely on outcome-based rewards in a RL
framework, lack the fine-grained control needed to address these
inefficiencies. To overcome this, we introduce Hierarchical Process Rewards for
Efficient agentic RAG (HiPRAG), a training methodology that incorporates a
fine-grained, knowledge-grounded process reward into the RL training. Our
approach evaluates the necessity of each search decision on-the-fly by
decomposing the agent's reasoning trajectory into discrete, parsable steps. We
then apply a hierarchical reward function that provides an additional bonus
based on the proportion of optimal search and non-search steps, on top of
commonly used outcome and format rewards. Experiments on the Qwen2.5 and
Llama-3.2 models across seven diverse QA benchmarks show that our method
achieves average accuracies of 65.4% (3B) and 67.2% (7B). This is accomplished
while improving search efficiency, reducing the over-search rate to just 2.3%
and concurrently lowering the under-search rate. These results demonstrate the
efficacy of optimizing the reasoning process itself, not just the final
outcome. Further experiments and analysis demonstrate that HiPRAG shows good
generalizability across a wide range of RL algorithms, model families, sizes,
and types. This work demonstrates the importance and potential of fine-grained
control through RL, for improving the efficiency and optimality of reasoning
for search agents.</details></td><td><details><summary>展开</summary>本文提出了一种名为HiPRAG的训练方法，通过分层过程奖励优化RAG中的搜索行为，减少过度搜索和不足搜索问题，提高搜索效率和回答准确性，并在多个QA基准测试中验证了其有效性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.07748v1">Haibu Mathematical-Medical Intelligent Agent:Enhancing Large Language Model Reliability in Medical Tasks via Verifiable Reasoning Chains</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) show promise in medicine but are prone to
factual and logical errors, which is unacceptable in this high-stakes field. To
address this, we introduce the "Haibu Mathematical-Medical Intelligent Agent"
(MMIA), an LLM-driven architecture that ensures reliability through a formally
verifiable reasoning process. MMIA recursively breaks down complex medical
tasks into atomic, evidence-based steps. This entire reasoning chain is then
automatically audited for logical coherence and evidence traceability, similar
to theorem proving. A key innovation is MMIA's "bootstrapping" mode, which
stores validated reasoning chains as "theorems." Subsequent tasks can then be
efficiently solved using Retrieval-Augmented Generation (RAG), shifting from
costly first-principles reasoning to a low-cost verification model. We
validated MMIA across four healthcare administration domains, including DRG/DIP
audits and medical insurance adjudication, using expert-validated benchmarks.
Results showed MMIA achieved an error detection rate exceeding 98% with a false
positive rate below 1%, significantly outperforming baseline LLMs. Furthermore,
the RAG matching mode is projected to reduce average processing costs by
approximately 85% as the knowledge base matures. In conclusion, MMIA's
verifiable reasoning framework is a significant step toward creating
trustworthy, transparent, and cost-effective AI systems, making LLM technology
viable for critical applications in medicine.</details></td><td><details><summary>展开</summary>该论文介绍了“Haibu Mathematical-Medical Intelligent Agent (MMIA)”，一种基于大语言模型（LLM）的架构，通过可验证的推理过程确保医学任务的可靠性。MMIA将复杂任务分解为基于证据的原子步骤，并利用检索增强生成（RAG）技术存储已验证的推理链作为“定理”，从而降低处理成本。实验表明，MMIA在医疗管理领域显著优于基线LLM，错误检测率达98%以上，且RAG模式预计可降低85%的成本。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.07728v1">Who Stole Your Data? A Method for Detecting Unauthorized RAG Theft</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) enhances Large Language Models (LLMs) by
mitigating hallucinations and outdated information issues, yet simultaneously
facilitates unauthorized data appropriation at scale. This paper addresses this
challenge through two key contributions. First, we introduce RPD, a novel
dataset specifically designed for RAG plagiarism detection that encompasses
diverse professional domains and writing styles, overcoming limitations in
existing resources. Second, we develop a dual-layered watermarking system that
embeds protection at both semantic and lexical levels, complemented by an
interrogator-detective framework that employs statistical hypothesis testing on
accumulated evidence. Extensive experimentation demonstrates our approach's
effectiveness across varying query volumes, defense prompts, and retrieval
parameters, while maintaining resilience against adversarial evasion
techniques. This work establishes a foundational framework for intellectual
property protection in retrieval-augmented AI systems.</details></td><td><details><summary>展开</summary>这篇论文针对RAG技术可能导致的未经授权数据滥用问题，提出了两个解决方案：一是构建专门用于检测RAG抄袭的多领域数据集RPD，二是设计一种结合语义和词汇双层次水印的保护系统，并通过统计假设检验框架验证其有效性，旨在保护检索增强AI系统中的知识产权。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.07718v1">SUBQRAG: sub-question driven dynamic graph rag</a></td><td><details><summary>展开</summary>Graph Retrieval-Augmented Generation (Graph RAG) effectively builds a
knowledge graph (KG) to connect disparate facts across a large document corpus.
However, this broad-view approach often lacks the deep structured reasoning
needed for complex multi-hop question answering (QA), leading to incomplete
evidence and error accumulation. To address these limitations, we propose
SubQRAG, a sub-question-driven framework that enhances reasoning depth. SubQRAG
decomposes a complex question into an ordered chain of verifiable
sub-questions. For each sub-question, it retrieves relevant triples from the
graph. When the existing graph is insufficient, the system dynamically expands
it by extracting new triples from source documents in real time. All triples
used in the reasoning process are aggregated into a "graph memory," forming a
structured and traceable evidence path for final answer generation. Experiments
on three multi-hop QA benchmarks demonstrate that SubQRAG achieves consistent
and significant improvements, especially in Exact Match scores.</details></td><td><details><summary>展开</summary>该论文提出了一种名为SubQRAG的子问题驱动的图检索增强生成框架，通过将复杂问题分解为可验证的子问题链，动态检索和扩展知识图谱中的三元组，并构建可追溯的"图记忆"路径，显著提升了多跳问答任务的性能。</details></td></tr></tbody></table>

### 📅 2025-10-08
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.07233v1">LAD-RAG: Layout-aware Dynamic RAG for Visually-Rich Document Understanding</a></td><td><details><summary>展开</summary>Question answering over visually rich documents (VRDs) requires reasoning not
only over isolated content but also over documents' structural organization and
cross-page dependencies. However, conventional retrieval-augmented generation
(RAG) methods encode content in isolated chunks during ingestion, losing
structural and cross-page dependencies, and retrieve a fixed number of pages at
inference, regardless of the specific demands of the question or context. This
often results in incomplete evidence retrieval and degraded answer quality for
multi-page reasoning tasks. To address these limitations, we propose LAD-RAG, a
novel Layout-Aware Dynamic RAG framework. During ingestion, LAD-RAG constructs
a symbolic document graph that captures layout structure and cross-page
dependencies, adding it alongside standard neural embeddings to yield a more
holistic representation of the document. During inference, an LLM agent
dynamically interacts with the neural and symbolic indices to adaptively
retrieve the necessary evidence based on the query. Experiments on
MMLongBench-Doc, LongDocURL, DUDE, and MP-DocVQA demonstrate that LAD-RAG
improves retrieval, achieving over 90% perfect recall on average without any
top-k tuning, and outperforming baseline retrievers by up to 20% in recall at
comparable noise levels, yielding higher QA accuracy with minimal latency.</details></td><td><details><summary>展开</summary>本文提出了一种名为LAD-RAG的新型布局感知动态RAG框架，旨在解决传统RAG方法在处理视觉丰富文档（VRDs）时因忽略文档结构和跨页依赖关系而导致的证据不完整和答案质量下降问题。LAD-RAG通过构建符号文档图来捕捉布局结构和跨页依赖，并结合神经嵌入技术，在推理阶段由LLM智能体动态交互以自适应检索证据，实验表明其在多个数据集上显著提升了检索效果和问答准确性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.07096v1">Making Machines Sound Sarcastic: LLM-Enhanced and Retrieval-Guided Sarcastic Speech Synthesis</a></td><td><details><summary>展开</summary>Sarcasm is a subtle form of non-literal language that poses significant
challenges for speech synthesis due to its reliance on nuanced semantic,
contextual, and prosodic cues. While existing speech synthesis research has
focused primarily on broad emotional categories, sarcasm remains largely
unexplored. In this paper, we propose a Large Language Model (LLM)-enhanced
Retrieval-Augmented framework for sarcasm-aware speech synthesis. Our approach
combines (1) semantic embeddings from a LoRA-fine-tuned LLaMA 3, which capture
pragmatic incongruity and discourse-level cues of sarcasm, and (2) prosodic
exemplars retrieved via a Retrieval Augmented Generation (RAG) module, which
provide expressive reference patterns of sarcastic delivery. Integrated within
a VITS backbone, this dual conditioning enables more natural and contextually
appropriate sarcastic speech. Experiments demonstrate that our method
outperforms baselines in both objective measures and subjective evaluations,
yielding improvements in speech naturalness, sarcastic expressivity, and
downstream sarcasm detection.</details></td><td><details><summary>展开</summary>该论文提出了一种基于LLM增强的检索增强框架，用于具有讽刺感知的语音合成，结合了微调LLaMA 3的语义嵌入和通过RAG模块检索的韵律范例，以生成更自然和符合上下文的讽刺语音。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.06999v1">Towards Reliable Retrieval in RAG Systems for Large Legal Datasets</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) is a promising approach to mitigate
hallucinations in Large Language Models (LLMs) for legal applications, but its
reliability is critically dependent on the accuracy of the retrieval step. This
is particularly challenging in the legal domain, where large databases of
structurally similar documents often cause retrieval systems to fail. In this
paper, we address this challenge by first identifying and quantifying a
critical failure mode we term Document-Level Retrieval Mismatch (DRM), where
the retriever selects information from entirely incorrect source documents. To
mitigate DRM, we investigate a simple and computationally efficient technique
which we refer to as Summary-Augmented Chunking (SAC). This method enhances
each text chunk with a document-level synthetic summary, thereby injecting
crucial global context that would otherwise be lost during a standard chunking
process. Our experiments on a diverse set of legal information retrieval tasks
show that SAC greatly reduces DRM and, consequently, also improves text-level
retrieval precision and recall. Interestingly, we find that a generic
summarization strategy outperforms an approach that incorporates legal expert
domain knowledge to target specific legal elements. Our work provides evidence
that this practical, scalable, and easily integrable technique enhances the
reliability of RAG systems when applied to large-scale legal document datasets.</details></td><td><details><summary>展开</summary>这篇论文针对法律领域中检索增强生成（RAG）系统的检索步骤准确性不足问题，提出了一种名为“摘要增强分块”（SAC）的方法，通过为文本块添加文档级合成摘要来减少文档级检索不匹配（DRM），从而提升检索精度和召回率，增强RAG系统在法律文档数据集上的可靠性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.06888v1">M3Retrieve: Benchmarking Multimodal Retrieval for Medicine</a></td><td><details><summary>展开</summary>With the increasing use of RetrievalAugmented Generation (RAG), strong
retrieval models have become more important than ever. In healthcare,
multimodal retrieval models that combine information from both text and images
offer major advantages for many downstream tasks such as question answering,
cross-modal retrieval, and multimodal summarization, since medical data often
includes both formats. However, there is currently no standard benchmark to
evaluate how well these models perform in medical settings. To address this
gap, we introduce M3Retrieve, a Multimodal Medical Retrieval Benchmark.
M3Retrieve, spans 5 domains,16 medical fields, and 4 distinct tasks, with over
1.2 Million text documents and 164K multimodal queries, all collected under
approved licenses. We evaluate leading multimodal retrieval models on this
benchmark to explore the challenges specific to different medical specialities
and to understand their impact on retrieval performance. By releasing
M3Retrieve, we aim to enable systematic evaluation, foster model innovation,
and accelerate research toward building more capable and reliable multimodal
retrieval systems for medical applications. The dataset and the baselines code
are available in this github page https://github.com/AkashGhosh/M3Retrieve.</details></td><td><details><summary>展开</summary>这篇论文介绍了M3Retrieve，一个多模态医学检索基准，旨在评估结合文本和图像的检索模型在医疗领域的性能，以支持RAG等下游任务，并促进医疗应用中更可靠的多模态检索系统的研究。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.06719v1">Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by
grounding them in external knowledge. However, its application in sensitive
domains is limited by privacy risks. Existing private RAG methods typically
rely on query-time differential privacy (DP), which requires repeated noise
injection and leads to accumulated privacy loss. To address this issue, we
propose DP-SynRAG, a framework that uses LLMs to generate differentially
private synthetic RAG databases. Unlike prior methods, the synthetic text can
be reused once created, thereby avoiding repeated noise injection and
additional privacy costs. To preserve essential information for downstream RAG
tasks, DP-SynRAG extends private prediction, which instructs LLMs to generate
text that mimics subsampled database records in a DP manner. Experiments show
that DP-SynRAG achieves superior performanec to the state-of-the-art private
RAG systems while maintaining a fixed privacy budget, offering a scalable
solution for privacy-preserving RAG.</details></td><td><details><summary>展开</summary>该论文提出了一种名为DP-SynRAG的隐私保护框架，通过生成差分隐私的合成RAG数据库来解决传统RAG在敏感领域应用时的隐私风险问题，避免了重复噪声注入和隐私损失累积，实验表明其在固定隐私预算下性能优于现有隐私RAG系统。</details></td></tr></tbody></table>

### 📅 2025-10-07
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.06002v1">Deterministic Legal Retrieval: An Action API for Querying the SAT-Graph RAG</a></td><td><details><summary>展开</summary>The Structure-Aware Temporal Graph RAG (SAT-Graph RAG) addresses core
limitations of standard Retrieval-Augmented Generation in the legal domain by
providing a verifiable knowledge graph that models hierarchical structure,
temporal evolution, and causal events of legal norms. However, a critical gap
remains: how to reliably query this structured knowledge without sacrificing
its deterministic properties. This paper introduces the SAT-Graph API, a formal
query execution layer centered on canonical actions-atomic, composable, and
auditable primitives that isolate probabilistic discovery from deterministic
retrieval. These actions enable: (i) high-precision hybrid search; (ii) robust
reference resolution; (iii) point-in-time version retrieval; and (iv) auditable
causal tracing. We demonstrate how planner-guided agents can decompose complex
queries into Directed Acyclic Graphs (DAGs) of these actions. This two-layer
architecture transforms retrieval from an opaque black box to a transparent,
auditable process, directly addressing Explainable AI (XAI) requirements for
high-stakes domains.</details></td><td><details><summary>展开</summary>这篇论文提出了SAT-Graph RAG，一种改进标准RAG在司法领域应用的技术，通过构建可验证的知识图谱来建模法律规范的结构、时间和因果关系。为了解决如何在不牺牲其确定性属性的前提下可靠查询结构化知识的问题，论文引入了SAT-Graph API，一个基于规范化操作的查询执行层，支持高精度混合搜索、版本检索和可审计的因果追踪等功能，并通过双层架构增强了检索过程的透明度和可解释性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.05691v1">DecEx-RAG: Boosting Agentic Retrieval-Augmented Generation with Decision and Execution Optimization via Process Supervision</a></td><td><details><summary>展开</summary>Agentic Retrieval-Augmented Generation (Agentic RAG) enhances the processing
capability for complex tasks through dynamic retrieval and adaptive workflows.
Recent advances (e.g., Search-R1) have shown that outcome-supervised
reinforcement learning demonstrate strong performance. However, this approach
still suffers from inefficient exploration, sparse reward signals, and
ambiguous global reward feedback. To address these challenges, we propose
DecEx-RAG, which models RAG as a Markov Decision Process (MDP) incorporating
decision-making and execution, while introducing an efficient pruning strategy
to optimize data expansion. Through comprehensive process-level policy
optimization, DecEx-RAG significantly enhances the autonomous task
decomposition, dynamic retrieval, and high-quality answer generation
capabilities of large language models (LLMs). Experiments show that DecEx-RAG
achieves an average absolute performance improvement of $6.2\%$ across six
datasets, significantly outperforming existing baselines. Moreover, the pruning
strategy improves data construction efficiency by nearly $6 \times$, providing
an efficient solution for process-supervised RAG training. The code is
available at https://github.com/sdsxdxl/DecEx-RAG.</details></td><td><details><summary>展开</summary>该论文提出了一种名为DecEx-RAG的改进方法，通过将RAG建模为马尔可夫决策过程（MDP）并引入高效剪枝策略，解决了传统RAG在探索效率、稀疏奖励和全局反馈模糊性上的问题，显著提升了任务分解、动态检索和答案生成能力，实验表明其在多个数据集上性能提升6.2%，数据构建效率提高近6倍。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.05524v1">KEO: Knowledge Extraction on OMIn via Knowledge Graphs and RAG for Safety-Critical Aviation Maintenance</a></td><td><details><summary>展开</summary>We present Knowledge Extraction on OMIn (KEO), a domain-specific knowledge
extraction and reasoning framework with large language models (LLMs) in
safety-critical contexts. Using the Operations and Maintenance Intelligence
(OMIn) dataset, we construct a QA benchmark spanning global sensemaking and
actionable maintenance tasks. KEO builds a structured Knowledge Graph (KG) and
integrates it into a retrieval-augmented generation (RAG) pipeline, enabling
more coherent, dataset-wide reasoning than traditional text-chunk RAG. We
evaluate locally deployable LLMs (Gemma-3, Phi-4, Mistral-Nemo) and employ
stronger models (GPT-4o, Llama-3.3) as judges. Experiments show that KEO
markedly improves global sensemaking by revealing patterns and system-level
insights, while text-chunk RAG remains effective for fine-grained procedural
tasks requiring localized retrieval. These findings underscore the promise of
KG-augmented LLMs for secure, domain-specific QA and their potential in
high-stakes reasoning.</details></td><td><details><summary>展开</summary>该论文提出了KEO框架，通过构建结构化知识图谱（KG）并将其集成到检索增强生成（RAG）流程中，提升大语言模型在安全关键领域（基于OMIn数据集）的全局推理能力，实验表明KG增强的RAG在系统级分析上优于传统文本片段检索，同时保留了细粒度任务的处理优势。</details></td></tr></tbody></table>

### 📅 2025-10-06
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.04905v1">Retrieval-Augmented Code Generation: A Survey with Focus on Repository-Level Approaches</a></td><td><details><summary>展开</summary>Recent advancements in large language models (LLMs) have substantially
improved automated code generation. While function-level and file-level
generation have achieved promising results, real-world software development
typically requires reasoning across entire repositories. This gives rise to the
challenging task of Repository-Level Code Generation (RLCG), where models must
capture long-range dependencies, ensure global semantic consistency, and
generate coherent code spanning multiple files or modules. To address these
challenges, Retrieval-Augmented Generation (RAG) has emerged as a powerful
paradigm that integrates external retrieval mechanisms with LLMs, enhancing
context-awareness and scalability. In this survey, we provide a comprehensive
review of research on Retrieval-Augmented Code Generation (RACG), with an
emphasis on repository-level approaches. We categorize existing work along
several dimensions, including generation strategies, retrieval modalities,
model architectures, training paradigms, and evaluation protocols. Furthermore,
we summarize widely used datasets and benchmarks, analyze current limitations,
and outline key challenges and opportunities for future research. Our goal is
to establish a unified analytical framework for understanding this rapidly
evolving field and to inspire continued progress in AI-powered software
engineering.</details></td><td><details><summary>展开</summary>这篇论文探讨了在大语言模型（LLMs）背景下，利用检索增强生成（RAG）技术解决**仓库级代码生成（RLCG）**挑战的研究进展，系统综述了检索增强代码生成（RACG）的方法、分类（如生成策略、检索模态等）、数据集及未来方向，旨在构建统一的分析框架并推动AI驱动的软件工程发展。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.04757v1">ModernBERT + ColBERT: Enhancing biomedical RAG through an advanced re-ranking retriever</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) is a powerful technique for enriching
Large Language Models (LLMs) with external knowledge, allowing for factually
grounded responses, a critical requirement in high-stakes domains such as
healthcare. However, the efficacy of RAG systems is fundamentally restricted by
the performance of their retrieval module, since irrelevant or semantically
misaligned documents directly compromise the accuracy of the final generated
response. General-purpose dense retrievers can struggle with the nuanced
language of specialised domains, while the high accuracy of in-domain models is
often achieved at prohibitive computational costs. In this work, we aim to
address this trade-off by developing and evaluating a two-stage retrieval
architecture that combines a lightweight ModernBERT bidirectional encoder for
efficient initial candidate retrieval with a ColBERTv2 late-interaction model
for fine-grained re-ranking. We conduct comprehensive evaluations of our
retriever module performance and RAG system performance in the biomedical
context, fine-tuning the IR module using 10k question-passage pairs from
PubMedQA. Our analysis of the retriever module confirmed the positive impact of
the ColBERT re-ranker, which improved Recall@3 by up to 4.2 percentage points
compared to its retrieve-only counterpart. When integrated into the biomedical
RAG, our IR module leads to a state-of-the-art average accuracy of 0.4448 on
the five tasks of the MIRAGE question-answering benchmark, outperforming strong
baselines such as MedCPT (0.4436). Our ablation studies reveal that this
performance is critically dependent on a joint fine-tuning process that aligns
the retriever and re-ranker; otherwise, the re-ranker might degrade the
performance.</details></td><td><details><summary>展开</summary>该论文提出了一种结合轻量级ModernBERT和ColBERTv2的两阶段检索架构，以提升生物医学领域RAG系统的检索性能，通过在PubMedQA数据集上的微调和实验验证，显著提高了召回率和问答准确性，并在MIRAGE基准测试中达到最优水平。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.04536v1">3Dify: a Framework for Procedural 3D-CG Generation Assisted by LLMs Using MCP and RAG</a></td><td><details><summary>展开</summary>This paper proposes "3Dify," a procedural 3D computer graphics (3D-CG)
generation framework utilizing Large Language Models (LLMs). The framework
enables users to generate 3D-CG content solely through natural language
instructions. 3Dify is built upon Dify, an open-source platform for AI
application development, and incorporates several state-of-the-art LLM-related
technologies such as the Model Context Protocol (MCP) and Retrieval-Augmented
Generation (RAG). For 3D-CG generation support, 3Dify automates the operation
of various Digital Content Creation (DCC) tools via MCP. When DCC tools do not
support MCP-based interaction, the framework employs the Computer-Using Agent
(CUA) method to automate Graphical User Interface (GUI) operations. Moreover,
to enhance image generation quality, 3Dify allows users to provide feedback by
selecting preferred images from multiple candidates. The LLM then learns
variable patterns from these selections and applies them to subsequent
generations. Furthermore, 3Dify supports the integration of locally deployed
LLMs, enabling users to utilize custom-developed models and to reduce both time
and monetary costs associated with external API calls by leveraging their own
computational resources.</details></td><td><details><summary>展开</summary>这篇论文提出了“3Dify”，一个基于大语言模型（LLMs）的程序化3D计算机图形生成框架，通过自然语言指令生成3D内容。它整合了包括检索增强生成（RAG）在内的先进LLM技术，并利用Model Context Protocol（MCP）和Computer-Using Agent（CUA）方法自动化数字内容创建工具的操作，同时支持用户反馈和本地LLM部署以优化生成质量和降低成本。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.04488v1">Multi-Agent Collaborative Intelligence: Dual-Dial Control for Reliable LLM Reasoning</a></td><td><details><summary>展开</summary>Multi-agent debate often wastes compute by using a fixed adversarial stance,
aggregating without deliberation, or stopping on heuristics. We introduce MACI,
an active controller with two independent dials that decouple information from
behavior: an information dial that gates evidence by quality, and a behavior
dial that schedules contentiousness from exploration to consolidation. A
moderator tracks disagreement, overlap, evidence quality, and argument quality,
and halts when gains plateau. We provide theory-lite guarantees for
nonincreasing dispersion and provable termination, with a budget-feasible
scheduler. Across clinical diagnosis and news-bias tasks, MACI improves
accuracy and calibration while reducing tokens, and converts residual
uncertainty into precision RAG plans that specify what to retrieve next. We use
a cross-family LLM judge (CRIT) as a conservative soft weight and stop signal,
validated for order invariance and judge-swap stability; stability depends on
using high-capability judges. MACI turns debate into a budget-aware,
measurable, and provably terminating controller.</details></td><td><details><summary>展开</summary>这篇论文介绍了MACI，一种多智能体辩论控制器，通过信息质量筛选和行为调度优化辩论过程，并在残余不确定性时生成精确的RAG计划以指导后续检索，从而提升任务准确性和校准度。</details></td></tr></tbody></table>

### 📅 2025-10-05
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.04392v1">Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards</a></td><td><details><summary>展开</summary>RAG systems are increasingly deployed in high-stakes domains where users
expect outputs to be consistent across semantically equivalent queries.
However, existing systems often exhibit significant inconsistencies due to
variability in both the retriever and generator (LLM), undermining trust and
reliability. In this work, we focus on information consistency, i.e., the
requirement that outputs convey the same core content across semantically
equivalent inputs. We introduce a principled evaluation framework that
decomposes RAG consistency into retriever-level, generator-level, and
end-to-end components, helping identify inconsistency sources. To improve
consistency, we propose Paraphrased Set Group Relative Policy Optimization
(PS-GRPO), an RL approach that leverages multiple rollouts across paraphrased
set to assign group similarity rewards. We leverage PS-GRPO to achieve
Information Consistent RAG (Con-RAG), training the generator to produce
consistent outputs across paraphrased queries and remain robust to
retrieval-induced variability. Because exact reward computation over paraphrase
sets is computationally expensive, we also introduce a scalable approximation
method that retains effectiveness while enabling efficient, large-scale
training. Empirical evaluations across short-form, multi-hop, and long-form QA
benchmarks demonstrate that Con-RAG significantly improves both consistency and
accuracy over strong baselines, even in the absence of explicit ground-truth
supervision. Our work provides practical solutions for evaluating and building
reliable RAG systems for safety-critical deployments.</details></td><td><details><summary>展开</summary>该论文针对RAG系统在语义等效查询下输出不一致的问题，提出了一种评估框架（分解检索器、生成器和端到端不一致性）和改进方法PS-GRPO（基于强化学习的组相似性奖励），最终实现了信息一致性更强的Con-RAG系统，并通过实验验证了其在多任务中提升一致性和准确性的有效性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.04293v1">Equipping Retrieval-Augmented Large Language Models with Document Structure Awareness</a></td><td><details><summary>展开</summary>While large language models (LLMs) demonstrate impressive capabilities, their
reliance on parametric knowledge often leads to factual inaccuracies.
Retrieval-Augmented Generation (RAG) mitigates this by leveraging external
documents, yet existing approaches treat retrieved passages as isolated chunks,
ignoring valuable structure that is crucial for document organization.
Motivated by this gap, we propose Retrieve-DocumentRoute-Read (RDR2), a novel
framework that explicitly incorporates structural information throughout the
RAG process. RDR2 employs an LLM-based router to dynamically navigate document
structure trees, jointly evaluating content relevance and hierarchical
relationships to assemble optimal evidence. Our key innovation lies in
formulating document routing as a trainable task, with automatic action
curation and structure-aware passage selection inspired by human reading
strategies. Through comprehensive evaluation on five challenging datasets, RDR2
achieves state-of-the-art performance, demonstrating that explicit structural
awareness significantly enhances RAG systems' ability to acquire and utilize
knowledge, particularly in complex scenarios requiring multi-document
synthesis.</details></td><td><details><summary>展开</summary>本文提出了一种名为RDR2的新型检索增强生成（RAG）框架，通过显式利用文档结构信息改进传统RAG方法。该框架采用基于LLM的路由器动态导航文档结构树，结合内容相关性和层次关系选择最优证据，并在五个数据集上实现了最先进的性能，证明了结构感知能显著提升RAG系统在复杂多文档场景下的知识获取与利用能力。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.04226v3">Epistemic Diversity and Knowledge Collapse in Large Language Models</a></td><td><details><summary>展开</summary>Large language models (LLMs) tend to generate lexically, semantically, and
stylistically homogenous texts. This poses a risk of knowledge collapse, where
homogenous LLMs mediate a shrinking in the range of accessible information over
time. Existing works on homogenization are limited by a focus on closed-ended
multiple-choice setups or fuzzy semantic features, and do not look at trends
across time and cultural contexts. To overcome this, we present a new
methodology to measure epistemic diversity, i.e., variation in real-world
claims in LLM outputs, which we use to perform a broad empirical study of LLM
knowledge collapse. We test 27 LLMs, 155 topics covering 12 countries, and 200
prompt variations sourced from real user chats. For the topics in our study, we
show that while newer models tend to generate more diverse claims, nearly all
models are less epistemically diverse than a basic web search. We find that
model size has a negative impact on epistemic diversity, while
retrieval-augmented generation (RAG) has a positive impact, though the
improvement from RAG varies by the cultural context. Finally, compared to a
traditional knowledge source (Wikipedia), we find that country-specific claims
reflect the English language more than the local one, highlighting a gap in
epistemic representation</details></td><td><details><summary>展开</summary>这篇论文研究了大型语言模型（LLMs）生成文本的同质化问题及其导致的“知识崩溃”风险，提出了一种衡量认知多样性（epistemic diversity）的新方法，并通过实验发现检索增强生成（RAG）技术能显著提升模型输出的多样性，但其效果受文化背景影响。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.04145v1">Automating construction safety inspections using a multi-modal vision-language RAG framework</a></td><td><details><summary>展开</summary>Conventional construction safety inspection methods are often inefficient as
they require navigating through large volume of information. Recent advances in
large vision-language models (LVLMs) provide opportunities to automate safety
inspections through enhanced visual and linguistic understanding. However,
existing applications face limitations including irrelevant or unspecific
responses, restricted modal inputs and hallucinations. Utilisation of Large
Language Models (LLMs) for this purpose is constrained by availability of
training data and frequently lack real-time adaptability. This study introduces
SiteShield, a multi-modal LVLM-based Retrieval-Augmented Generation (RAG)
framework for automating construction safety inspection reports by integrating
visual and audio inputs. Using real-world data, SiteShield outperformed
unimodal LLMs without RAG with an F1 score of 0.82, hamming loss of 0.04,
precision of 0.76, and recall of 0.96. The findings indicate that SiteShield
offers a novel pathway to enhance information retrieval and efficiency in
generating safety reports.</details></td><td><details><summary>展开</summary>该论文提出了一种基于多模态大视觉语言模型（LVLM）的检索增强生成（RAG）框架SiteShield，用于自动化生成建筑安全检查报告，通过整合视觉和音频输入提升检索和生成效率，实验表明其性能优于单模态LLM模型。</details></td></tr></tbody></table>

### 📅 2025-10-04
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.03847v1">Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade offs</a></td><td><details><summary>展开</summary>Small language models (SLMs; 1-12B params, sometimes up to 20B) are
sufficient and often superior for agentic workloads where the objective is
schema- and API-constrained accuracy rather than open-ended generation. We
synthesize recent evidence across open and proprietary SLMs (Phi-4-Mini,
Qwen-2.5-7B, Gemma-2-9B, Llama-3.2-1B/3B, Ministral-3B/8B, Apple on-device 3B,
DeepSeek-R1-Distill) and connect it to modern evaluations (BFCL v3/v4,
StableToolBench) and serving stacks (vLLM, SGLang, TensorRT-LLM) paired with
guided decoding libraries (XGrammar, Outlines). We formalize SLM-default,
LLM-fallback systems with uncertainty-aware routing and verifier cascades, and
propose engineering metrics that reflect real production goals: cost per
successful task (CPS), schema validity rate, executable call rate, p50/p95
latency, and energy per request. Guided decoding, strict JSON Schema outputs,
and validator-first tool execution close much of the capability gap with larger
models and often let SLMs match or surpass LLMs on tool use, function calling,
and RAG at 10x-100x lower token cost with materially better latency and energy.
We provide design patterns for agent stacks that prioritize SLMs: schema-first
prompting, type-safe function registries, confidence scoring with verifier
rollups, and lightweight adaptation via LoRA/QLoRA. We also delineate limits
where fallback remains valuable (open-domain reasoning and some long-horizon
planning). The result is a practical blueprint for building fast, inexpensive,
and reliable agents that default to SLMs while preserving headroom with
targeted LLM assistance.
  Keywords: small language models, agents, function calling, structured
outputs, JSON Schema, guided decoding, LoRA/QLoRA, routing, energy efficiency,
edge inference</details></td><td><details><summary>展开</summary>这篇论文探讨了小型语言模型（SLMs）在代理任务中的优势，特别是在结构化输出和API调用等受限场景下的高效表现，并提出了结合不确定性感知路由和验证器级联的SLM-default系统。虽然主要聚焦于工具使用和函数调用，但明确提到SLMs在RAG任务中能以更低成本匹配或超越大型模型，同时提供了优化RAG性能的设计模式（如schema-first提示和轻量级适配技术）。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.03687v1">MedReflect: Teaching Medical LLMs to Self-Improve via Reflective Correction</a></td><td><details><summary>展开</summary>Medical problem solving demands expert knowledge and intricate reasoning.
Recent studies of large language models (LLMs) attempt to ease this complexity
by introducing external knowledge verification through retrieval-augmented
generation or by training on reasoning datasets. However, these approaches
suffer from drawbacks such as retrieval overhead and high annotation costs, and
they heavily rely on substituted external assistants to reach limited
performance in medical field. In this paper, we introduce MedReflect, a
generalizable framework designed to inspire LLMs with a physician-like
reflective thinking mode. MedReflect generates a single-pass reflection chain
that includes initial hypothesis generation, self-questioning, self-answering
and decision refinement. This self-verified and self-reflective nature releases
large language model's latent capability in medical problem-solving without
external retrieval or heavy annotation. We demonstrate that MedReflect enables
cost-efficient medical dataset construction: with merely 2,000 randomly sampled
training examples and a light fine-tuning, this approach achieves notable
absolute accuracy improvements across a series of medical benchmarks while
cutting annotation requirements. Our results provide evidence that LLMs can
learn to solve specialized medical problems via self-reflection and
self-improve, reducing reliance on external supervision and extensive
task-specific fine-tuning data.</details></td><td><details><summary>展开</summary>这篇论文探讨了在医学问题解决中，大型语言模型（LLMs）通过自我反思模式（MedReflect框架）提升性能的方法，同时对比了传统检索增强生成（RAG）的局限性（如检索开销和依赖外部知识）。研究提出了一种无需外部检索或大量标注的自验证反思链机制，显著降低了数据需求并提高了模型在医学任务中的准确性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.03663v2">UNIDOC-BENCH: A Unified Benchmark for Document-Centric Multimodal RAG</a></td><td><details><summary>展开</summary>Multimodal retrieval-augmented generation (MM-RAG) is a key approach for
applying large language models (LLMs) and agents to real-world knowledge bases,
yet current evaluations are fragmented, focusing on either text or images in
isolation or on simplified multimodal setups that fail to capture
document-centric multimodal use cases. In this paper, we introduce
UniDoc-Bench, the first large-scale, realistic benchmark for MM-RAG built from
70k real-world PDF pages across eight domains. Our pipeline extracts and links
evidence from text, tables, and figures, then generates 1,600 multimodal QA
pairs spanning factual retrieval, comparison, summarization, and logical
reasoning queries. To ensure reliability, 20% of QA pairs are validated by
multiple annotators and expert adjudication. UniDoc-Bench supports
apples-to-apples comparison across four paradigms: (1) text-only, (2)
image-only, (3) multimodal text-image fusion, and (4) multimodal joint
retrieval -- under a unified protocol with standardized candidate pools,
prompts, and evaluation metrics. Our experiments show that multimodal
text-image fusion RAG systems consistently outperform both unimodal and jointly
multimodal embedding-based retrieval, indicating that neither text nor images
alone are sufficient and that current multimodal embeddings remain inadequate.
Beyond benchmarking, our analysis reveals when and how visual context
complements textual evidence, uncovers systematic failure modes, and offers
actionable guidance for developing more robust MM-RAG pipelines.</details></td><td><details><summary>展开</summary>这篇论文介绍了UniDoc-Bench，一个针对多模态检索增强生成（MM-RAG）的大规模真实基准测试，通过从8个领域的7万页PDF中提取文本、表格和图像证据并生成1,600个多模态QA对，评估了四种检索范式（纯文本、纯图像、多模态融合等），揭示了多模态融合系统的优势及当前嵌入方法的不足，为MM-RAG系统开发提供了实践指导。</details></td></tr></tbody></table>

### 📅 2025-10-03
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.03521v1">Identifying Financial Risk Information Using RAG with a Contrastive Insight</a></td><td><details><summary>展开</summary>In specialized domains, humans often compare new problems against similar
examples, highlight nuances, and draw conclusions instead of analyzing
information in isolation. When applying reasoning in specialized contexts with
LLMs on top of a RAG, the pipeline can capture contextually relevant
information, but it is not designed to retrieve comparable cases or related
problems.
  While RAG is effective at extracting factual information, its outputs in
specialized reasoning tasks often remain generic, reflecting broad facts rather
than context-specific insights. In finance, it results in generic risks that
are true for the majority of companies. To address this limitation, we propose
a peer-aware comparative inference layer on top of RAG.
  Our contrastive approach outperforms baseline RAG in text generation metrics
such as ROUGE and BERTScore in comparison with human-generated equity research
and risk.</details></td><td><details><summary>展开</summary>该论文提出在RAG基础上增加一个同行感知的比较推理层，以解决RAG在专业领域推理任务中输出过于通用的问题，并在金融领域通过对比实验验证了该方法在生成质量指标（如ROUGE和BERTScore）上优于基线RAG模型。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.03458v1">Omni-Embed-Nemotron: A Unified Multimodal Retrieval Model for Text, Image, Audio, and Video</a></td><td><details><summary>展开</summary>We present Omni-Embed-Nemotron, a unified multimodal retrieval embedding
model developed to handle the increasing complexity of real-world information
needs. While Retrieval-Augmented Generation (RAG) has significantly advanced
language models by incorporating external knowledge, existing text-based
retrievers rely on clean, structured input and struggle with the visually and
semantically rich content found in real-world documents such as PDFs, slides,
or videos. Recent work such as ColPali has shown that preserving document
layout using image-based representations can improve retrieval quality.
Building on this, and inspired by the capabilities of recent multimodal models
such as Qwen2.5-Omni, we extend retrieval beyond text and images to also
support audio and video modalities. Omni-Embed-Nemotron enables both
cross-modal (e.g., text - video) and joint-modal (e.g., text - video+audio)
retrieval using a single model. We describe the architecture, training setup,
and evaluation results of Omni-Embed-Nemotron, and demonstrate its
effectiveness in text, image, and video retrieval.</details></td><td><details><summary>展开</summary>该论文提出了Omni-Embed-Nemotron，一种统一的多模态检索嵌入模型，旨在处理现实世界中复杂的信息需求。文章指出，尽管RAG技术通过整合外部知识显著提升了语言模型的能力，但现有的基于文本的检索器在处理PDF、幻灯片或视频等视觉和语义丰富的内容时存在局限。Omni-Embed-Nemotron扩展了检索范围，支持文本、图像、音频和视频的多模态检索，包括跨模态和联合模态检索，并展示了其在文本、图像和视频检索中的有效性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.03418v1">ContraGen: A Multi-Agent Generation Framework for Enterprise Contradictions Detection</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) integrates LLMs with external sources,
offering advanced capabilities for information access and decision-making.
However, contradictions in retrieved evidence can result in inconsistent or
untrustworthy outputs, which is especially problematic in enterprise settings
where compliance, governance, and accountability are critical. Existing
benchmarks for contradiction detection are limited to sentence-level analysis
and do not capture the complexity of enterprise documents such as contracts,
financial filings, compliance reports, or policy manuals. To address this
limitation, we propose ContraGen, a contradiction-aware benchmark framework
tailored to enterprise domain. The framework generates synthetic
enterprise-style documents with embedded contradictions, enabling systematic
evaluation of both intra-document and cross-document consistency. Automated
contradiction mining is combined with human-in-the-loop validation to ensure
high accuracy. Our contributions include generating realistic enterprise
documents, modeling a taxonomy of contradiction types common in business
processes, enabling controlled creation of self- and pairwise contradictions,
developing a contradiction-aware retrieval evaluation pipeline and embedding
human oversight to reflect domain-specific judgment complexity. This work
establishes a foundation for more trustworthy and accountable RAG systems in
enterprise information-seeking applications, where detecting and resolving
contradictions is essential for reducing risk and ensuring compliance.</details></td><td><details><summary>展开</summary>该论文针对RAG系统中检索证据矛盾导致输出不可信的问题，提出面向企业领域的ContraGen基准框架，通过生成含矛盾的企业文档、构建矛盾分类体系及评估流程，提升RAG在企业合规场景下的可靠性与矛盾检测能力。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.02967v1">Grounding Large Language Models in Clinical Evidence: A Retrieval-Augmented Generation System for Querying UK NICE Clinical Guidelines</a></td><td><details><summary>展开</summary>This paper presents the development and evaluation of a Retrieval-Augmented
Generation (RAG) system for querying the United Kingdom's National Institute
for Health and Care Excellence (NICE) clinical guidelines using Large Language
Models (LLMs). The extensive length and volume of these guidelines can impede
their utilisation within a time-constrained healthcare system, a challenge this
project addresses through the creation of a system capable of providing users
with precisely matched information in response to natural language queries. The
system's retrieval architecture, composed of a hybrid embedding mechanism, was
evaluated against a database of 10,195 text chunks derived from three hundred
guidelines. It demonstrates high performance, with a Mean Reciprocal Rank (MRR)
of 0.814, a Recall of 81% at the first chunk and of 99.1% within the top ten
retrieved chunks, when evaluated on 7901 queries.
  The most significant impact of the RAG system was observed during the
generation phase. When evaluated on a manually curated dataset of seventy
question-answer pairs, RAG-enhanced models showed substantial gains in
performance. Faithfulness, the measure of whether an answer is supported by the
source text, was increased by 64.7 percentage points to 99.5% for the
RAG-enhanced O4-Mini model and significantly outperformed the medical-focused
Meditron3-8B LLM, which scored 43%. This, combined with a perfect Context
Precision score of 1 for all RAG-enhanced models, confirms the system's ability
to prevent information fabrication by grounding its answers in relevant source
material. This study thus establishes RAG as an effective, reliable, and
scalable approach for applying generative AI in healthcare, enabling
cost-effective access to medical guidelines.</details></td><td><details><summary>展开</summary>该论文开发并评估了一个基于RAG的系统，用于通过大语言模型查询英国NICE临床指南，通过混合嵌入检索架构从大量指南文本中精准匹配信息，显著提升了生成答案的准确性和可靠性（如忠实度提升至99.5%），验证了RAG在医疗领域的高效应用。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.02936v1">RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification</a></td><td><details><summary>展开</summary>Medical time series analysis is challenging due to data sparsity, noise, and
highly variable recording lengths. Prior work has shown that stochastic sparse
sampling effectively handles variable-length signals, while retrieval-augmented
approaches improve explainability and robustness to noise and weak temporal
correlations. In this study, we generalize the stochastic sparse sampling
framework for retrieval-informed classification. Specifically, we weight window
predictions by within-channel similarity and aggregate them in probability
space, yielding convex series-level scores and an explicit evidence trail for
explainability. Our method achieves competitive iEEG classification performance
and provides practitioners with greater transparency and explainability. We
evaluate our method in iEEG recordings collected in four medical centers,
demonstrating its potential for reliable and explainable clinical
variable-length time series classification.</details></td><td><details><summary>展开</summary>该论文提出了一种结合随机稀疏采样和检索增强方法的框架，用于处理医疗时间序列分类问题，通过基于通道内相似性加权的窗口预测和概率空间聚合，提高了分类性能、可解释性及对噪声的鲁棒性，并在多中心iEEG数据上验证了其可靠性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.02827v1">StepChain GraphRAG: Reasoning Over Knowledge Graphs for Multi-Hop Question Answering</a></td><td><details><summary>展开</summary>Recent progress in retrieval-augmented generation (RAG) has led to more
accurate and interpretable multi-hop question answering (QA). Yet, challenges
persist in integrating iterative reasoning steps with external knowledge
retrieval. To address this, we introduce StepChain GraphRAG, a framework that
unites question decomposition with a Breadth-First Search (BFS) Reasoning Flow
for enhanced multi-hop QA. Our approach first builds a global index over the
corpus; at inference time, only retrieved passages are parsed on-the-fly into a
knowledge graph, and the complex query is split into sub-questions. For each
sub-question, a BFS-based traversal dynamically expands along relevant edges,
assembling explicit evidence chains without overwhelming the language model
with superfluous context. Experiments on MuSiQue, 2WikiMultiHopQA, and HotpotQA
show that StepChain GraphRAG achieves state-of-the-art Exact Match and F1
scores. StepChain GraphRAG lifts average EM by 2.57% and F1 by 2.13% over the
SOTA method, achieving the largest gain on HotpotQA (+4.70% EM, +3.44% F1).
StepChain GraphRAG also fosters enhanced explainability by preserving the
chain-of-thought across intermediate retrieval steps. We conclude by discussing
how future work can mitigate the computational overhead and address potential
hallucinations from large language models to refine efficiency and reliability
in multi-hop QA.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为StepChain GraphRAG的框架，通过结合问题分解和广度优先搜索（BFS）推理流程，改进了多跳问答（QA）任务。该方法在检索时动态构建知识图，将复杂查询拆分为子问题，并通过BFS遍历扩展相关证据链，从而提升准确性（在多个数据集上达到SOTA性能）和可解释性，同时讨论了未来优化计算效率和减少大模型幻觉的方向。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.02803v1">Work Zones challenge VLM Trajectory Planning: Toward Mitigation and Robust Autonomous Driving</a></td><td><details><summary>展开</summary>Visual Language Models (VLMs), with powerful multimodal reasoning
capabilities, are gradually integrated into autonomous driving by several
automobile manufacturers to enhance planning capability in challenging
environments. However, the trajectory planning capability of VLMs in work
zones, which often include irregular layouts, temporary traffic control, and
dynamically changing geometric structures, is still unexplored. To bridge this
gap, we conduct the \textit{first} systematic study of VLMs for work zone
trajectory planning, revealing that mainstream VLMs fail to generate correct
trajectories in $68.0%$ of cases. To better understand these failures, we first
identify candidate patterns via subgraph mining and clustering analysis, and
then confirm the validity of $8$ common failure patterns through human
verification. Building on these findings, we propose REACT-Drive, a trajectory
planning framework that integrates VLMs with Retrieval-Augmented Generation
(RAG). Specifically, REACT-Drive leverages VLMs to convert prior failure cases
into constraint rules and executable trajectory planning code, while RAG
retrieves similar patterns in new scenarios to guide trajectory generation.
Experimental results on the ROADWork dataset show that REACT-Drive yields a
reduction of around $3\times$ in average displacement error relative to VLM
baselines under evaluation with Qwen2.5-VL. In addition, REACT-Drive yields the
lowest inference time ($0.58$s) compared with other methods such as fine-tuning
($17.90$s). We further conduct experiments using a real vehicle in 15 work zone
scenarios in the physical world, demonstrating the strong practicality of
REACT-Drive.</details></td><td><details><summary>展开</summary>这篇文章探讨了视觉语言模型（VLMs）在自主驾驶工作区轨迹规划中的局限性和改进方法，提出了一种名为REACT-Drive的框架，该框架通过结合检索增强生成（RAG）技术，利用检索到的相似失败模式指导轨迹规划，显著提升了轨迹生成的准确性和效率。实验证明REACT-Drive在减少平均位移误差和推理时间方面优于基线方法，并在真实场景中验证了其实用性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.02668v1">AgenticRAG: Tool-Augmented Foundation Models for Zero-Shot Explainable Recommender Systems</a></td><td><details><summary>展开</summary>Foundation models have revolutionized artificial intelligence, yet their
application in recommender systems remains limited by reasoning opacity and
knowledge constraints. This paper introduces AgenticRAG, a novel framework that
combines tool-augmented foundation models with retrieval-augmented generation
for zero-shot explainable recommendations. Our approach integrates external
tool invocation, knowledge retrieval, and chain-of-thought reasoning to create
autonomous recommendation agents capable of transparent decision-making without
task-specific training. Experimental results on three real-world datasets
demonstrate that AgenticRAG achieves consistent improvements over
state-of-the-art baselines, with NDCG@10 improvements of 0.4\% on Amazon
Electronics, 0.8\% on MovieLens-1M, and 1.6\% on Yelp datasets. The framework
exhibits superior explainability while maintaining computational efficiency
comparable to traditional methods.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为AgenticRAG的新型框架，将工具增强的基础模型与检索增强生成（RAG）相结合，用于零样本可解释推荐。该框架通过整合外部工具调用、知识检索和思维链推理，创建了能够透明决策的自主推荐代理，无需特定任务训练。实验表明其在多个数据集上的性能优于现有基准模型，同时保持了较好的解释性和计算效率。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.02657v2">Less LLM, More Documents: Searching for Improved RAG</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) couples document retrieval with large
language models (LLMs). While scaling generators improves accuracy, it also
raises cost and limits deployability. We explore an orthogonal axis: enlarging
the retriever's corpus to reduce reliance on large LLMs. Experimental results
show that corpus scaling consistently strengthens RAG and can often serve as a
substitute for increasing model size, though with diminishing returns at larger
scales. Small- and mid-sized generators paired with larger corpora often rival
much larger models with smaller corpora; mid-sized models tend to gain the
most, while tiny and large models benefit less. Our analysis shows that
improvements arise primarily from increased coverage of answer-bearing
passages, while utilization efficiency remains largely unchanged. These
findings establish a principled corpus-generator trade-off: investing in larger
corpora offers an effective path to stronger RAG, often comparable to enlarging
the LLM itself.</details></td><td><details><summary>展开</summary>该论文探讨了通过扩大检索器的语料库来减少对大型语言模型（LLM）依赖的方法，实验表明语料库扩展能有效增强RAG性能，可作为增大模型规模的替代方案，尤其对中小型生成器效果显著，并揭示了语料覆盖范围与模型效率之间的权衡关系。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.02653v1">Geolog-IA: Conversational System for Academic Theses</a></td><td><details><summary>展开</summary>This study presents the development of Geolog-IA, a novel conversational
system based on artificial intelligence that responds naturally to questions
about geology theses from the Central University of Ecuador. Our proposal uses
the Llama 3.1 and Gemini 2.5 language models, which are complemented by a
Retrieval Augmented Generation (RAG) architecture and an SQLite database. This
strategy allows us to overcome problems such as hallucinations and outdated
knowledge. The evaluation of Geolog-IA's performance with the BLEU metric
reaches an average of 0.87, indicating high consistency and accuracy in the
responses generated. The system offers an intuitive, web-based interface that
facilitates interaction and information retrieval for directors, teachers,
students, and administrative staff at the institution. This tool can be a key
support in education, training, and research and establishes a basis for future
applications in other disciplines.</details></td><td><details><summary>展开</summary>该论文介绍了Geolog-IA，一个基于人工智能的对话系统，利用Llama 3.1和Gemini 2.5语言模型，结合RAG架构和SQLite数据库，以解决幻觉和知识过时问题，为厄瓜多尔中央大学的地质学论文提供高准确性的自然语言回答，并通过BLEU指标评估显示其高一致性（平均0.87）。系统提供基于网页的直观界面，支持教育、培训和研究，并适用于其他学科。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.02634v1">Automatic Building Code Review: A Case Study</a></td><td><details><summary>展开</summary>Building officials, particularly those in resource-constrained or rural
jurisdictions, face labor-intensive, error-prone, and costly manual reviews of
design documents as projects increase in size and complexity. The growing
adoption of Building Information Modeling (BIM) and Large Language Models
(LLMs) presents opportunities for automated code review (ACR) solutions. This
study introduces a novel agent-driven framework that integrates BIM-based data
extraction with automated verification using both retrieval-augmented
generation (RAG) and Model Context Protocol (MCP) agent pipelines. The
framework employs LLM-enabled agents to extract geometry, schedules, and system
attributes from heterogeneous file types, which are then processed for building
code checking through two complementary mechanisms: (1) direct API calls to the
US Department of Energy COMcheck engine, providing deterministic and
audit-ready outputs, and (2) RAG-based reasoning over rule provisions, enabling
flexible interpretation where coverage is incomplete or ambiguous.
  The framework was evaluated through case demonstrations, including automated
extraction of geometric attributes (such as surface area, tilt, and insulation
values), parsing of operational schedules, and validation of lighting
allowances under ASHRAE Standard 90.1-2022. Comparative performance tests
across multiple LLMs showed that GPT-4o achieved the best balance of efficiency
and stability, while smaller models exhibited inconsistencies or failures.
Results confirm that MCP agent pipelines outperform RAG reasoning pipelines in
rigor and reliability. This work advances ACR research by demonstrating a
scalable, interoperable, and production-ready approach that bridges BIM with
authoritative code review tools.</details></td><td><details><summary>展开</summary>该论文提出了一种基于BIM和LLM的自动化建筑规范审查框架，整合了检索增强生成（RAG）和模型上下文协议（MCP）代理流程，通过两种机制验证建筑规范：直接调用COMcheck引擎和RAG对规则条款进行推理，案例测试表明MCP在严谨性和可靠性上优于RAG，但RAG在规则不明确时提供了灵活解释能力。</details></td></tr></tbody></table>

### 📅 2025-10-02
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.02243v1">AccurateRAG: A Framework for Building Accurate Retrieval-Augmented Question-Answering Applications</a></td><td><details><summary>展开</summary>We introduce AccurateRAG -- a novel framework for constructing
high-performance question-answering applications based on retrieval-augmented
generation (RAG). Our framework offers a pipeline for development efficiency
with tools for raw dataset processing, fine-tuning data generation, text
embedding & LLM fine-tuning, output evaluation, and building RAG systems
locally. Experimental results show that our framework outperforms previous
strong baselines and obtains new state-of-the-art question-answering
performance on benchmark datasets.</details></td><td><details><summary>展开</summary>这篇论文介绍了名为AccurateRAG的新框架，旨在基于检索增强生成（RAG）构建高性能问答应用。该框架提供了一套开发流程工具，包括原始数据集处理、微调数据生成、文本嵌入与大模型微调、输出评估及本地RAG系统构建，并在实验中超越现有基线，实现了基准数据集上的最新最先进问答性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.02044v1">Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usage</a></td><td><details><summary>展开</summary>End-to-end speech-in speech-out dialogue systems are emerging as a powerful
alternative to traditional ASR-LLM-TTS pipelines, generating more natural,
expressive responses with significantly lower latency. However, these systems
remain prone to hallucinations due to limited factual grounding. While
text-based dialogue systems address this challenge by integrating tools such as
web search and knowledge graph APIs, we introduce the first approach to extend
tool use directly into speech-in speech-out systems. A key challenge is that
tool integration substantially increases response latency, disrupting
conversational flow. To mitigate this, we propose Streaming Retrieval-Augmented
Generation (Streaming RAG), a novel framework that reduces user-perceived
latency by predicting tool queries in parallel with user speech, even before
the user finishes speaking. Specifically, we develop a post-training pipeline
that teaches the model when to issue tool calls during ongoing speech and how
to generate spoken summaries that fuse audio queries with retrieved text
results, thereby improving both accuracy and responsiveness. To evaluate our
approach, we construct AudioCRAG, a benchmark created by converting queries
from the publicly available CRAG dataset into speech form. Experimental results
demonstrate that our streaming RAG approach increases QA accuracy by up to 200%
relative (from 11.1% to 34.2% absolute) and further enhances user experience by
reducing tool use latency by 20%. Importantly, our streaming RAG approach is
modality-agnostic and can be applied equally to typed input, paving the way for
more agentic, real-time AI assistants.</details></td><td><details><summary>展开</summary>该论文提出了一种名为"Streaming Retrieval-Augmented Generation (Streaming RAG)"的新型框架，旨在解决端到端语音对话系统中存在的事实基础不足和延迟问题。通过并行预测工具查询并与用户语音同步处理，该方法显著提高了问答准确性（相对提升200%）并降低20%的工具使用延迟，同时构建了专门的语音评测基准AudioCRAG。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.01910v1">Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement</a></td><td><details><summary>展开</summary>Graph Neural Networks (GNNs) are widely adopted in Web-related applications,
serving as a core technique for learning from graph-structured data, such as
text-attributed graphs. Yet in real-world scenarios, such graphs exhibit
deficiencies that substantially undermine GNN performance. While prior
GNN-based augmentation studies have explored robustness against individual
imperfections, a systematic understanding of how graph-native and Large
Language Models (LLMs) enhanced methods behave under compound deficiencies is
still missing. Specifically, there has been no comprehensive investigation
comparing conventional approaches and recent LLM-on-graph frameworks, leaving
their merits unclear. To fill this gap, we conduct the first empirical study
that benchmarks these two lines of methods across diverse graph deficiencies,
revealing overlooked vulnerabilities and challenging the assumption that LLM
augmentation is consistently superior. Building on empirical findings, we
propose Robust Graph Learning via Retrieval-Augmented Contrastive Refinement
(RoGRAD) framework. Unlike prior one-shot LLM-as-Enhancer designs, RoGRAD is
the first iterative paradigm that leverages Retrieval-Augmented Generation
(RAG) to inject retrieval-grounded augmentations by supplying class-consistent,
diverse augmentations and enforcing discriminative representations through
iterative graph contrastive learning. It transforms LLM augmentation for graphs
from static signal injection into dynamic refinement. Extensive experiments
demonstrate RoGRAD's superiority over both conventional GNN- and LLM-enhanced
baselines, achieving up to 82.43% average improvement.</details></td><td><details><summary>展开</summary>该论文提出了一种名为RoGRAD的新型图学习框架，首次将检索增强生成（RAG）技术迭代应用于图神经网络（GNN）增强任务，通过动态对比学习注入检索到的类别一致性数据，解决了传统LLM静态增强和复合图缺陷下的性能瓶颈，实验显示其效果显著优于基线方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.01800v1">REBot: From RAG to CatRAG with Semantic Enrichment and Graph Routing</a></td><td><details><summary>展开</summary>Academic regulation advising is essential for helping students interpret and
comply with institutional policies, yet building effective systems requires
domain specific regulatory resources. To address this challenge, we propose
REBot, an LLM enhanced advisory chatbot powered by CatRAG, a hybrid retrieval
reasoning framework that integrates retrieval augmented generation with graph
based reasoning. CatRAG unifies dense retrieval and graph reasoning, supported
by a hierarchical, category labeled knowledge graph enriched with semantic
features for domain alignment. A lightweight intent classifier routes queries
to the appropriate retrieval modules, ensuring both factual accuracy and
contextual depth. We construct a regulation specific dataset and evaluate REBot
on classification and question answering tasks, achieving state of the art
performance with an F1 score of 98.89%. Finally, we implement a web application
that demonstrates the practical value of REBot in real world academic advising
scenarios.</details></td><td><details><summary>展开</summary>这篇论文提出了REBot，一种基于CatRAG（结合检索增强生成与图推理的混合框架）的学术法规咨询聊天机器人。CatRAG通过分层标记的知识图谱和语义特征整合密集检索与图推理，轻量级意图分类器确保查询的准确性和上下文深度。实验表明REBot在分类和问答任务中表现优异（F1分数98.89%），并通过网页应用验证了其实际应用价值。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.01622v1">LLM4Rec: Large Language Models for Multimodal Generative Recommendation with Causal Debiasing</a></td><td><details><summary>展开</summary>Contemporary generative recommendation systems face significant challenges in
handling multimodal data, eliminating algorithmic biases, and providing
transparent decision-making processes. This paper introduces an enhanced
generative recommendation framework that addresses these limitations through
five key innovations: multimodal fusion architecture, retrieval-augmented
generation mechanisms, causal inference-based debiasing, explainable
recommendation generation, and real-time adaptive learning capabilities. Our
framework leverages advanced large language models as the backbone while
incorporating specialized modules for cross-modal understanding, contextual
knowledge integration, bias mitigation, explanation synthesis, and continuous
model adaptation. Extensive experiments on three benchmark datasets
(MovieLens-25M, Amazon-Electronics, Yelp-2023) demonstrate consistent
improvements in recommendation accuracy, fairness, and diversity compared to
existing approaches. The proposed framework achieves up to 2.3% improvement in
NDCG@10 and 1.4% enhancement in diversity metrics while maintaining
computational efficiency through optimized inference strategies.</details></td><td><details><summary>展开</summary>这篇论文提出了一个改进的生成式推荐框架，通过五项关键创新解决多模态数据处理、算法偏差消除和决策透明度等问题，其中包括检索增强生成机制（RAG）。该框架结合大型语言模型与多模态融合、去偏因果推理等技术，在多个基准数据集上验证了其在推荐准确性、公平性和多样性方面的提升。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.01612v1">RAG-BioQA Retrieval-Augmented Generation for Long-Form Biomedical Question Answering</a></td><td><details><summary>展开</summary>The exponential growth of biomedical literature creates significant
challenges for accessing precise medical information. Current biomedical
question-answering systems primarily focus on short-form answers, failing to
provide the comprehensive explanations necessary for clinical decision-making.
We present RAG-BioQA, a novel framework combining retrieval-augmented
generation with domain-specific fine-tuning to produce evidence-based,
long-form biomedical answers. Our approach integrates BioBERT embeddings with
FAISS indexing and compares various re-ranking strategies (BM25, ColBERT,
MonoT5) to optimize context selection before synthesizing evidence through a
fine-tuned T5 model. Experimental results on the PubMedQA dataset show
significant improvements over baselines, with our best model achieving
substantial gains across BLEU, ROUGE, and METEOR metrics, advancing the state
of accessible, evidence-based biomedical knowledge retrieval.</details></td><td><details><summary>展开</summary>这篇论文介绍了RAG-BioQA框架，通过结合检索增强生成（RAG）和领域特定微调，生成基于证据的长篇生物医学答案，优化了上下文检索与合成，并在PubMedQA数据集上表现出显著优于基线的性能提升。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.01600v1">A Comparison of Independent and Joint Fine-tuning Strategies for Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>A Comparison of Independent and Joint Fine-tuning Strategies for
Retrieval-Augmented Generation Download PDF Neal Gregory Lawton, Alfy Samuel,
Anoop Kumar, Daben Liu Published: 20 Aug 2025, Last Modified: 17 Sept 2025EMNLP
2025 FindingsConference, Publication Chairs, AuthorsRevisionsBibTeXCC BY 4.0
Keywords: Retrieval-Augmented Generation (RAG), Large Language Models (LLMs),
Fine-tuning, Question Answering, Joint fine-tuning TL;DR: We evaluate and
compare strategies for fine-tuning Retrieval Augmented Generation (RAG)
pipelines, including independent fine-tuning, joint fine-tuning, and two-phase
fine-tuning. Abstract: Retrieval augmented generation (RAG) is a popular
framework for question answering that is powered by two large language models
(LLMs): an embedding model that retrieves context documents from a database
that are relevant to a given question, and a generator model that uses the
retrieved context to generate an answer to the question. Both the embedding and
generator models can be fine-tuned to increase performance of a RAG pipeline on
a new task, but multiple fine-tuning strategies exist with different costs and
benefits. In this paper, we evaluate and compare several RAG fine-tuning
strategies, including independent, joint, and two-phase fine-tuning. In our
experiments, we observe that all of these strategies achieve about equal
improvement in EM and F1 generation quality metrics, although they have
significantly different computational costs. We conclude the optimal
fine-tuning strategy to use depends on whether the training dataset includes
context labels and whether a grid search over the learning rates for the
embedding and generator models is required.</details></td><td><details><summary>展开</summary>这篇论文比较了检索增强生成（RAG）中不同微调策略（独立、联合和两阶段微调）的性能和计算成本，发现在生成质量上表现相近但计算代价差异显著，并指出最优策略取决于训练数据是否包含上下文标签及是否需要学习率调优。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.01558v1">CardioRAG: A Retrieval-Augmented Generation Framework for Multimodal Chagas Disease Detection</a></td><td><details><summary>展开</summary>Chagas disease affects nearly 6 million people worldwide, with Chagas
cardiomyopathy representing its most severe complication. In regions where
serological testing capacity is limited, AI-enhanced electrocardiogram (ECG)
screening provides a critical diagnostic alternative. However, existing machine
learning approaches face challenges such as limited accuracy, reliance on large
labeled datasets, and more importantly, weak integration with evidence-based
clinical diagnostic indicators. We propose a retrieval-augmented generation
framework, CardioRAG, integrating large language models with interpretable
ECG-based clinical features, including right bundle branch block, left anterior
fascicular block, and heart rate variability metrics. The framework uses
variational autoencoder-learned representations for semantic case retrieval,
providing contextual cases to guide clinical reasoning. Evaluation demonstrated
high recall performance of 89.80%, with a maximum F1 score of 0.68 for
effective identification of positive cases requiring prioritized serological
testing. CardioRAG provides an interpretable, clinical evidence-based approach
particularly valuable for resource-limited settings, demonstrating a pathway
for embedding clinical indicators into trustworthy medical AI systems.</details></td><td><details><summary>展开</summary>该论文提出了一种名为CardioRAG的检索增强生成框架，结合大语言模型和可解释的心电图临床特征（如右束支传导阻滞等），通过检索相关病例提供临床推理指导，显著提升了恰加斯病心肌病筛查的准确性和可解释性，适用于资源有限地区。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.01553v1">IoDResearch: Deep Research on Private Heterogeneous Data via the Internet of Data</a></td><td><details><summary>展开</summary>The rapid growth of multi-source, heterogeneous, and multimodal scientific
data has increasingly exposed the limitations of traditional data management.
Most existing DeepResearch (DR) efforts focus primarily on web search while
overlooking local private data. Consequently, these frameworks exhibit low
retrieval efficiency for private data and fail to comply with the FAIR
principles, ultimately resulting in inefficiency and limited reusability. To
this end, we propose IoDResearch (Internet of Data Research), a private
data-centric Deep Research framework that operationalizes the Internet of Data
paradigm. IoDResearch encapsulates heterogeneous resources as FAIR-compliant
digital objects, and further refines them into atomic knowledge units and
knowledge graphs, forming a heterogeneous graph index for multi-granularity
retrieval. On top of this representation, a multi-agent system supports both
reliable question answering and structured scientific report generation.
Furthermore, we establish the IoD DeepResearch Benchmark to systematically
evaluate both data representation and Deep Research capabilities in IoD
scenarios. Experimental results on retrieval, QA, and report-writing tasks show
that IoDResearch consistently surpasses representative RAG and Deep Research
baselines. Overall, IoDResearch demonstrates the feasibility of
private-data-centric Deep Research under the IoD paradigm, paving the way
toward more trustworthy, reusable, and automated scientific discovery.</details></td><td><details><summary>展开</summary>该论文提出IoDResearch框架，通过将异构数据封装为FAIR合规的数字对象并构建多粒度检索的异构图索引，结合多智能体系统实现可靠问答和结构化报告生成，实验表明其在检索和生成任务上优于RAG基线，属于RAG技术在私有数据场景下的优化应用。</details></td></tr></tbody></table>

### 📅 2025-10-01
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2510.01523v1">MetaSynth: Multi-Agent Metadata Generation from Implicit Feedback in Black-Box Systems</a></td><td><details><summary>展开</summary>Meta titles and descriptions strongly shape engagement in search and
recommendation platforms, yet optimizing them remains challenging. Search
engine ranking models are black box environments, explicit labels are
unavailable, and feedback such as click-through rate (CTR) arrives only
post-deployment. Existing template, LLM, and retrieval-augmented approaches
either lack diversity, hallucinate attributes, or ignore whether candidate
phrasing has historically succeeded in ranking. This leaves a gap in directly
leveraging implicit signals from observable outcomes. We introduce MetaSynth, a
multi-agent retrieval-augmented generation framework that learns from implicit
search feedback. MetaSynth builds an exemplar library from top-ranked results,
generates candidate snippets conditioned on both product content and exemplars,
and iteratively refines outputs via evaluator-generator loops that enforce
relevance, promotional strength, and compliance. On both proprietary e-commerce
data and the Amazon Reviews corpus, MetaSynth outperforms strong baselines
across NDCG, MRR, and rank metrics. Large-scale A/B tests further demonstrate
10.26% CTR and 7.51% clicks. Beyond metadata, this work contributes a general
paradigm for optimizing content in black-box systems using implicit signals.</details></td><td><details><summary>展开</summary>这篇论文介绍了MetaSynth，一个多智能体检索增强生成框架，用于优化搜索引擎的元标题和描述。它通过从排名靠前的搜索结果中构建示例库，结合产品内容和历史成功案例生成候选片段，并通过评估-生成循环迭代优化输出，以提高相关性、推广强度和合规性。实验表明，MetaSynth在多个指标上优于现有基线，并显著提升了点击率。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.01409v1">OntoLogX: Ontology-Guided Knowledge Graph Extraction from Cybersecurity Logs with Large Language Models</a></td><td><details><summary>展开</summary>System logs represent a valuable source of Cyber Threat Intelligence (CTI),
capturing attacker behaviors, exploited vulnerabilities, and traces of
malicious activity. Yet their utility is often limited by lack of structure,
semantic inconsistency, and fragmentation across devices and sessions.
Extracting actionable CTI from logs therefore requires approaches that can
reconcile noisy, heterogeneous data into coherent and interoperable
representations. We introduce OntoLogX, an autonomous Artificial Intelligence
(AI) agent that leverages Large Language Models (LLMs) to transform raw logs
into ontology-grounded Knowledge Graphs (KGs). OntoLogX integrates a
lightweight log ontology with Retrieval Augmented Generation (RAG) and
iterative correction steps, ensuring that generated KGs are syntactically and
semantically valid. Beyond event-level analysis, the system aggregates KGs into
sessions and employs a LLM to predict MITRE ATT&CK tactics, linking low-level
log evidence to higher-level adversarial objectives. We evaluate OntoLogX on
both logs from a public benchmark and a real-world honeypot dataset,
demonstrating robust KG generation across multiple KGs backends and accurate
mapping of adversarial activity to ATT&CK tactics. Results highlight the
benefits of retrieval and correction for precision and recall, the
effectiveness of code-oriented models in structured log analysis, and the value
of ontology-grounded representations for actionable CTI extraction.</details></td><td><details><summary>展开</summary>这篇论文介绍了OntoLogX，一个利用大语言模型（LLMs）将原始日志转化为基于本体的知识图谱（KGs）的自主AI代理。它结合了轻量级日志本体与检索增强生成（RAG）技术，通过迭代校正步骤确保生成的KGs在语法和语义上有效，并将日志事件关联到MITRE ATT&CK框架中的高级攻击策略，从而提升网络威胁情报（CTI）的可操作性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.01375v1">Fine-tuning with RAG for Improving LLM Learning of New Skills</a></td><td><details><summary>展开</summary>Large language model (LLM) agents deployed for multi-step tasks frequently
fail in predictable ways: attempting actions with unmet preconditions, issuing
redundant commands, or mishandling environment constraints. While
retrieval-augmented generation (RAG) can improve performance by providing
runtime guidance, it requires maintaining external knowledge databases and adds
computational overhead at every deployment. We propose a simple pipeline that
converts inference-time retrieval into learned competence through distillation.
Our approach: (1) extracts compact, reusable hints from agent failures, (2)
uses these hints to generate improved teacher trajectories via one-shot
retrieval at episode start, and (3) trains student models on these trajectories
with hint strings removed, forcing internalization rather than memorization.
Across two interactive benchmarks, ALFWorld (household tasks) and WebShop
(online shopping), distilled students consistently outperform baseline agents,
achieving up to 91% success on ALFWorld (vs. 79% for baselines) and improving
WebShop scores to 72 (vs. 61 for baselines), while using 10-60% fewer tokens
than retrieval-augmented teachers depending on the environment. The approach
generalizes across model scales (7B/14B parameters) and agent architectures
(ReAct/StateAct), demonstrating that retrieval benefits can be effectively
internalized through targeted fine-tuning without permanent runtime
dependencies.</details></td><td><details><summary>展开</summary>这篇论文提出了一种通过知识蒸馏将检索增强生成（RAG）的运行时检索转化为模型内部能力的方法，以减少对外部知识库的依赖和计算开销。该方法从智能体失败中提取紧凑的提示，生成改进的教师轨迹，并训练学生模型内部化这些知识，在ALFWorld和WebShop基准测试中表现优于基线模型。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.01363v1">Retrieval-Augmented Framework for LLM-Based Clinical Decision Support</a></td><td><details><summary>展开</summary>The increasing complexity of clinical decision-making, alongside the rapid
expansion of electronic health records (EHR), presents both opportunities and
challenges for delivering data-informed care. This paper proposes a clinical
decision support system powered by Large Language Models (LLMs) to assist
prescribing clinicians. The system generates therapeutic suggestions by
analyzing historical EHR data, including patient demographics, presenting
complaints, clinical symptoms, diagnostic information, and treatment histories.
The framework integrates natural language processing with structured clinical
inputs to produce contextually relevant recommendations. Rather than replacing
clinician judgment, it is designed to augment decision-making by retrieving and
synthesizing precedent cases with comparable characteristics, drawing on local
datasets or federated sources where applicable. At its core, the system employs
a retrieval-augmented generation (RAG) pipeline that harmonizes unstructured
narratives and codified data to support LLM-based inference. We outline the
system's technical components, including representation representation
alignment and generation strategies. Preliminary evaluations, conducted with
de-identified and synthetic clinical datasets, examine the clinical
plausibility and consistency of the model's outputs. Early findings suggest
that LLM-based tools may provide valuable decision support in prescribing
workflows when appropriately constrained and rigorously validated. This work
represents an initial step toward integration of generative AI into real-world
clinical decision-making with an emphasis on transparency, safety, and
alignment with established practices.</details></td><td><details><summary>展开</summary>这篇论文提出了一种基于大语言模型（LLMs）的临床决策支持系统，通过检索增强生成（RAG）技术整合电子健康记录（EHR）中的结构化和非结构化数据，生成治疗建议。系统利用历史病例数据检索相似案例，辅助临床医生决策，并强调透明度、安全性与临床验证。初步评估表明其在处方工作流中具有潜在价值。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.01165v1">GRAD: Generative Retrieval-Aligned Demonstration Sampler for Efficient Few-Shot Reasoning</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) achieve strong performance across diverse tasks,
but their effectiveness often depends on the quality of the provided context.
Retrieval-Augmented Generation (RAG) enriches prompts with external
information, but its reliance on static databases constrains adaptability and
can result in irrelevant demonstrations. In this work, we propose a Generative
Retrieval-Aligned Demonstrator (GRAD), a dynamic demonstration-based approach
where an LLM model is trained to generate input-specific concise
demonstrations. By tailoring demonstrations to each input, our method offers
better contextual support than traditional RAG approaches. We demonstrate the
superiority of GRAD under budget constraints, where we limit both the number of
tokens used per demonstration and the number of tokens used for the final
output. Trained solely on a math dataset, GRAD consistently outperforms strong
baselines on Qwen2.5-14B across mathematical reasoning and advanced STEM
questions, highlighting GRAD's robust generalization to out-of-distribution
(OOD) domains such as physics, chemistry, and computer science. Furthermore, we
show that demonstrations generated by trained smaller models can effectively
guide larger target models, reducing training costs while maintaining
competitive accuracy. Overall, this work introduces a scalable demonstration
generator model presenting the first step toward a dynamic few-shot learning
paradigm in resource-constrained settings. We release the code used for the
project.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为GRAD（Generative Retrieval-Aligned Demonstrator）的动态演示生成方法，通过训练大语言模型为每个输入生成特定且简洁的演示，以提供比传统RAG更精准的上下文支持。实验表明，GRAD在数学推理和STEM问题中表现优异，并能泛化到物理、化学等OOD领域，同时小模型生成的演示可有效指导大模型，降低训练成本。该研究为资源受限环境下的动态小样本学习提供了新思路。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.01115v1">Exploring Network-Knowledge Graph Duality: A Case Study in Agentic Supply Chain Risk Analysis</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) struggle with the complex, multi-modal, and
network-native data underlying financial risk. Standard Retrieval-Augmented
Generation (RAG) oversimplifies relationships, while specialist models are
costly and static. We address this gap with an LLM-centric agent framework for
supply chain risk analysis. Our core contribution is to exploit the inherent
duality between networks and knowledge graphs (KG). We treat the supply chain
network as a KG, allowing us to use structural network science principles for
retrieval. A graph traverser, guided by network centrality scores, efficiently
extracts the most economically salient risk paths. An agentic architecture
orchestrates this graph retrieval alongside data from numerical factor tables
and news streams. Crucially, it employs novel ``context shells'' -- descriptive
templates that embed raw figures in natural language -- to make quantitative
data fully intelligible to the LLM. This lightweight approach enables the model
to generate concise, explainable, and context-rich risk narratives in real-time
without costly fine-tuning or a dedicated graph database.</details></td><td><details><summary>展开</summary>该论文提出了一种基于LLM的供应链风险分析框架，通过将供应链网络视为知识图谱（KG），利用网络中心性评分指导检索，并结合数值因子表和新闻流数据，采用创新的"context shells"技术使定量数据更易被LLM理解，从而生成实时、可解释且上下文的风险分析报告，改进了传统RAG方法在金融风险领域的局限性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.00919v2">Benchmarking Foundation Models with Retrieval-Augmented Generation in Olympic-Level Physics Problem Solving</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) with foundation models has achieved
strong performance across diverse tasks, but their capacity for expert-level
reasoning-such as solving Olympiad-level physics problems-remains largely
unexplored. Inspired by the way students prepare for competitions by reviewing
past problems, we investigate the potential of RAG to enhance physics reasoning
in foundation models. We introduce PhoPile, a high-quality multimodal dataset
specifically designed for Olympiad-level physics, enabling systematic study of
retrieval-based reasoning. PhoPile includes diagrams, graphs, and equations,
capturing the inherently multimodal nature of physics problem solving. Using
PhoPile, we benchmark RAG-augmented foundation models, covering both large
language models (LLMs) and large multimodal models (LMMs) with multiple
retrievers. Our results demonstrate that integrating retrieval with physics
corpora can improve model performance, while also highlighting challenges that
motivate further research in retrieval-augmented physics reasoning.</details></td><td><details><summary>展开</summary>这篇论文探讨了RAG技术在基础模型中增强物理推理能力的潜力，特别是针对奥林匹克级物理问题的解决。作者提出了一个高质量的多模态数据集PhoPile，用于系统研究基于检索的推理，并评估了不同检索器和基础模型（包括LLMs和LMMs）的性能，结果表明检索物理语料库能提升模型表现，但也指出了未来研究的挑战。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.00880v1">HalluGuard: Evidence-Grounded Small Reasoning Models to Mitigate Hallucinations in Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) excel in many NLP tasks but remain prone to
hallucinations, limiting trust in real-world applications. We present
HalluGuard, a 4B-parameter Small Reasoning Model (SRM) for mitigating
hallucinations in Retrieval-Augmented Generation (RAG). HalluGuard classifies
document-claim pairs as grounded or hallucinated and produces evidence-grounded
justifications for transparency. Our approach combines (i) a domain-agnostic
synthetic dataset derived from FineWeb and refined through multi-stage curation
and data reformation, (ii) synthetic grounded and hallucinated claims, and
(iii) preference-based fine-tuning with Odds Ratio Preference Optimization to
distill large-model reasoning into a smaller backbone. On the RAGTruth subset
of the LLM-AggreFact benchmark, HalluGuard achieves 84.0% balanced accuracy
(BAcc), rivaling specialized models, MiniCheck (7B; 84.0%) and Granite Guardian
3.3 (8B; 82.2%) while using roughly half their parameters. Over the full
benchmark it reaches 75.7% BAcc, matching larger general-purpose LLMs such as
GPT-4o (75.9%). We will release HalluGuard and datasets under Apache 2.0 upon
acceptance.</details></td><td><details><summary>展开</summary>该论文提出了HalluGuard，一个4B参数的小型推理模型（SRM），用于缓解检索增强生成（RAG）中的幻觉问题。HalluGuard通过分类文档-声明对是否基于事实，并生成证据支持的合理解释，结合了领域无关的数据集、合成数据及偏好微调技术，在多个基准测试中表现优异，性能接近或超越更大规模的模型。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.00829v1">Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-based Machine Translation</a></td><td><details><summary>展开</summary>\textbf{RE}trieval-\textbf{A}ugmented \textbf{L}LM-based \textbf{M}achine
\textbf{T}ranslation (REAL-MT) shows promise for knowledge-intensive tasks like
idiomatic translation, but its reliability under noisy retrieval contexts
remains poorly understood despite this being a common challenge in real-world
deployment. To address this gap, we propose a noise synthesis framework and new
metrics to evaluate the robustness of REAL-MT systematically. Using this
framework, we instantiate REAL-MT with Qwen-series models, including standard
LLMs and large reasoning models (LRMs) with enhanced reasoning, and evaluate
their performance on idiomatic translation across high-, medium-, and
low-resource language pairs under synthesized noise. Our results show that
low-resource language pairs, which rely more heavily on retrieved context,
degrade more severely under noise than high-resource ones and often produce
nonsensical translations. Although LRMs possess enhanced reasoning
capabilities, they show no improvement in error correction and are even more
susceptible to noise, tending to rationalize incorrect contexts. We find that
this stems from an attention shift away from the source idiom to noisy content,
while confidence increases despite declining accuracy, indicating poor
calibration. To mitigate these issues, we investigate training-free and
fine-tuning strategies, which improve robustness at the cost of performance in
clean contexts, revealing a fundamental trade-off. Our findings highlight the
limitations of current approaches, underscoring the need for self-verifying
integration mechanisms.</details></td><td><details><summary>展开</summary>该论文研究了基于检索增强的大语言模型机器翻译（REAL-MT）在噪声检索环境下的鲁棒性，提出噪声合成框架和评估指标，发现低资源语言对和增强推理模型（LRMs）易受噪声干扰，并探讨了无训练和微调策略的改进方法，揭示了性能与鲁棒性的权衡。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.00662v1">Facilitating Cognitive Accessibility with LLMs: A Multi-Task Approach to Easy-to-Read Text Generation</a></td><td><details><summary>展开</summary>Simplifying complex texts is essential for ensuring equitable access to
information, especially for individuals with cognitive impairments. The
Easy-to-Read (ETR) initiative offers a framework for making content accessible
to the neurodivergent population, but the manual creation of such texts remains
time-consuming and resource-intensive. In this work, we investigate the
potential of large language models (LLMs) to automate the generation of ETR
content. To address the scarcity of aligned corpora and the specificity of ETR
constraints, we propose a multi-task learning (MTL) approach that trains models
jointly on text summarization, text simplification, and ETR generation. We
explore two different strategies: multi-task retrieval-augmented generation
(RAG) for in-context learning, and MTL-LoRA for parameter-efficient
fine-tuning. Our experiments with Mistral-7B and LLaMA-3-8B, based on ETR-fr, a
new high-quality dataset, demonstrate the benefits of multi-task setups over
single-task baselines across all configurations. Moreover, results show that
the RAG-based strategy enables generalization in out-of-domain settings, while
MTL-LoRA outperforms all learning strategies within in-domain configurations.</details></td><td><details><summary>展开</summary>该论文研究了利用大语言模型（LLMs）自动生成易读文本（ETR）的方法，提出了一种多任务学习（MTL）框架，结合文本摘要、文本简化和ETR生成任务，并探索了基于检索增强生成（RAG）的上下文学习策略和参数高效微调方法（MTL-LoRA）。实验表明，多任务设置优于单任务基线，RAG策略在跨领域场景中表现良好，而MTL-LoRA在领域内配置中效果最佳。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.05137v1">Demystifying deep search: a holistic evaluation with hint-free multi-hop questions and factorised metrics</a></td><td><details><summary>展开</summary>RAG (Retrieval-Augmented Generation) systems and web agents are increasingly
evaluated on multi-hop deep search tasks, yet current practice suffers from two
major limitations. First, most benchmarks leak the reasoning path in the
question text, allowing models to follow surface cues rather than discover
reasoning chains autonomously. Second, evaluation is typically reduced to a
single pass rate, which collapses diverse behaviours into one score and
obscures whether failures stem from inadequate search, poor knowledge use, or
inappropriate refusal. To address these issues, we present WebDetective, a
benchmark of hint-free multi-hop questions paired with a controlled Wikipedia
sandbox that ensures full traceability of model actions, and a holistic
evaluation framework that separates search sufficiency, knowledge utilisation,
and refusal behaviour. Our evaluation of 25 state-of-the-art models reveals
systematic weaknesses across all architectures: models struggle with knowledge
utilisation despite having sufficient evidence and demonstrate near-absent
appropriate refusal when evidence is lacking. These patterns expose a
fundamental gap: today's systems excel at executing given reasoning paths but
fail when required to discover them. We develop an agentic workflow,
EvidenceLoop, that explicitly targets the challenges our benchmark identifies,
incorporating verification loops and systematic evidence tracking that improve
both search and synthesis capabilities. This baseline demonstrates that
WebDetective's diagnostic framework can guide concrete architectural
improvements, establishing our benchmark as a critical tool for developing
genuinely autonomous reasoning systems rather than pattern-following agents.</details></td><td><details><summary>展开</summary>这篇论文提出了WebDetective，一个针对RAG系统和网页代理的多跳深度搜索任务的基准测试和评估框架，旨在解决现有评估方法中推理路径泄露和单一评分的问题，并揭示模型在知识利用和拒绝行为方面的系统性弱点，最终提出了一个改进的代理工作流程EvidenceLoop来提升搜索和合成能力。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.00586v1">Eyes-on-Me: Scalable RAG Poisoning through Transferable Attention-Steering Attractors</a></td><td><details><summary>展开</summary>Existing data poisoning attacks on retrieval-augmented generation (RAG)
systems scale poorly because they require costly optimization of poisoned
documents for each target phrase. We introduce Eyes-on-Me, a modular attack
that decomposes an adversarial document into reusable Attention Attractors and
Focus Regions. Attractors are optimized to direct attention to the Focus
Region. Attackers can then insert semantic baits for the retriever or malicious
instructions for the generator, adapting to new targets at near zero cost. This
is achieved by steering a small subset of attention heads that we empirically
identify as strongly correlated with attack success. Across 18 end-to-end RAG
settings (3 datasets $\times$ 2 retrievers $\times$ 3 generators), Eyes-on-Me
raises average attack success rates from 21.9 to 57.8 (+35.9 points,
2.6$\times$ over prior work). A single optimized attractor transfers to unseen
black box retrievers and generators without retraining. Our findings establish
a scalable paradigm for RAG data poisoning and show that modular, reusable
components pose a practical threat to modern AI systems. They also reveal a
strong link between attention concentration and model outputs, informing
interpretability research.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为Eyes-on-Me的新型数据投毒攻击方法，针对检索增强生成（RAG）系统。该方法通过将对抗性文档分解为可重用的“注意力吸引器”和“焦点区域”，显著提高了攻击效率，无需针对每个目标短语进行昂贵的优化。实验表明，该方法在多种RAG设置下将攻击成功率从21.9%提升至57.8%，并揭示了注意力集中与模型输出之间的强关联，为RAG系统的安全性和可解释性研究提供了新见解。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.00566v2">Panorama: Fast-Track Nearest Neighbors</a></td><td><details><summary>展开</summary>Approximate Nearest-Neighbor Search (ANNS) efficiently finds data items whose
embeddings are close to that of a given query in a high-dimensional space,
aiming to balance accuracy with speed. Used in recommendation systems, image
and video retrieval, natural language processing, and retrieval-augmented
generation (RAG), ANNS algorithms such as IVFPQ, HNSW graphs, Annoy, and MRPT
utilize graph, tree, clustering, and quantization techniques to navigate large
vector spaces. Despite this progress, ANNS systems spend up to 99\% of query
time to compute distances in their final refinement phase. In this paper, we
present PANORAMA, a machine learning-driven approach that tackles the ANNS
verification bottleneck through data-adaptive learned orthogonal transforms
that facilitate the accretive refinement of distance bounds. Such transforms
compact over 90\% of signal energy into the first half of dimensions, enabling
early candidate pruning with partial distance computations. We integrate
PANORAMA into state-of-the-art ANNS methods, namely IVFPQ/Flat, HNSW, MRPT, and
Annoy, without index modification, using level-major memory layouts,
SIMD-vectorized partial distance computations, and cache-aware access patterns.
Experiments across diverse datasets -- from image-based CIFAR-10 and GIST to
modern embedding spaces including OpenAI's Ada 2 and Large 3 -- demonstrate
that PANORAMA affords a 2--30$\times$ end-to-end speedup with no recall loss.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为PANORAMA的机器学习驱动方法，通过数据自适应的学习正交变换优化近似最近邻搜索（ANNS）的验证瓶颈，显著提升检索效率，并特别提到ANNS在检索增强生成（RAG）等领域的应用。该方法在不修改索引的情况下集成到现有ANNS算法中，实现了2-30倍的端到端加速且不损失召回率。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.00552v1">Data Quality Challenges in Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Organizations increasingly adopt Retrieval-Augmented Generation (RAG) to
enhance Large Language Models with enterprise-specific knowledge. However,
current data quality (DQ) frameworks have been primarily developed for static
datasets, and only inadequately address the dynamic, multi-stage nature of RAG
systems. This study aims to develop DQ dimensions for this new type of AI-based
systems. We conduct 16 semi-structured interviews with practitioners of leading
IT service companies. Through a qualitative content analysis, we inductively
derive 15 distinct DQ dimensions across the four processing stages of RAG
systems: data extraction, data transformation, prompt & search, and generation.
Our findings reveal that (1) new dimensions have to be added to traditional DQ
frameworks to also cover RAG contexts; (2) these new dimensions are
concentrated in early RAG steps, suggesting the need for front-loaded quality
management strategies, and (3) DQ issues transform and propagate through the
RAG pipeline, necessitating a dynamic, step-aware approach to quality
management.</details></td><td><details><summary>展开</summary>这篇论文探讨了在检索增强生成（RAG）系统中数据质量（DQ）维度的开发，通过访谈IT服务公司的实践者，归纳出15个DQ维度，覆盖RAG系统的四个处理阶段，并指出需要更新传统DQ框架以适应RAG的动态特性，强调早期阶段的质量管理重要性及问题在流程中的传递性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.00508v1">Copy-Paste to Mitigate Large Language Model Hallucinations</a></td><td><details><summary>展开</summary>While Retrieval-Augmented Generation (RAG) enables large language models
(LLMs) to generate contextually grounded responses, contextual faithfulness
remains challenging as LLMs may not consistently trust provided context,
leading to hallucinations that undermine reliability. We observe an inverse
correlation between response copying degree and context-unfaithful
hallucinations on RAGTruth, suggesting that higher copying degrees reduce
hallucinations by fostering genuine contextual belief. We propose CopyPasteLLM,
obtained through two-stage high-copying response preference training. We design
three prompting methods to enhance copying degree, demonstrating that
high-copying responses achieve superior contextual faithfulness and
hallucination control. These approaches enable a fully automated pipeline that
transforms generated responses into high-copying preference data for training
CopyPasteLLM. On FaithEval, ConFiQA and PubMedQA, CopyPasteLLM achieves best
performance in both counterfactual and original contexts, remarkably with 12.2%
to 24.5% accuracy improvements on FaithEval over the best baseline, while
requiring only 365 training samples -- 1/50th of baseline data. To elucidate
CopyPasteLLM's effectiveness, we propose the Context-Parameter Copying
Capturing algorithm. Interestingly, this reveals that CopyPasteLLM recalibrates
reliance on internal parametric knowledge rather than external knowledge during
generation. All codes are available at
https://github.com/longyongchao/CopyPasteLLM</details></td><td><details><summary>展开</summary>该论文针对RAG中LLMs对检索内容信任不足导致的幻觉问题，提出CopyPasteLLM模型，通过高复制度响应训练和提示方法增强上下文忠实度，显著降低幻觉并提升准确性，在多个基准测试中表现优异。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2510.00482v1">Agent Fine-tuning through Distillation for Domain-specific LLMs in Microdomains</a></td><td><details><summary>展开</summary>Agentic large language models (LLMs) have become prominent for autonomously
interacting with external environments and performing multi-step reasoning
tasks. Most approaches leverage these capabilities via in-context learning with
few-shot prompts, but this often results in lengthy inputs and higher
computational costs. Agent fine-tuning offers an alternative by enabling LLMs
to internalize procedural reasoning and domain-specific knowledge through
training on relevant data and demonstration trajectories. While prior studies
have focused on general domains, their effectiveness in specialized technical
microdomains remains unclear. This paper explores agent fine-tuning for domain
adaptation within Hitachi's JP1 middleware, a microdomain for specialized IT
operations. We fine-tuned LLMs using JP1-specific datasets derived from domain
manuals and distilled reasoning trajectories generated by LLMs themselves,
enhancing decision making accuracy and search efficiency. During inference, we
used an agentic prompt with retrieval-augmented generation and introduced a
context-answer extractor to improve information relevance. On JP1 certification
exam questions, our method achieved a 14% performance improvement over the base
model, demonstrating the potential of agent fine-tuning for domain-specific
reasoning in complex microdomains.</details></td><td><details><summary>展开</summary>该论文探讨了在日立JP1中间件这一特定技术微领域中，通过微调大型语言模型（LLMs）以提升领域适应性的方法，其中在推理阶段采用了检索增强生成（RAG）技术和上下文-答案提取器来提高信息的相关性，最终在JP1认证考试问题上实现了14%的性能提升。</details></td></tr></tbody></table>

### 📅 2025-09-30
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.26584v1">Fairness Testing in Retrieval-Augmented Generation: How Small Perturbations Reveal Bias in Small Language Models</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) are widely used across multiple domains but
continue to raise concerns regarding security and fairness. Beyond known attack
vectors such as data poisoning and prompt injection, LLMs are also vulnerable
to fairness bugs. These refer to unintended behaviors influenced by sensitive
demographic cues (e.g., race or sexual orientation) that should not affect
outcomes. Another key issue is hallucination, where models generate plausible
yet false information. Retrieval-Augmented Generation (RAG) has emerged as a
strategy to mitigate hallucinations by combining external retrieval with text
generation. However, its adoption raises new fairness concerns, as the
retrieved content itself may surface or amplify bias. This study conducts
fairness testing through metamorphic testing (MT), introducing controlled
demographic perturbations in prompts to assess fairness in sentiment analysis
performed by three Small Language Models (SLMs) hosted on HuggingFace
(Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3, and Llama-3.1-Nemotron-8B),
each integrated into a RAG pipeline. Results show that minor demographic
variations can break up to one third of metamorphic relations (MRs). A detailed
analysis of these failures reveals a consistent bias hierarchy, with
perturbations involving racial cues being the predominant cause of the
violations. In addition to offering a comparative evaluation, this work
reinforces that the retrieval component in RAG must be carefully curated to
prevent bias amplification. The findings serve as a practical alert for
developers, testers and small organizations aiming to adopt accessible SLMs
without compromising fairness or reliability.</details></td><td><details><summary>展开</summary>这篇论文探讨了RAG技术中的公平性问题，通过蜕变测试评估小型语言模型在RAG流程中对敏感人口统计线索的偏差表现，并揭示检索内容可能加剧偏见的现象，提出需谨慎处理检索组件以避免偏见放大。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.26383v1">Efficient and Transferable Agentic Knowledge Graph RAG via Reinforcement Learning</a></td><td><details><summary>展开</summary>Knowledge-graph retrieval-augmented generation (KG-RAG) couples large
language models (LLMs) with structured, verifiable knowledge graphs (KGs) to
reduce hallucinations and expose reasoning traces. However, many KG-RAG systems
compose multiple LLM modules (e.g planning, reasoning, and responding),
inflating inference cost and binding behavior to a specific target KG. To
address this, we introduce KG-R1, an agentic KG retrieval-augmented generation
(KG-RAG) framework through reinforcement learning (RL). KG-R1 utilizes a single
agent that interacts with KGs as its environment, learning to retrieve at each
step and incorporating the retrieved information into its reasoning and
generation. The process is optimized through end-to-end RL. In controlled
experiments across Knowledge-Graph Question Answering (KGQA) benchmarks, our
method demonstrates both efficiency and transferability: Using Qwen-2.5-3B,
KG-R1 improves answer accuracy with fewer generation tokens than prior
multi-module workflow methods that use larger foundation or fine-tuned models.
Furthermore, KG-R1 enables plug and play: after training, it maintains strong
accuracy on new KGs without modification. These properties make KG-R1 a
promising KG-RAG framework for real-world deployment. Our code is publicly
available at https://github.com/Jinyeop3110/KG-R1.</details></td><td><details><summary>展开</summary>该论文提出了一种基于强化学习的知识图谱检索增强生成框架KG-R1，通过单智能体与知识图谱交互，优化检索和生成过程，在降低推理成本的同时提高准确性和可迁移性，并在KGQA基准测试中验证了其高效性和即插即用能力。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.26205v1">Human-Centered Evaluation of RAG outputs: a framework and questionnaire for human-AI collaboration</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) systems are increasingly deployed in
user-facing applications, yet systematic, human-centered evaluation of their
outputs remains underexplored. Building on Gienapp's utility-dimension
framework, we designed a human-centred questionnaire that assesses RAG outputs
across 12 dimensions. We iteratively refined the questionnaire through several
rounds of ratings on a set of query-output pairs and semantic discussions.
Ultimately, we incorporated feedback from both a human rater and a human-LLM
pair. Results indicate that while large language models (LLMs) reliably focus
on metric descriptions and scale labels, they exhibit weaknesses in detecting
textual format variations. Humans struggled to focus strictly on metric
descriptions and labels. LLM ratings and explanations were viewed as a helpful
support, but numeric LLM and human ratings lacked agreement. The final
questionnaire extends the initial framework by focusing on user intent, text
structuring, and information verifiability.</details></td><td><details><summary>展开</summary>该论文探讨了以用户为中心的RAG系统评估方法，通过设计包含12个维度的问卷，结合人类与LLM的反馈迭代优化，发现LLM在文本格式识别上的不足及人机评分差异，最终扩展了评估框架以重点关注用户意图、文本结构和信息可验证性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.26184v1">Auto-ARGUE: LLM-Based Report Generation Evaluation</a></td><td><details><summary>展开</summary>Generation of long-form, citation-backed reports is a primary use case for
retrieval augmented generation (RAG) systems. While open-source evaluation
tools exist for various RAG tasks, ones tailored to report generation are
lacking. Accordingly, we introduce Auto-ARGUE, a robust LLM-based
implementation of the recent ARGUE framework for report generation evaluation.
We present analysis of Auto-ARGUE on the report generation pilot task from the
TREC 2024 NeuCLIR track, showing good system-level correlations with human
judgments. We further release a web app for visualization of Auto-ARGUE
outputs.</details></td><td><details><summary>展开</summary>这篇论文介绍了Auto-ARGUE，一个基于大语言模型的工具，用于评估检索增强生成（RAG）系统在生成带引用的长篇报告任务中的性能，并展示了其在TREC 2024 NeuCLIR任务上与人类评价的良好相关性，同时发布了可视化输出的网页应用。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.26136v1">CliniBench: A Clinical Outcome Prediction Benchmark for Generative and Encoder-Based Language Models</a></td><td><details><summary>展开</summary>With their growing capabilities, generative large language models (LLMs) are
being increasingly investigated for complex medical tasks. However, their
effectiveness in real-world clinical applications remains underexplored. To
address this, we present CliniBench, the first benchmark that enables
comparability of well-studied encoder-based classifiers and generative LLMs for
discharge diagnosis prediction from admission notes in MIMIC-IV dataset. Our
extensive study compares 12 generative LLMs and 3 encoder-based classifiers and
demonstrates that encoder-based classifiers consistently outperform generative
models in diagnosis prediction. We assess several retrieval augmentation
strategies for in-context learning from similar patients and find that they
provide notable performance improvements for generative LLMs.</details></td><td><details><summary>展开</summary>这篇文章介绍了CliniBench，一个用于比较基于编码器的分类器和生成式大语言模型在MIMIC-IV数据集出院诊断预测任务中表现的基准测试，研究发现编码器模型表现更优，并通过检索增强策略提升了生成模型的性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.26011v1">RAGferee: Building Contextual Reward Models for Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Existing Reward Models (RMs), typically trained on general preference data,
struggle in Retrieval Augmented Generation (RAG) settings, which require
judging responses for faithfulness to retrieved context, relevance to the user
query, appropriate refusals when context is insufficient, completeness and
conciseness of information. To address the lack of publicly available
RAG-centric preference datasets and specialised RMs, we introduce RAGferee, a
methodology that repurposes question-answering (QA) datasets into preference
pairs that prioritise groundedness over stylistic features, enabling the
training of contextual RMs better suited to judging RAG responses. Using
RAGferee, we curate a small preference dataset of 4K samples and fine-tune RMs
ranging from 7B to 24B parameters. Our RAG-centric RMs achieve state-of-the-art
performance on ContextualJudgeBench, surpassing existing 70B+ RMs trained on
much larger (up to 2.4M samples) general corpora, with an absolute improvement
of +15.5%.</details></td><td><details><summary>展开</summary>该论文针对现有奖励模型（RMs）在RAG场景中的不足（如对检索内容忠实度、查询相关性、信息完整性的评估），提出RAGferee方法，将问答数据集转换为优先考虑事实准确性的偏好数据，并训练出专用于RAG响应的RM，其性能在ContextualJudgeBench上超越通用大型RM（+15.5%）。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.25973v1">Scalable and Robust LLM Unlearning by Correcting Responses with Retrieved Exclusions</a></td><td><details><summary>展开</summary>Language models trained on web-scale corpora risk memorizing and exposing
sensitive information, prompting the need for effective machine unlearning.
Prior methods mainly focus on input queries to suppress sensitive outputs, yet
this often fails to eliminate the underlying knowledge and limits scalability.
To address this, we propose Corrective Unlearning with Retrieved Exclusions
(CURE), a novel unlearning framework that verifies model outputs for leakage
and revises them into safe responses. Specifically, CURE employs a lightweight
corrector that is applied to the original model to verify whether outputs
contain target knowledge and to rewrite them if any leakage is detected. To
efficiently handle large-scale unlearning requests, CURE retrieves unlearning
targets that are relevant to the initial response and provides them as
in-context references to the corrector for detection and conditional revision.
By leveraging this retrieval augmentation, the corrector can adapt to new
unlearning requests without additional training. Extensive evaluations
demonstrate that CURE substantially reduces information leakage, even from
indirect queries where prior works fall short, while maintaining response
quality and general utility. Moreover, it demonstrates robustness under
continual unlearning scenarios, making it practical for real-world
applications.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为CURE的机器遗忘框架，通过检索增强生成技术检测和修正模型输出中的敏感信息泄漏。它利用轻量级校正器结合检索到的相关遗忘目标，动态调整输出以确保安全，同时保持模型性能和扩展性，适用于大规模持续遗忘场景。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.25839v1">RAE: A Neural Network Dimensionality Reduction Method for Nearest Neighbors Preservation in Vector Search</a></td><td><details><summary>展开</summary>While high-dimensional embedding vectors are being increasingly employed in
various tasks like Retrieval-Augmented Generation and Recommendation Systems,
popular dimensionality reduction (DR) methods such as PCA and UMAP have rarely
been adopted for accelerating the retrieval process due to their inability of
preserving the nearest neighbor (NN) relationship among vectors. Empowered by
neural networks' optimization capability and the bounding effect of Rayleigh
quotient, we propose a Regularized Auto-Encoder (RAE) for k-NN preserving
dimensionality reduction. RAE constrains the network parameter variation
through regularization terms, adjusting singular values to control embedding
magnitude changes during reduction, thus preserving k-NN relationships. We
provide a rigorous mathematical analysis demonstrating that regularization
establishes an upper bound on the norm distortion rate of transformed vectors,
thereby offering provable guarantees for k-NN preservation. With modest
training overhead, RAE achieves superior k-NN recall compared to existing DR
approaches while maintaining fast retrieval efficiency.</details></td><td><details><summary>展开</summary>这篇论文提出了一种用于保持k最近邻（k-NN）关系的正则化自动编码器（RAE），旨在解决高维嵌入向量在检索增强生成（RAG）等任务中因维度缩减导致的最近邻关系破坏问题，从而提升检索效率并保证准确性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.25736v1">Think Less, Label Better: Multi-Stage Domain-Grounded Synthetic Data Generation for Fine-Tuning Large Language Models in Telecommunications</a></td><td><details><summary>展开</summary>The success of large language models (LLMs) depends heavily on large-scale,
high-quality instruction-following and reinforcement datasets. However,
generating such data through human annotation is prohibitively time-consuming
particularly for domain-specific tasks like telecom network troubleshooting,
where accurate responses require deep technical expertise and contextual
understanding. In this paper, we present a fully automated, retrieval-augmented
pipeline for generating synthetic question-answer (QA) pairs grounded in
structured domain knowledge. Our multi-stage framework integrates a retriever,
base generator, and refinement model to synthesize and enhance QA pairs using
documents retrieved from a domain-specific knowledge graph. To ensure data
quality, we employ customized RAGAS-based scoring to filter low-quality
samples, producing a high-quality dataset suitable for reinforcement
fine-tuning (RFT). We demonstrate our approach in a real-world telecom scenario
focused on radio access network (RAN) troubleshooting. The resulting pipeline
generates complex, context-rich troubleshooting solution plans without human
intervention. This work offers a scalable solution for building instruction and
reinforcement datasets in specialized domains, significantly reducing
dependence on manual labeling while maintaining high technical fidelity.</details></td><td><details><summary>展开</summary>本文提出了一种全自动、检索增强的流程，用于生成基于结构化领域知识的合成问答对（QA），通过结合检索器、基础生成器和精炼模型的多阶段框架，从领域特定知识图谱中检索文档并生成高质量QA数据集，应用于电信网络故障排除等专业领域。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.25716v1">DeepCodeSeek: Real-Time API Retrieval for Context-Aware Code Generation</a></td><td><details><summary>展开</summary>Current search techniques are limited to standard RAG query-document
applications. In this paper, we propose a novel technique to expand the code
and index for predicting the required APIs, directly enabling high-quality,
end-to-end code generation for auto-completion and agentic AI applications. We
address the problem of API leaks in current code-to-code benchmark datasets by
introducing a new dataset built from real-world ServiceNow Script Includes that
capture the challenge of unclear API usage intent in the code. Our evaluation
metrics show that this method achieves 87.86% top-40 retrieval accuracy,
allowing the critical context with APIs needed for successful downstream code
generation. To enable real-time predictions, we develop a comprehensive
post-training pipeline that optimizes a compact 0.6B reranker through synthetic
dataset generation, supervised fine-tuning, and reinforcement learning. This
approach enables our compact reranker to outperform a much larger 8B model
while maintaining 2.5x reduced latency, effectively addressing the nuances of
enterprise-specific code without the computational overhead of larger models.</details></td><td><details><summary>展开</summary>这篇论文提出了一种扩展RAG技术和索引的新方法，专注于通过检索预测所需API以实现高质量的端到端代码生成，解决了当前代码基准数据集中API泄露问题，并通过优化的后训练流程提升实时预测性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.25669v1">GroundSight: Augmenting Vision-Language Models with Grounding Information and De-hallucination</a></td><td><details><summary>展开</summary>We propose a method to improve Visual Question Answering (VQA) with
Retrieval-Augmented Generation (RAG) by introducing text-grounded object
localization. Rather than retrieving information based on the entire image, our
approach enables the model to generate a bounding box around the object most
relevant to the question, allowing for targeted image cropping and focused
retrieval. This reduces background noise, improves alignment between visual and
textual cues, and helps mitigate hallucinations. Our RAG method enhances
context-aware VQA responses increased the accuracy from 22.19% to 25.64%, with
an absolute increase of 3.45 percentage points, compared to the baseline
Llama-3.2-Vision-11B agent. We also proposed a de-hallucination method based on
question type which can effectively reduce the hallucination rate from 65.79%
to 13.88% and improves the truthfulness score.</details></td><td><details><summary>展开</summary>该论文提出了一种改进视觉问答（VQA）的方法，通过结合RAG技术和基于文本的物体定位，模型能够生成与问题最相关物体的边界框，从而进行针对性图像裁剪和聚焦检索。此方法减少了背景噪声，提升了视觉与文本线索的对齐，并降低了幻觉现象。实验表明，该RAG方法将VQA准确率从22.19%提升至25.64%，并提出了基于问题类型的去幻觉方法，将幻觉率从65.79%降低至13.88%。</details></td></tr></tbody></table>

### 📅 2025-09-29
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.25143v1">TemMed-Bench: Evaluating Temporal Medical Image Reasoning in Vision-Language Models</a></td><td><details><summary>展开</summary>Existing medical reasoning benchmarks for vision-language models primarily
focus on analyzing a patient's condition based on an image from a single visit.
However, this setting deviates significantly from real-world clinical practice,
where doctors typically refer to a patient's historical conditions to provide a
comprehensive assessment by tracking their changes over time. In this paper, we
introduce TemMed-Bench, the first benchmark designed for analyzing changes in
patients' conditions between different clinical visits, which challenges large
vision-language models (LVLMs) to reason over temporal medical images.
TemMed-Bench consists of a test set comprising three tasks - visual
question-answering (VQA), report generation, and image-pair selection - and a
supplementary knowledge corpus of over 17,000 instances. With TemMed-Bench, we
conduct an evaluation of six proprietary and six open-source LVLMs. Our results
show that most LVLMs lack the ability to analyze patients' condition changes
over temporal medical images, and a large proportion perform only at a
random-guessing level in the closed-book setting. In contrast, GPT o3, o4-mini
and Claude 3.5 Sonnet demonstrate comparatively decent performance, though they
have yet to reach the desired level. Furthermore, we explore augmenting the
input with both retrieved visual and textual modalities in the medical domain.
We also show that multi-modal retrieval augmentation yields notably higher
performance gains than no retrieval and textual retrieval alone across most
models on our benchmark, with the VQA task showing an average improvement of
2.59%. Overall, we compose a benchmark grounded on real-world clinical
practice, and it reveals LVLMs' limitations in temporal medical image
reasoning, as well as highlighting the use of multi-modal retrieval
augmentation as a potentially promising direction worth exploring to address
this challenge.</details></td><td><details><summary>展开</summary>这篇论文介绍了TemMed-Bench，一个用于评估大视觉语言模型（LVLMs）在时间性医学图像推理中分析患者病情变化能力的基准测试。研究揭示了现有模型在此任务上的局限性，并探讨了通过多模态检索增强（结合视觉和文本检索）提升模型性能的方法，证明了其在视觉问答等任务中的有效性（平均提升2.59%），表明多模态检索增强是解决这一挑战的潜在方向。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.24869v1">Retro*: Optimizing LLMs for Reasoning-Intensive Document Retrieval</a></td><td><details><summary>展开</summary>With the growing popularity of LLM agents and RAG, it has become increasingly
important to retrieve documents that are essential for solving a task, even
when their connection to the task is indirect or implicit. Addressing this
problem requires fine-grained reasoning to accurately assess the relevance
between the task and each candidate document. This capability, however, poses a
significant challenge for existing IR techniques. Despite recent progress in
reasoning-enhanced IR, existing approaches still face significant challenges in
applicability, scalability, and efficiency. In this work, we propose Retro*, a
novel approach for reasoning-intensive document retrieval. Our method
introduces a rubric-based relevance scoring mechanism, enabling the model to
reason about the relationship between a task and a document based on explicitly
defined criteria, whereby producing a fine-grained, interpretable relevance
score. Retro* also supports test-time scaling by combining multiple reasoning
trajectories via score integration, which produces more reliable relevance
estimates. To optimize Retro*'s reasoning capabilities, we introduce a novel
reinforcement learning algorithm tailored for its relevance scoring mechanism,
which employs two composite rewards to fully exploit the trajectories of each
training sample. Our experiments show that Retro* outperforms existing document
retrieval methods with notable advantages, leading to state-of-the-art
performance on the BRIGHT benchmark.</details></td><td><details><summary>展开</summary>这篇论文提出了Retro*方法，通过细粒度推理和基于标准的评分机制来改进文档检索，解决RAG中因任务与文档间接关联导致的检索难题，并在BRIGHT基准测试中达到了最先进的性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.24866v1">Metaphor identification using large language models: A comparison of RAG, prompt engineering, and fine-tuning</a></td><td><details><summary>展开</summary>Metaphor is a pervasive feature of discourse and a powerful lens for
examining cognition, emotion, and ideology. Large-scale analysis, however, has
been constrained by the need for manual annotation due to the context-sensitive
nature of metaphor. This study investigates the potential of large language
models (LLMs) to automate metaphor identification in full texts. We compare
three methods: (i) retrieval-augmented generation (RAG), where the model is
provided with a codebook and instructed to annotate texts based on its rules
and examples; (ii) prompt engineering, where we design task-specific verbal
instructions; and (iii) fine-tuning, where the model is trained on hand-coded
texts to optimize performance. Within prompt engineering, we test zero-shot,
few-shot, and chain-of-thought strategies. Our results show that
state-of-the-art closed-source LLMs can achieve high accuracy, with fine-tuning
yielding a median F1 score of 0.79. A comparison of human and LLM outputs
reveals that most discrepancies are systematic, reflecting well-known grey
areas and conceptual challenges in metaphor theory. We propose that LLMs can be
used to at least partly automate metaphor identification and can serve as a
testbed for developing and refining metaphor identification protocols and the
theory that underpins them.</details></td><td><details><summary>展开</summary>该论文研究了利用大语言模型（LLMs）自动化识别文本中隐喻的三种方法，包括检索增强生成（RAG）、提示工程和微调，并发现RAG结合代码书规则与示例的方法能有效提升隐喻标注的准确性，同时揭示了模型与人类标注差异的系统性理论根源。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.24276v1">G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge</a></td><td><details><summary>展开</summary>Large language models (LLMs) excel at complex reasoning but remain limited by
static and incomplete parametric knowledge. Retrieval-augmented generation
(RAG) mitigates this by incorporating external knowledge, yet existing RAGs
struggle with knowledge-intensive tasks due to fragmented information and weak
modeling of knowledge structure. Graphs offer a natural way to model
relationships within knowledge, but LLMs are inherently unstructured and cannot
effectively reason over graph-structured data. Recent graph-enhanced RAG
(GraphRAG) attempts to bridge this gap by constructing tailored graphs and
enabling LLMs to reason on them. However, these methods often depend on ad-hoc
graph designs, heuristic search, or costly agent pipelines, which hinder
scalability and generalization. To address these challenges, we present
G-reasoner, a unified framework that integrates graph and language foundation
models for reasoning over diverse graph-structured knowledge. Central to our
approach is QuadGraph, a standardized four-layer abstraction that unifies
heterogeneous knowledge sources into a common graph representation. Building on
this, we introduce a 34M-parameter graph foundation model (GFM) that jointly
captures graph topology and textual semantics, and is integrated with LLMs to
enhance reasoning in downstream applications. To ensure scalability and
efficiency, mixed-precision training and distributed message-passing are
implemented to scale GFM with more GPUs. Extensive experiments on six
benchmarks show that G-reasoner consistently outperforms state-of-the-art
baselines, significantly enhances LLM reasoning, and achieves strong efficiency
and cross-graph generalization.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为G-reasoner的统一框架，通过结合图结构和语言基础模型（如QuadGraph标准化抽象层和图基础模型GFM），改进现有RAG在知识密集型任务中的局限性（如信息碎片化和知识结构建模薄弱），并实验证明了其在增强大语言模型推理能力、效率及跨图泛化性方面的优越性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.24253v1">MRAG-Suite: A Diagnostic Evaluation Platform for Visual Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Multimodal Retrieval-Augmented Generation (Visual RAG) significantly advances
question answering by integrating visual and textual evidence. Yet, current
evaluations fail to systematically account for query difficulty and ambiguity.
We propose MRAG-Suite, a diagnostic evaluation platform integrating diverse
multimodal benchmarks (WebQA, Chart-RAG, Visual-RAG, MRAG-Bench). We introduce
difficulty-based and ambiguity-aware filtering strategies, alongside
MM-RAGChecker, a claim-level diagnostic tool. Our results demonstrate
substantial accuracy reductions under difficult and ambiguous queries,
highlighting prevalent hallucinations. MM-RAGChecker effectively diagnoses
these issues, guiding future improvements in Visual RAG systems.</details></td><td><details><summary>展开</summary>该论文提出了MRAG-Suite，一个针对多模态检索增强生成（Visual RAG）的诊断评估平台，通过整合多种多模态基准和引入基于难度及模糊性的过滤策略，揭示了现有系统在面对困难和模糊查询时的准确率下降问题，并提供了诊断工具MM-RAGChecker以指导未来改进。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.24212v1">ScenarioBench: Trace-Grounded Compliance Evaluation for Text-to-SQL and RAG</a></td><td><details><summary>展开</summary>ScenarioBench is a policy-grounded, trace-aware benchmark for evaluating
Text-to-SQL and retrieval-augmented generation in compliance contexts. Each
YAML scenario includes a no-peek gold-standard package with the expected
decision, a minimal witness trace, the governing clause set, and the canonical
SQL, enabling end-to-end scoring of both what a system decides and why. Systems
must justify outputs using clause IDs from the same policy canon, making
explanations falsifiable and audit-ready. The evaluator reports decision
accuracy, trace quality (completeness, correctness, order), retrieval
effectiveness, SQL correctness via result-set equivalence, policy coverage,
latency, and an explanation-hallucination rate. A normalized Scenario
Difficulty Index (SDI) and a budgeted variant (SDI-R) aggregate results while
accounting for retrieval difficulty and time. Compared with prior Text-to-SQL
or KILT/RAG benchmarks, ScenarioBench ties each decision to clause-level
evidence under strict grounding and no-peek rules, shifting gains toward
justification quality under explicit time budgets.</details></td><td><details><summary>展开</summary>ScenarioBench是一个用于评估文本到SQL（Text-to-SQL）和检索增强生成（RAG）在合规场景下的基准测试工具，它通过YAML场景整合了决策依据、追踪信息、条款集和标准SQL，支持端到端评分，并强调输出的可验证性与审计就绪性，同时提供多维度评估指标和难度指数。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.24183v1">Retrieval-augmented GUI Agents with Generative Guidelines</a></td><td><details><summary>展开</summary>GUI agents powered by vision-language models (VLMs) show promise in
automating complex digital tasks. However, their effectiveness in real-world
applications is often limited by scarce training data and the inherent
complexity of these tasks, which frequently require long-tailed knowledge
covering rare, unseen scenarios. We propose RAG-GUI , a lightweight VLM that
leverages web tutorials at inference time. RAG-GUI is first warm-started via
supervised finetuning (SFT) and further refined through self-guided rejection
sampling finetuning (RSF). Designed to be model-agnostic, RAG-GUI functions as
a generic plug-in that enhances any VLM-based agent. Evaluated across three
distinct tasks, it consistently outperforms baseline agents and surpasses other
inference baselines by 2.6% to 13.3% across two model sizes, demonstrating
strong generalization and practical plug-and-play capabilities in real-world
scenarios.</details></td><td><details><summary>展开</summary>这篇论文提出了RAG-GUI，一种基于视觉语言模型（VLM）的轻量级GUI代理，通过利用网页教程作为检索增强的推理资源来解决复杂数字任务中训练数据稀缺和长尾知识不足的问题。该方法通过监督微调和自引导拒绝采样微调优化模型，展现出强泛化能力和即插即用特性，在多项任务中优于基线模型。</details></td></tr></tbody></table>

### 📅 2025-09-28
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.23874v1">Multi-Value-Product Retrieval-Augmented Generation for Industrial Product Attribute Value Identification</a></td><td><details><summary>展开</summary>Identifying attribute values from product profiles is a key task for
improving product search, recommendation, and business analytics on e-commerce
platforms, which we called Product Attribute Value Identification (PAVI) .
However, existing PAVI methods face critical challenges, such as cascading
errors, inability to handle out-of-distribution (OOD) attribute values, and
lack of generalization capability. To address these limitations, we introduce
Multi-Value-Product Retrieval-Augmented Generation (MVP-RAG), combining the
strengths of retrieval, generation, and classification paradigms. MVP-RAG
defines PAVI as a retrieval-generation task, where the product title
description serves as the query, and products and attribute values act as the
corpus. It first retrieves similar products of the same category and candidate
attribute values, and then generates the standardized attribute values. The key
advantages of this work are: (1) the proposal of a multi-level retrieval
scheme, with products and attribute values as distinct hierarchical levels in
PAVI domain (2) attribute value generation of large language model to
significantly alleviate the OOD problem and (3) its successful deployment in a
real-world industrial environment. Extensive experimental results demonstrate
that MVP-RAG performs better than the state-of-the-art baselines.</details></td><td><details><summary>展开</summary>该论文提出了一种称为MVP-RAG（Multi-Value-Product Retrieval-Augmented Generation）的方法，用于解决电子商务平台中的产品属性值识别（PAVI）问题。MVP-RAG结合了检索、生成和分类范式，通过多级检索方案（产品层级和属性值层级）检索相似产品和候选属性值，然后利用大语言模型生成标准化的属性值，显著缓解了分布外（OOD）问题，并已在工业环境中成功部署，实验结果显示其性能优于现有最优基线方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.23793v1">Transformer Tafsir at QIAS 2025 Shared Task: Hybrid Retrieval-Augmented Generation for Islamic Knowledge Question Answering</a></td><td><details><summary>展开</summary>This paper presents our submission to the QIAS 2025 shared task on Islamic
knowledge understanding and reasoning. We developed a hybrid
retrieval-augmented generation (RAG) system that combines sparse and dense
retrieval methods with cross-encoder reranking to improve large language model
(LLM) performance. Our three-stage pipeline incorporates BM25 for initial
retrieval, a dense embedding retrieval model for semantic matching, and
cross-encoder reranking for precise content retrieval. We evaluate our approach
on both subtasks using two LLMs, Fanar and Mistral, demonstrating that the
proposed RAG pipeline enhances performance across both, with accuracy
improvements up to 25%, depending on the task and model configuration. Our best
configuration is achieved with Fanar, yielding accuracy scores of 45% in
Subtask 1 and 80% in Subtask 2.</details></td><td><details><summary>展开</summary>这篇论文介绍了一个用于伊斯兰知识理解和推理的混合检索增强生成（RAG）系统，结合了稀疏与密集检索方法及交叉编码器重排序，通过三阶段流程（BM25初检、密集嵌入语义匹配、交叉编码器精检）提升大语言模型（Fanar和Mistral）性能，实验显示最高可将准确率提升25%，其中Fanar模型在两项子任务中分别达到45%和80%的准确率。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.23659v1">Aligning LLMs for Multilingual Consistency in Enterprise Applications</a></td><td><details><summary>展开</summary>Large language models (LLMs) remain unreliable for global enterprise
applications due to substantial performance gaps between high-resource and
mid/low-resource languages, driven by English-centric pretraining and internal
reasoning biases. This inconsistency undermines customer experience and
operational reliability in multilingual settings such as customer support,
content moderation, and information retrieval. Even with advanced
Retrieval-Augmented Generation (RAG) systems, we observe up to an 29% accuracy
drop in non-English languages compared to English.
  We propose a practical, batch-wise alignment strategy for fine-tuning LLMs,
leveraging semantically equivalent multilingual data in each training batch to
directly align model outputs across languages. This approach improves
non-English accuracy by up to 23.9\% without compromising English performance,
model reasoning, or retrieval quality. Our method is simple to implement,
scalable, and integrates seamlessly with existing LLM training \& deployment
pipelines, enabling more robust and equitable multilingual AI solutions in
industry.</details></td><td><details><summary>展开</summary>这篇论文探讨了大语言模型（LLMs）在多语言环境下（尤其是中低资源语言）性能下降的问题，指出即使采用RAG系统，非英语语言的准确率仍显著低于英语。作者提出了一种基于批量对齐的微调策略，利用多语言语义等效数据直接对齐模型输出，从而提升非英语语言的准确率（最高23.9%），同时不影响英语性能或检索质量。研究旨在增强RAG在多语言工业场景（如客服、内容审核等）中的公平性和可靠性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.23630v1">Game-Oriented ASR Error Correction via RAG-Enhanced LLM</a></td><td><details><summary>展开</summary>With the rise of multiplayer online games, real-time voice communication is
essential for team coordination. However, general ASR systems struggle with
gaming-specific challenges like short phrases, rapid speech, jargon, and noise,
leading to frequent errors. To address this, we propose the GO-AEC framework,
which integrates large language models, Retrieval-Augmented Generation (RAG),
and a data augmentation strategy using LLMs and TTS. GO-AEC includes data
augmentation, N-best hypothesis-based correction, and a dynamic game knowledge
base. Experiments show GO-AEC reduces character error rate by 6.22% and
sentence error rate by 29.71%, significantly improving ASR accuracy in gaming
scenarios.</details></td><td><details><summary>展开</summary>这篇论文提出了GO-AEC框架，结合大语言模型、检索增强生成（RAG）以及数据增强策略，针对游戏场景中的实时语音识别（ASR）挑战（如短短语、快速语音、术语和噪声）进行优化，通过动态游戏知识库和N-best假设校正显著降低了错误率。</details></td></tr></tbody></table>

### 📅 2025-09-27
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.23519v1">ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) enhances Large Language Models by
grounding their outputs in external documents. These systems, however, remain
vulnerable to attacks on the retrieval corpus, such as prompt injection.
RAG-based search systems (e.g., Google's Search AI Overview) present an
interesting setting for studying and protecting against such threats, as
defense algorithms can benefit from built-in reliability signals -- like
document ranking -- and represent a non-LLM challenge for the adversary due to
decades of work to thwart SEO.
  Motivated by, but not limited to, this scenario, this work introduces
ReliabilityRAG, a framework for adversarial robustness that explicitly
leverages reliability information of retrieved documents.
  Our first contribution adopts a graph-theoretic perspective to identify a
"consistent majority" among retrieved documents to filter out malicious ones.
We introduce a novel algorithm based on finding a Maximum Independent Set (MIS)
on a document graph where edges encode contradiction. Our MIS variant
explicitly prioritizes higher-reliability documents and provides provable
robustness guarantees against bounded adversarial corruption under natural
assumptions. Recognizing the computational cost of exact MIS for large
retrieval sets, our second contribution is a scalable weighted sample and
aggregate framework. It explicitly utilizes reliability information, preserving
some robustness guarantees while efficiently handling many documents.
  We present empirical results showing ReliabilityRAG provides superior
robustness against adversarial attacks compared to prior methods, maintains
high benign accuracy, and excels in long-form generation tasks where prior
robustness-focused methods struggled. Our work is a significant step towards
more effective, provably robust defenses against retrieved corpus corruption in
RAG.</details></td><td><details><summary>展开</summary>这篇文章提出了ReliabilityRAG框架，旨在增强RAG系统对抗检索文档库中恶意攻击（如提示注入）的鲁棒性。通过图论方法识别文档间的矛盾关系并优先选择高可靠性文档，结合可扩展的加权采样聚合技术，该框架在保证高效处理大规模检索集的同时，提供了理论上的对抗攻击防御保证，并在实验中展现出优于现有方法的性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.23233v1">Detecting Corpus-Level Knowledge Inconsistencies in Wikipedia with Large Language Models</a></td><td><details><summary>展开</summary>Wikipedia is the largest open knowledge corpus, widely used worldwide and
serving as a key resource for training large language models (LLMs) and
retrieval-augmented generation (RAG) systems. Ensuring its accuracy is
therefore critical. But how accurate is Wikipedia, and how can we improve it?
  We focus on inconsistencies, a specific type of factual inaccuracy, and
introduce the task of corpus-level inconsistency detection. We present CLAIRE,
an agentic system that combines LLM reasoning with retrieval to surface
potentially inconsistent claims along with contextual evidence for human
review. In a user study with experienced Wikipedia editors, 87.5% reported
higher confidence when using CLAIRE, and participants identified 64.7% more
inconsistencies in the same amount of time.
  Combining CLAIRE with human annotation, we contribute WIKICOLLIDE, the first
benchmark of real Wikipedia inconsistencies. Using random sampling with
CLAIRE-assisted analysis, we find that at least 3.3% of English Wikipedia facts
contradict another fact, with inconsistencies propagating into 7.3% of FEVEROUS
and 4.0% of AmbigQA examples. Benchmarking strong baselines on this dataset
reveals substantial headroom: the best fully automated system achieves an AUROC
of only 75.1%.
  Our results show that contradictions are a measurable component of Wikipedia
and that LLM-based systems like CLAIRE can provide a practical tool to help
editors improve knowledge consistency at scale.</details></td><td><details><summary>展开</summary>本文聚焦于维基百科中的事实不一致性问题，提出了一种结合大语言模型（LLM）与检索技术的智能系统CLAIRE，用于检测语料库级别的不一致主张并提供上下文证据，最终构建了首个真实维基百科不一致性基准WIKICOLLIDE。研究证实LLM驱动的系统（如CLAIRE）可辅助编辑高效提升知识一致性，同时揭示了此类不一致在现有数据集（如FEVEROUS、AmbigQA）中的渗透情况，凸显了自动化系统的改进空间。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.23071v1">From Evidence to Trajectory: Abductive Reasoning Path Synthesis for Training Retrieval-Augmented Generation Agents</a></td><td><details><summary>展开</summary>Retrieval-augmented generation agents development is hindered by the lack of
process-level supervision to effectively guide agentic capabilities like task
decomposition, retriever invocation, and stepwise decision-making. While
reinforcement learning offers a potential solution, it suffers from sparse
rewards and the limited reasoning capabilities of large language models (LLMs).
Meanwhile, existing data synthesis methods only produce chain-of-thought
rationales and fail to model environmental interactions. In this paper, we
propose EviPath, an evidence-anchored reasoning path synthesis paradigm for RAG
agent development. EviPath comprises: (i) Abductive Subtask Planning, which
decomposes the problem into sub-questions and iteratively plans an optimal
solution path based on the dependencies between them; (ii) Faithful
Sub-question Answering, which uses supporting evidence to construct a proxy
environment to generate reasoning thoughts and answers for each sub-question;
and (iii) Conversational Fine-Tuning, which formats the complete
agent-environment interaction trajectory into a dialogue format suitable for
Supervised Fine-Tuning. EviPath allows LLMs to learn complex reasoning and
tool-use capabilities directly from synthesized data. Extensive experiments on
widely-used question-answering benchmarks show that an 8B parameter model
trained with EviPath-synthesized data significantly and consistently
outperforms state-of-the-art baselines with a double-digit absolute EM gain of
14.7% in open-domain question answering.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为EviPath的证据锚定推理路径合成范式，用于解决RAG（检索增强生成）代理开发中过程级监督不足的问题。通过将问题分解为子任务、利用支持证据构建代理环境生成子问题答案，并将交互轨迹格式化为对话数据进行监督微调，EviPath显著提升了模型在开放域问答任务中的性能（EM增益达14.7%）。</details></td></tr></tbody></table>

### 📅 2025-09-26
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.22565v1">Retrieval-Augmented Guardrails for AI-Drafted Patient-Portal Messages: Error Taxonomy Construction and Large-Scale Evaluation</a></td><td><details><summary>展开</summary>Asynchronous patient-clinician messaging via EHR portals is a growing source
of clinician workload, prompting interest in large language models (LLMs) to
assist with draft responses. However, LLM outputs may contain clinical
inaccuracies, omissions, or tone mismatches, making robust evaluation
essential. Our contributions are threefold: (1) we introduce a clinically
grounded error ontology comprising 5 domains and 59 granular error codes,
developed through inductive coding and expert adjudication; (2) we develop a
retrieval-augmented evaluation pipeline (RAEC) that leverages semantically
similar historical message-response pairs to improve judgment quality; and (3)
we provide a two-stage prompting architecture using DSPy to enable scalable,
interpretable, and hierarchical error detection. Our approach assesses the
quality of drafts both in isolation and with reference to similar past
message-response pairs retrieved from institutional archives. Using a two-stage
DSPy pipeline, we compared baseline and reference-enhanced evaluations on over
1,500 patient messages. Retrieval context improved error identification in
domains such as clinical completeness and workflow appropriateness. Human
validation on 100 messages demonstrated superior agreement (concordance = 50%
vs. 33%) and performance (F1 = 0.500 vs. 0.256) of context-enhanced labels vs.
baseline, supporting the use of our RAEC pipeline as AI guardrails for patient
messaging.</details></td><td><details><summary>展开</summary>这篇论文提出了一种检索增强的评估管道（RAEC），利用语义相似的历史消息-响应对来改进对大型语言模型（LLMs）生成的临床回复草案的质量评估，并通过两阶段提示架构实现可扩展和分层次的错误检测，验证了检索上下文在提升临床完整性和工作流适当性等领域的错误识别效果。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.22516v1">TrueGradeAI: Retrieval-Augmented and Bias-Resistant AI for Transparent and Explainable Digital Assessments</a></td><td><details><summary>展开</summary>This paper introduces TrueGradeAI, an AI-driven digital examination framework
designed to overcome the shortcomings of traditional paper-based assessments,
including excessive paper usage, logistical complexity, grading delays, and
evaluator bias. The system preserves natural handwriting by capturing stylus
input on secure tablets and applying transformer-based optical character
recognition for transcription. Evaluation is conducted through a
retrieval-augmented pipeline that integrates faculty solutions, cache layers,
and external references, enabling a large language model to assign scores with
explicit, evidence-linked reasoning. Unlike prior tablet-based exam systems
that primarily digitize responses, TrueGradeAI advances the field by
incorporating explainable automation, bias mitigation, and auditable grading
trails. By uniting handwriting preservation with scalable and transparent
evaluation, the framework reduces environmental costs, accelerates feedback
cycles, and progressively builds a reusable knowledge base, while actively
working to mitigate grading bias and ensure fairness in assessment.</details></td><td><details><summary>展开</summary>该论文提出TrueGradeAI框架，通过基于触控笔输入的数字化考试系统结合检索增强流程（集成教师答案、缓存层和外部参考），利用大语言模型进行可解释、证据关联的评分，解决传统考试弊端并提升透明度和公平性，属于RAG在自动化评估领域的应用。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.22490v1">JGU Mainz's Submission to the WMT25 Shared Task on LLMs with Limited Resources for Slavic Languages: MT and QA</a></td><td><details><summary>展开</summary>This paper presents the JGU Mainz submission to the WMT25 Shared Task on LLMs
with Limited Resources for Slavic Languages: Machine Translation and Question
Answering, focusing on Ukrainian, Upper Sorbian, and Lower Sorbian. For each
language, we jointly fine-tune a Qwen2.5-3B-Instruct model for both tasks with
parameter-efficient finetuning. Our pipeline integrates additional translation
and multiple-choice question answering (QA) data. For Ukrainian QA, we further
use retrieval-augmented generation. We also apply ensembling for QA in Upper
and Lower Sorbian. Experiments show that our models outperform the baseline on
both tasks.</details></td><td><details><summary>展开</summary>该论文介绍了JGU Mainz团队针对低资源斯拉夫语（乌克兰语、上索布语和下索布语）的机器翻译和问答任务，使用Qwen2.5-3B-Instruct模型进行联合微调，并在乌克兰语问答中采用检索增强生成（RAG）技术，实验表明模型性能优于基线。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.22486v1">Your RAG is Unfair: Exposing Fairness Vulnerabilities in Retrieval-Augmented Generation via Backdoor Attacks</a></td><td><details><summary>展开</summary>Retrieval-augmented generation (RAG) enhances factual grounding by
integrating retrieval mechanisms with generative models but introduces new
attack surfaces, particularly through backdoor attacks. While prior research
has largely focused on disinformation threats, fairness vulnerabilities remain
underexplored. Unlike conventional backdoors that rely on direct
trigger-to-target mappings, fairness-driven attacks exploit the interaction
between retrieval and generation models, manipulating semantic relationships
between target groups and social biases to establish a persistent and covert
influence on content generation.
  This paper introduces BiasRAG, a systematic framework that exposes fairness
vulnerabilities in RAG through a two-phase backdoor attack. During the
pre-training phase, the query encoder is compromised to align the target group
with the intended social bias, ensuring long-term persistence. In the
post-deployment phase, adversarial documents are injected into knowledge bases
to reinforce the backdoor, subtly influencing retrieved content while remaining
undetectable under standard fairness evaluations. Together, BiasRAG ensures
precise target alignment over sensitive attributes, stealthy execution, and
resilience. Empirical evaluations demonstrate that BiasRAG achieves high attack
success rates while preserving contextual relevance and utility, establishing a
persistent and evolving threat to fairness in RAG.</details></td><td><details><summary>展开</summary>这篇论文探讨了RAG技术在公平性方面的潜在漏洞，提出了一种名为BiasRAG的两阶段后门攻击框架。该攻击通过在预训练阶段操纵查询编码器使其与特定社会偏见对齐，并在部署后阶段向知识库注入对抗性文档，从而在保持隐蔽性的同时持续影响生成内容。研究表明，BiasRAG不仅能高效实施攻击，还揭示了现有公平性评估的局限性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.22378v1">Zero-Effort Image-to-Music Generation: An Interpretable RAG-based VLM Approach</a></td><td><details><summary>展开</summary>Recently, Image-to-Music (I2M) generation has garnered significant attention,
with potential applications in fields such as gaming, advertising, and
multi-modal art creation. However, due to the ambiguous and subjective nature
of I2M tasks, most end-to-end methods lack interpretability, leaving users
puzzled about the generation results. Even methods based on emotion mapping
face controversy, as emotion represents only a singular aspect of art.
Additionally, most learning-based methods require substantial computational
resources and large datasets for training, hindering accessibility for common
users. To address these challenges, we propose the first Vision Language Model
(VLM)-based I2M framework that offers high interpretability and low
computational cost. Specifically, we utilize ABC notation to bridge the text
and music modalities, enabling the VLM to generate music using natural
language. We then apply multi-modal Retrieval-Augmented Generation (RAG) and
self-refinement techniques to allow the VLM to produce high-quality music
without external training. Furthermore, we leverage the generated motivations
in text and the attention maps from the VLM to provide explanations for the
generated results in both text and image modalities. To validate our method, we
conduct both human studies and machine evaluations, where our method
outperforms others in terms of music quality and music-image consistency,
indicating promising results. Our code is available at
https://github.com/RS2002/Image2Music .</details></td><td><details><summary>展开</summary>这篇文章提出了一种基于视觉语言模型（VLM）的Image-to-Music（I2M）生成框架，通过多模态检索增强生成（RAG）和自优化技术，无需外部训练即可生成高质量音乐，并利用文本动机和注意力图提供跨模态解释，在音乐质量与图文一致性上优于现有方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.22325v1">Can Synthetic Query Rewrites Capture User Intent Better than Humans in Retrieval-Augmented Generation?</a></td><td><details><summary>展开</summary>Multi-turn RAG systems often face queries with colloquial omissions and
ambiguous references, posing significant challenges for effective retrieval and
generation. Traditional query rewriting relies on human annotators to clarify
queries, but due to limitations in annotators' expressive ability and depth of
understanding, manually rewritten queries often diverge from those needed in
real-world RAG systems, resulting in a gap between user intent and system
response. We observe that high-quality synthetic queries can better bridge this
gap, achieving superior performance in both retrieval and generation compared
to human rewrites. This raises an interesting question: Can rewriting models
trained on synthetic queries better capture user intent than human annotators?
In this paper, we propose SynRewrite, a synthetic data-driven query rewriting
model to generate high-quality synthetic rewrites more aligned with user
intent. To construct training data, we prompt GPT-4o with dialogue history,
current queries, positive documents, and answers to synthesize high-quality
rewrites. A Flan-T5 model is then finetuned on this dataset to map dialogue
history and queries to synthetic rewrites. Finally, we further enhance the
rewriter using the generator's feedback through the DPO algorithm to boost
end-task performance. Experiments on TopiOCQA and QRECC datasets show that
SynRewrite consistently outperforms human rewrites in both retrieval and
generation tasks. Our results demonstrate that synthetic rewrites can serve as
a scalable and effective alternative to human annotations.</details></td><td><details><summary>展开</summary>这篇论文探讨了多轮RAG系统中面对口语化省略和模糊指代查询时的挑战，提出了一种基于合成数据的查询重写模型SynRewrite。该方法利用GPT-4o生成高质量的重写查询训练数据，并微调Flan-T5模型，再通过DPO算法结合生成器反馈优化性能。实验表明，SynRewrite在检索和生成任务中表现优于人工重写，证明合成数据能有效替代人工标注。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.22009v1">GraphSearch: An Agentic Deep Searching Workflow for Graph Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Graph Retrieval-Augmented Generation (GraphRAG) enhances factual reasoning in
LLMs by structurally modeling knowledge through graph-based representations.
However, existing GraphRAG approaches face two core limitations: shallow
retrieval that fails to surface all critical evidence, and inefficient
utilization of pre-constructed structural graph data, which hinders effective
reasoning from complex queries. To address these challenges, we propose
\textsc{GraphSearch}, a novel agentic deep searching workflow with dual-channel
retrieval for GraphRAG. \textsc{GraphSearch} organizes the retrieval process
into a modular framework comprising six modules, enabling multi-turn
interactions and iterative reasoning. Furthermore, \textsc{GraphSearch} adopts
a dual-channel retrieval strategy that issues semantic queries over chunk-based
text data and relational queries over structural graph data, enabling
comprehensive utilization of both modalities and their complementary strengths.
Experimental results across six multi-hop RAG benchmarks demonstrate that
\textsc{GraphSearch} consistently improves answer accuracy and generation
quality over the traditional strategy, confirming \textsc{GraphSearch} as a
promising direction for advancing graph retrieval-augmented generation.</details></td><td><details><summary>展开</summary>该论文提出了一种名为GraphSearch的新型图检索增强生成（GraphRAG）方法，通过双通道检索策略（语义查询和关系查询）及模块化工作流解决了传统GraphRAG方法中检索浅层化和图数据利用效率低的问题，实验证明其在多跳RAG基准测试中显著提升了答案准确性和生成质量。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.21875v1">LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) aims to mitigate hallucinations in large
language models (LLMs) by grounding responses in retrieved documents. Yet,
RAG-based LLMs still hallucinate even when provided with correct and sufficient
context. A growing line of work suggests that this stems from an imbalance
between how models use external context and their internal knowledge, and
several approaches have attempted to quantify these signals for hallucination
detection. However, existing methods require extensive hyperparameter tuning,
limiting their generalizability. We propose LUMINA, a novel framework that
detects hallucinations in RAG systems through context-knowledge signals:
external context utilization is quantified via distributional distance, while
internal knowledge utilization is measured by tracking how predicted tokens
evolve across transformer layers. We further introduce a framework for
statistically validating these measurements. Experiments on common RAG
hallucination benchmarks and four open-source LLMs show that LUMINA achieves
consistently high AUROC and AUPRC scores, outperforming prior utilization-based
methods by up to +13% AUROC on HalluRAG. Moreover, LUMINA remains robust under
relaxed assumptions about retrieval quality and model matching, offering both
effectiveness and practicality.</details></td><td><details><summary>展开</summary>这篇论文提出了LUMINA框架，专门用于检测RAG系统中因上下文与内部知识利用不平衡导致的幻觉问题。通过量化外部上下文分布距离和内部知识在Transformer层中的演化，结合统计验证方法，LUMINA在多个RAG基准测试中显著优于现有方法（如AUROC提升13%），且对检索质量和模型适配具有更强鲁棒性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.21865v1">Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) is a framework for grounding Large
Language Models (LLMs) in external, up-to-date information. However, recent
advancements in context window size allow LLMs to process inputs of up to 128K
tokens or more, offering an alternative strategy: supplying the full document
context directly to the model, rather than relying on RAG to retrieve a subset
of contexts. Nevertheless, this emerging alternative strategy has notable
limitations: (i) it is token-inefficient to handle large and potentially
redundant contexts; (ii) it exacerbates the `lost in the middle' phenomenon;
and (iii) under limited model capacity, it amplifies distraction, ultimately
degrading LLM output quality. In this paper, we propose LDAR (Learning
Distraction-Aware Retrieval), an adaptive retriever that learns to retrieve
contexts in a way that mitigates interference from distracting passages,
thereby achieving significantly higher performance with reduced token usage
compared to long-context approaches. Extensive experiments across diverse LLM
architectures and six knowledge-intensive benchmarks demonstrate the
effectiveness and robustness of our approach, highlighting the importance of
balancing the trade-off between information coverage and distraction.</details></td><td><details><summary>展开</summary>这篇论文探讨了在大型语言模型（LLM）上下文窗口增大的背景下，RAG（检索增强生成）面临的挑战与改进方法，提出了LDAR（Learning Distraction-Aware Retrieval）算法，通过自适应检索减少干扰性段落的影响，以提高性能并降低token使用量，证明了在信息覆盖与干扰间平衡的重要性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.21856v1">KnowMT-Bench: Benchmarking Knowledge-Intensive Long-Form Question Answering in Multi-Turn Dialogues</a></td><td><details><summary>展开</summary>Multi-Turn Long-Form Question Answering (MT-LFQA) is a key application
paradigm of Large Language Models (LLMs) in knowledge-intensive domains.
However, existing benchmarks are limited to single-turn dialogue, while
multi-turn dialogue benchmarks typically assess other orthogonal capabilities
rather than knowledge-intensive factuality. To bridge this critical gap, we
introduce \textbf{KnowMT-Bench}, the \textit{first-ever} benchmark designed to
systematically evaluate MT-LFQA for LLMs across knowledge-intensive fields,
including medicine, finance, and law. To faithfully assess the model's
real-world performance, KnowMT-Bench employs a dynamic evaluation setting where
models generate their own multi-turn dialogue histories given logically
progressive question sequences. The factual capability and information delivery
efficiency of the \textit{final-turn} answer are then evaluated using a
human-validated automated pipeline. Our experiments reveal that multi-turn
contexts degrade performance: factual capability declines due to the contextual
noise from self-generated histories, while information efficiency drops as
models become more verbose with increasing dialogue length. We then investigate
mitigation strategies, demonstrating that retrieval-augmented generation (RAG)
can effectively alleviate and even reverse this factual degradation. These
findings underscore the importance of our benchmark in evaluating and enhancing
the conversational factual capabilities of LLMs in real-world
knowledge-intensive applications. Code is available at
\href{https://github.com/hardenyu21/KnowMT-Bench}{\textcolor{cyan}{\texttt{KnowMT-Bench}}}.</details></td><td><details><summary>展开</summary>这篇文章介绍了KnowMT-Bench，首个用于评估大语言模型在多轮长形式问答（MT-LFQA）中知识密集型领域性能的基准测试，研究发现多轮对话会降低模型的事实性表现，但检索增强生成（RAG）能有效缓解这一退化，强调了RAG在提升模型对话事实性能力中的重要性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.21848v1">Graph of Agents: Principled Long Context Modeling by Emergent Multi-Agent Collaboration</a></td><td><details><summary>展开</summary>As a model-agnostic approach to long context modeling, multi-agent systems
can process inputs longer than a large language model's context window without
retraining or architectural modifications. However, their performance often
heavily relies on hand-crafted multi-agent collaboration strategies and prompt
engineering, which limit generalizability. In this work, we introduce a
principled framework that formalizes the model-agnostic long context modeling
problem as a compression problem, yielding an information-theoretic compression
objective. Building on this framework, we propose Graph of Agents (GoA), which
dynamically constructs an input-dependent collaboration structure that
maximizes this objective. For Llama 3.1 8B and Qwen3 8B across six document
question answering benchmarks, GoA improves the average $F_1$ score of
retrieval-augmented generation by 5.7\% and a strong multi-agent baseline using
a fixed collaboration structure by 16.35\%, respectively. Even with only a 2K
context window, GoA surpasses the 128K context window Llama 3.1 8B on
LongBench, showing a dramatic increase in effective context length. Our source
code is available at https://github.com/tjoo512/graph-of-agents.</details></td><td><details><summary>展开</summary>本文提出了一种名为"Graph of Agents (GoA)"的多智能体框架，通过将长上下文建模问题形式化为压缩问题，并动态构建输入相关的协作结构来优化检索增强生成（RAG）性能。实验表明，GoA在六个文档问答基准测试中显著提升了RAG的F1分数（5.7%）和多智能体基线性能（16.35%），且在仅有2K上下文窗口时超越128K窗口的Llama 3.1模型，大幅提升了有效上下文长度。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.21730v1">ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation</a></td><td><details><summary>展开</summary>As large language models (LLMs) become increasingly integrated into daily
life, there is growing demand for AI assistants that are not only reactive but
also proactive and personalized. While recent advances have pushed forward
proactivity and personalization individually, their combination remains
underexplored. To bridge this gap, we introduce ProPerSim, a new task and
simulation framework for developing assistants capable of making timely,
personalized recommendations in realistic home scenarios. In our simulation
environment, a user agent with a rich persona interacts with the assistant,
providing ratings on how well each suggestion aligns with its preferences and
context. The assistant's goal is to use these ratings to learn and adapt to
achieve higher scores over time. Built on ProPerSim, we propose
ProPerAssistant, a retrieval-augmented, preference-aligned assistant that
continually learns and adapts through user feedback. Experiments across 32
diverse personas show that ProPerAssistant adapts its strategy and steadily
improves user satisfaction, highlighting the promise of uniting proactivity and
personalization.</details></td><td><details><summary>展开</summary>这篇论文提出了一个名为ProPerSim的任务和仿真框架，旨在开发能够在现实家庭场景中提供主动且个性化推荐的AI助手。论文介绍了ProPerAssistant，一种基于检索增强（retrieval-augmented）、偏好对齐的助手，通过用户反馈持续学习和适应，实验结果表明其在32种不同用户角色中能逐步提升用户满意度。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.21710v1">Think-on-Graph 3.0: Efficient and Adaptive LLM Reasoning on Heterogeneous Graphs via Multi-Agent Dual-Evolving Context Retrieval</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) and Graph-based RAG has become the
important paradigm for enhancing Large Language Models (LLMs) with external
knowledge. However, existing approaches face a fundamental trade-off. While
graph-based methods are inherently dependent on high-quality graph structures,
they face significant practical constraints: manually constructed knowledge
graphs are prohibitively expensive to scale, while automatically extracted
graphs from corpora are limited by the performance of the underlying LLM
extractors, especially when using smaller, local-deployed models. This paper
presents Think-on-Graph 3.0 (ToG-3), a novel framework that introduces
Multi-Agent Context Evolution and Retrieval (MACER) mechanism to overcome these
limitations. Our core innovation is the dynamic construction and refinement of
a Chunk-Triplets-Community heterogeneous graph index, which pioneeringly
incorporates a dual-evolution mechanism of Evolving Query and Evolving
Sub-Graph for precise evidence retrieval. This approach addresses a critical
limitation of prior Graph-based RAG methods, which typically construct a static
graph index in a single pass without adapting to the actual query. A
multi-agent system, comprising Constructor, Retriever, Reflector, and Responser
agents, collaboratively engages in an iterative process of evidence retrieval,
answer generation, sufficiency reflection, and, crucially, evolving query and
subgraph. This dual-evolving multi-agent system allows ToG-3 to adaptively
build a targeted graph index during reasoning, mitigating the inherent
drawbacks of static, one-time graph construction and enabling deep, precise
reasoning even with lightweight LLMs. Extensive experiments demonstrate that
ToG-3 outperforms compared baselines on both deep and broad reasoning
benchmarks, and ablation studies confirm the efficacy of the components of
MACER framework.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为Think-on-Graph 3.0（ToG-3）的新型RAG框架，通过创新的多代理上下文演化与检索（MACER）机制和动态构建的Chunk-Triplets-Community异构图索引，改进了传统基于图的RAG方法中静态图索引的局限性，实现了查询和子图的双重演化，从而在轻量级大语言模型上实现了更深更精准的推理。</details></td></tr></tbody></table>

### 📅 2025-09-25
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.21237v1">Query-Centric Graph Retrieval Augmented Generation</a></td><td><details><summary>展开</summary>Graph-based retrieval-augmented generation (RAG) enriches large language
models (LLMs) with external knowledge for long-context understanding and
multi-hop reasoning, but existing methods face a granularity dilemma:
fine-grained entity-level graphs incur high token costs and lose context, while
coarse document-level graphs fail to capture nuanced relations. We introduce
QCG-RAG, a query-centric graph RAG framework that enables query-granular
indexing and multi-hop chunk retrieval. Our query-centric approach leverages
Doc2Query and Doc2Query{-}{-} to construct query-centric graphs with
controllable granularity, improving graph quality and interpretability. A
tailored multi-hop retrieval mechanism then selects relevant chunks via the
generated queries. Experiments on LiHuaWorld and MultiHop-RAG show that QCG-RAG
consistently outperforms prior chunk-based and graph-based RAG methods in
question answering accuracy, establishing a new paradigm for multi-hop
reasoning.</details></td><td><details><summary>展开</summary>该论文提出了QCG-RAG框架，通过构建查询为中心的图结构解决现有基于图的RAG方法中粒度困境（细粒度导致高开销，粗粒度丢失细节关系），结合可控粒度索引和多跳分块检索机制，在问答任务中超越传统分块和图基方法，提升了多跳推理性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.21208v1">CLaw: Benchmarking Chinese Legal Knowledge in Large Language Models - A Fine-grained Corpus and Reasoning Analysis</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) are increasingly tasked with analyzing legal
texts and citing relevant statutes, yet their reliability is often compromised
by general pre-training that ingests legal texts without specialized focus,
obscuring the true depth of their legal knowledge. This paper introduces CLaw,
a novel benchmark specifically engineered to meticulously evaluate LLMs on
Chinese legal knowledge and its application in reasoning. CLaw comprises two
key components: (1) a comprehensive, fine-grained corpus of all 306 Chinese
national statutes, segmented to the subparagraph level and incorporating
precise historical revision timesteps for rigorous recall evaluation (64,849
entries), and (2) a challenging set of 254 case-based reasoning instances
derived from China Supreme Court curated materials to assess the practical
application of legal knowledge. Our empirical evaluation reveals that most
contemporary LLMs significantly struggle to faithfully reproduce legal
provisions. As accurate retrieval and citation of legal provisions form the
basis of legal reasoning, this deficiency critically undermines the reliability
of their responses. We contend that achieving trustworthy legal reasoning in
LLMs requires a robust synergy of accurate knowledge retrieval--potentially
enhanced through supervised fine-tuning (SFT) or retrieval-augmented generation
(RAG)--and strong general reasoning capabilities. This work provides an
essential benchmark and critical insights for advancing domain-specific LLM
reasoning, particularly within the complex legal sphere.</details></td><td><details><summary>展开</summary>这篇论文介绍了CLaw基准，旨在评估大语言模型（LLMs）在中国法律知识及其推理应用中的表现，发现现有模型在准确检索和引用法律条文方面存在重大缺陷，并指出通过监督微调（SFT）或检索增强生成（RAG）等技术改进知识检索能力是实现可靠法律推理的关键。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.21193v1">Eigen-1: Adaptive Multi-Agent Refinement with Monitor-Based RAG for Scientific Reasoning</a></td><td><details><summary>展开</summary>Large language models (LLMs) have recently shown strong progress on
scientific reasoning, yet two major bottlenecks remain. First, explicit
retrieval fragments reasoning, imposing a hidden "tool tax" of extra tokens and
steps. Second, multi-agent pipelines often dilute strong solutions by averaging
across all candidates. We address these challenges with a unified framework
that combines implicit retrieval and structured collaboration. At its
foundation, a Monitor-based retrieval module operates at the token level,
integrating external knowledge with minimal disruption to reasoning. On top of
this substrate, Hierarchical Solution Refinement (HSR) iteratively designates
each candidate as an anchor to be repaired by its peers, while Quality-Aware
Iterative Reasoning (QAIR) adapts refinement to solution quality. On Humanity's
Last Exam (HLE) Bio/Chem Gold, our framework achieves 48.3\% accuracy -- the
highest reported to date, surpassing the strongest agent baseline by 13.4
points and leading frontier LLMs by up to 18.1 points, while simultaneously
reducing token usage by 53.5\% and agent steps by 43.7\%. Results on SuperGPQA
and TRQA confirm robustness across domains. Error analysis shows that reasoning
failures and knowledge gaps co-occur in over 85\% of cases, while diversity
analysis reveals a clear dichotomy: retrieval tasks benefit from solution
variety, whereas reasoning tasks favor consensus. Together, these findings
demonstrate how implicit augmentation and structured refinement overcome the
inefficiencies of explicit tool use and uniform aggregation. Code is available
at: https://github.com/tangxiangru/Eigen-1.</details></td><td><details><summary>展开</summary>这篇论文提出了一种结合隐式检索和结构化协作的统一框架来解决大语言模型在科学推理中的问题。该框架通过基于Monitor的检索模块在token级别集成外部知识，减少推理中断，并采用分层解决方案精炼（HSR）和质量感知迭代推理（QAIR）来优化结果。实验表明，该框架在多项任务中实现了最高准确率，同时显著降低了token和计算步骤的消耗。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.21188v1">Adoption, usability and perceived clinical value of a UK AI clinical reference platform (iatroX): a mixed-methods formative evaluation of real-world usage and a 1,223-respondent user survey</a></td><td><details><summary>展开</summary>Clinicians face growing information overload from biomedical literature and
guidelines, hindering evidence-based care. Retrieval-augmented generation (RAG)
with large language models may provide fast, provenance-linked answers, but
requires real-world evaluation. We describe iatroX, a UK-centred RAG-based
clinical reference platform, and report early adoption, usability, and
perceived clinical value from a formative implementation evaluation. Methods
comprised a retrospective analysis of usage across web, iOS, and Android over
16 weeks (8 April-31 July 2025) and an in-product intercept survey. Usage
metrics were drawn from web and app analytics with bot filtering. A client-side
script randomized single-item prompts to approx. 10% of web sessions from a
predefined battery assessing usefulness, reliability, and adoption intent.
Proportions were summarized with Wilson 95% confidence intervals; free-text
comments underwent thematic content analysis. iatroX reached 19,269 unique web
users, 202,660 engagement events, and approx. 40,000 clinical queries. Mobile
uptake included 1,960 iOS downloads and Android growth (peak >750 daily active
users). The survey yielded 1,223 item-level responses: perceived usefulness
86.2% (95% CI 74.8-93.9%; 50/58); would use again 93.3% (95% CI 68.1-99.8%;
14/15); recommend to a colleague 88.4% (95% CI 75.1-95.9%; 38/43); perceived
accuracy 75.0% (95% CI 58.8-87.3%; 30/40); reliability 79.4% (95% CI
62.1-91.3%; 27/34). Themes highlighted speed, guideline-linked answers, and UK
specificity. Early real-world use suggests iatroX can mitigate information
overload and support timely answers for UK clinicians. Limitations include
small per-item samples and early-adopter bias; future work will include
accuracy audits and prospective studies on workflow and care quality.</details></td><td><details><summary>展开</summary>这篇文章介绍了基于RAG技术的临床参考平台iatroX，旨在解决临床医生面临的信息过载问题。该平台通过检索增强生成提供快速、可溯源的医疗答案，并在英国进行实际应用评估，结果显示早期用户对其有用性、准确性和可靠性持积极评价。研究还分析了平台的使用数据、用户反馈及局限性，并展望了未来的研究方向。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.21035v1">CLAUSE: Agentic Neuro-Symbolic Knowledge Graph Reasoning via Dynamic Learnable Context Engineering</a></td><td><details><summary>展开</summary>Knowledge graphs provide structured context for multi-hop question answering,
but deployed systems must balance answer accuracy with strict latency and cost
targets while preserving provenance. Static k-hop expansions and "think-longer"
prompting often over-retrieve, inflate context, and yield unpredictable
runtime. We introduce CLAUSE, an agentic three-agent neuro-symbolic framework
that treats context construction as a sequential decision process over
knowledge graphs, deciding what to expand, which paths to follow or backtrack,
what evidence to keep, and when to stop. Latency (interaction steps) and prompt
cost (selected tokens) are exposed as user-specified budgets or prices,
allowing per-query adaptation to trade-offs among accuracy, latency, and cost
without retraining. CLAUSE employs the proposed Lagrangian-Constrained
Multi-Agent Proximal Policy Optimization (LC-MAPPO) algorithm to coordinate
three agents: Subgraph Architect, Path Navigator, and Context Curator, so that
subgraph construction, reasoning-path discovery, and evidence selection are
jointly optimized under per-query resource budgets on edge edits, interaction
steps, and selected tokens. Across HotpotQA, MetaQA, and FactKG, CLAUSE yields
higher EM@1 while reducing subgraph growth and end-to-end latency at equal or
lower token budgets. On MetaQA-2-hop, relative to the strongest RAG baseline
(GraphRAG), CLAUSE achieves +39.3 EM@1 with 18.6% lower latency and 40.9% lower
edge growth. The resulting contexts are compact, provenance-preserving, and
deliver predictable performance under deployment constraints.</details></td><td><details><summary>展开</summary>这篇论文提出了CLAUSE，一种基于智能神经符号框架的多代理系统，用于优化知识图谱上的上下文构建过程，通过动态决策在准确性、延迟和成本之间进行权衡。CLAUSE利用LC-MAPPO算法协调三个代理（子图构建、路径导航和上下文管理），在资源限制下提升多跳问答的性能，相较于传统RAG方法（如GraphRAG），它在减少子图增长和延迟的同时显著提高了准确率。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.20953v1">Beyond Stars: Bridging the Gap Between Ratings and Review Sentiment with LLM</a></td><td><details><summary>展开</summary>We present an advanced approach to mobile app review analysis aimed at
addressing limitations inherent in traditional star-rating systems. Star
ratings, although intuitive and popular among users, often fail to capture the
nuanced feedback present in detailed review texts. Traditional NLP techniques
-- such as lexicon-based methods and classical machine learning classifiers --
struggle to interpret contextual nuances, domain-specific terminology, and
subtle linguistic features like sarcasm. To overcome these limitations, we
propose a modular framework leveraging large language models (LLMs) enhanced by
structured prompting techniques. Our method quantifies discrepancies between
numerical ratings and textual sentiment, extracts detailed, feature-level
insights, and supports interactive exploration of reviews through
retrieval-augmented conversational question answering (RAG-QA). Comprehensive
experiments conducted on three diverse datasets (AWARE, Google Play, and
Spotify) demonstrate that our LLM-driven approach significantly surpasses
baseline methods, yielding improved accuracy, robustness, and actionable
insights in challenging and context-rich review scenarios.</details></td><td><details><summary>展开</summary>本文提出了一种利用大型语言模型（LLMs）和结构化提示技术的模块化框架，通过检索增强的对话问答（RAG-QA）来分析移动应用评论，以克服传统星级评分和非结构化NLP方法的局限性，并在多数据集实验中展现出优越性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.20859v1">Concise and Sufficient Sub-Sentence Citations for Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>In retrieval-augmented generation (RAG) question answering systems,
generating citations for large language model (LLM) outputs enhances
verifiability and helps users identify potential hallucinations. However, we
observe two problems in the citations produced by existing attribution methods.
First, the citations are typically provided at the sentence or even paragraph
level. Long sentences or paragraphs may include a substantial amount of
irrelevant content. Second, sentence-level citations may omit information that
is essential for verifying the output, forcing users to read the surrounding
context. In this paper, we propose generating sub-sentence citations that are
both concise and sufficient, thereby reducing the effort required by users to
confirm the correctness of the generated output. To this end, we first develop
annotation guidelines for such citations and construct a corresponding dataset.
Then, we propose an attribution framework for generating citations that adhere
to our standards. This framework leverages LLMs to automatically generate
fine-tuning data for our task and employs a credit model to filter out
low-quality examples. Our experiments on the constructed dataset demonstrate
that the propose approach can generate high-quality and more readable
citations.</details></td><td><details><summary>展开</summary>该论文探讨了在RAG问答系统中为LLM输出生成更精确的子句级别引用（而非传统句子或段落级）的方法，旨在提升引用信息的简洁性和充分性，减少用户验证成本，并提出了一种结合自动标注和数据过滤的归因框架，通过实验验证了其有效性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.20769v1">Provenance Analysis of Archaeological Artifacts via Multimodal RAG Systems</a></td><td><details><summary>展开</summary>In this work, we present a retrieval-augmented generation (RAG)-based system
for provenance analysis of archaeological artifacts, designed to support expert
reasoning by integrating multimodal retrieval and large vision-language models
(VLMs). The system constructs a dual-modal knowledge base from reference texts
and images, enabling raw visual, edge-enhanced, and semantic retrieval to
identify stylistically similar objects. Retrieved candidates are synthesized by
the VLM to generate structured inferences, including chronological,
geographical, and cultural attributions, alongside interpretive justifications.
We evaluate the system on a set of Eastern Eurasian Bronze Age artifacts from
the British Museum. Expert evaluation demonstrates that the system produces
meaningful and interpretable outputs, offering scholars concrete starting
points for analysis and significantly alleviating the cognitive burden of
navigating vast comparative corpora.</details></td><td><details><summary>展开</summary>该论文提出了一种基于检索增强生成（RAG）的系统，用于考古文物来源分析，通过整合多模态检索和大型视觉-语言模型（VLMs），构建双模态知识库以检索风格相似的文物，并生成结构化推断（如年代、地理和文化属性）及解释性理由，经大英博物馆的欧亚青铜器文物验证，专家评估表明系统能有效支持学术分析并减轻认知负担。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.20707v1">An Automated Retrieval-Augmented Generation LLaMA-4 109B-based System for Evaluating Radiotherapy Treatment Plans</a></td><td><details><summary>展开</summary>Purpose: To develop a retrieval-augmented generation (RAG) system powered by
LLaMA-4 109B for automated, protocol-aware, and interpretable evaluation of
radiotherapy treatment plans.
  Methods and Materials: We curated a multi-protocol dataset of 614
radiotherapy plans across four disease sites and constructed a knowledge base
containing normalized dose metrics and protocol-defined constraints. The RAG
system integrates three core modules: a retrieval engine optimized across five
SentenceTransformer backbones, a percentile prediction component based on
cohort similarity, and a clinical constraint checker. These tools are directed
by a large language model (LLM) using a multi-step prompt-driven reasoning
pipeline to produce concise, grounded evaluations.
  Results: Retrieval hyperparameters were optimized using Gaussian Process on a
scalarized loss function combining root mean squared error (RMSE), mean
absolute error (MAE), and clinically motivated accuracy thresholds. The best
configuration, based on all-MiniLM-L6-v2, achieved perfect nearest-neighbor
accuracy within a 5-percentile-point margin and a sub-2pt MAE. When tested
end-to-end, the RAG system achieved 100% agreement with the computed values by
standalone retrieval and constraint-checking modules on both percentile
estimates and constraint identification, confirming reliable execution of all
retrieval, prediction and checking steps.
  Conclusion: Our findings highlight the feasibility of combining structured
population-based scoring with modular tool-augmented reasoning for transparent,
scalable plan evaluation in radiation therapy. The system offers traceable
outputs, minimizes hallucination, and demonstrates robustness across protocols.
Future directions include clinician-led validation, and improved domain-adapted
retrieval models to enhance real-world integration.</details></td><td><details><summary>展开</summary>这篇文章提出了一种基于LLaMA-4 109B的RAG系统，用于放射治疗计划的自动化、协议感知和可解释性评估。该系统通过整合检索引擎、百分位数预测组件和临床约束检查器，利用多步提示驱动的推理流程生成精确评估，并在实验中展现了高准确性和可靠性，同时减少了幻觉输出。</details></td></tr></tbody></table>

### 📅 2025-09-24
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.20324v1">RAG Security and Privacy: Formalizing the Threat Model and Attack Surface</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) is an emerging approach in natural
language processing that combines large language models (LLMs) with external
document retrieval to produce more accurate and grounded responses. While RAG
has shown strong potential in reducing hallucinations and improving factual
consistency, it also introduces new privacy and security challenges that differ
from those faced by traditional LLMs. Existing research has demonstrated that
LLMs can leak sensitive information through training data memorization or
adversarial prompts, and RAG systems inherit many of these vulnerabilities. At
the same time, reliance of RAG on an external knowledge base opens new attack
surfaces, including the potential for leaking information about the presence or
content of retrieved documents, or for injecting malicious content to
manipulate model behavior. Despite these risks, there is currently no formal
framework that defines the threat landscape for RAG systems. In this paper, we
address a critical gap in the literature by proposing, to the best of our
knowledge, the first formal threat model for retrieval-RAG systems. We
introduce a structured taxonomy of adversary types based on their access to
model components and data, and we formally define key threat vectors such as
document-level membership inference and data poisoning, which pose serious
privacy and integrity risks in real-world deployments. By establishing formal
definitions and attack models, our work lays the foundation for a more rigorous
and principled understanding of privacy and security in RAG systems.</details></td><td><details><summary>展开</summary>这篇论文探讨了检索增强生成（RAG）系统在隐私和安全方面的新挑战，提出了首个针对RAG系统的正式威胁模型，并定义了包括文档级成员推断和数据投毒在内的关键威胁向量，为理解和应对RAG系统的安全风险提供了理论基础。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.20190v1">STAF: Leveraging LLMs for Automated Attack Tree-Based Security Test Generation</a></td><td><details><summary>展开</summary>In modern automotive development, security testing is critical for
safeguarding systems against increasingly advanced threats. Attack trees are
widely used to systematically represent potential attack vectors, but
generating comprehensive test cases from these trees remains a labor-intensive,
error-prone task that has seen limited automation in the context of testing
vehicular systems. This paper introduces STAF (Security Test Automation
Framework), a novel approach to automating security test case generation.
Leveraging Large Language Models (LLMs) and a four-step self-corrective
Retrieval-Augmented Generation (RAG) framework, STAF automates the generation
of executable security test cases from attack trees, providing an end-to-end
solution that encompasses the entire attack surface. We particularly show the
elements and processes needed to provide an LLM to actually produce sensible
and executable automotive security test suites, along with the integration with
an automated testing framework. We further compare our tailored approach with
general purpose (vanilla) LLMs and the performance of different LLMs (namely
GPT-4.1 and DeepSeek) using our approach. We also demonstrate the method of our
operation step-by-step in a concrete case study. Our results show significant
improvements in efficiency, accuracy, scalability, and easy integration in any
workflow, marking a substantial advancement in automating automotive security
testing methodologies. Using TARAs as an input for verfication tests, we create
synergies by connecting two vital elements of a secure automotive development
process.</details></td><td><details><summary>展开</summary>这篇论文介绍了STAF框架，利用LLM和四步自校正RAG技术，自动化生成汽车安全测试用例，显著提升了测试效率、准确性及可扩展性，并对比了不同LLM（如GPT-4.1和DeepSeek）的性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.19980v1">RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis</a></td><td><details><summary>展开</summary>Clinical diagnosis is a highly specialized discipline requiring both domain
expertise and strict adherence to rigorous guidelines. While current AI-driven
medical research predominantly focuses on knowledge graphs or natural text
pretraining paradigms to incorporate medical knowledge, these approaches
primarily rely on implicitly encoded knowledge within model parameters,
neglecting task-specific knowledge required by diverse downstream tasks. To
address this limitation, we propose Retrieval-Augmented Diagnosis (RAD), a
novel framework that explicitly injects external knowledge into multimodal
models directly on downstream tasks. Specifically, RAD operates through three
key mechanisms: retrieval and refinement of disease-centered knowledge from
multiple medical sources, a guideline-enhanced contrastive loss that constrains
the latent distance between multi-modal features and guideline knowledge, and
the dual transformer decoder that employs guidelines as queries to steer
cross-modal fusion, aligning the models with clinical diagnostic workflows from
guideline acquisition to feature extraction and decision-making. Moreover,
recognizing the lack of quantitative evaluation of interpretability for
multimodal diagnostic models, we introduce a set of criteria to assess the
interpretability from both image and text perspectives. Extensive evaluations
across four datasets with different anatomies demonstrate RAD's
generalizability, achieving state-of-the-art performance. Furthermore, RAD
enables the model to concentrate more precisely on abnormal regions and
critical indicators, ensuring evidence-based, trustworthy diagnosis. Our code
is available at https://github.com/tdlhl/RAD.</details></td><td><details><summary>展开</summary>这篇论文提出了一个名为“Retrieval-Augmented Diagnosis (RAD)”的新框架，通过检索和整合多源医学知识（如疾病指南），结合对比损失和双Transformer解码器等机制，显式地将外部知识注入多模态模型，以提升临床诊断的准确性、可解释性及与工作流程的契合度，并设计了定量评估指标。该框架在多个数据集上表现优异，属于RAG技术在医疗诊断领域的扩展应用。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.19952v1">When Words Can't Capture It All: Towards Video-Based User Complaint Text Generation with Multimodal Video Complaint Dataset</a></td><td><details><summary>展开</summary>While there exists a lot of work on explainable complaint mining,
articulating user concerns through text or video remains a significant
challenge, often leaving issues unresolved. Users frequently struggle to
express their complaints clearly in text but can easily upload videos depicting
product defects (e.g., vague text such as `worst product' paired with a
5-second video depicting a broken headphone with the right earcup). This paper
formulates a new task in the field of complaint mining to aid the common users'
need to write an expressive complaint, which is Complaint Description from
Videos (CoD-V) (e.g., to help the above user articulate her complaint about the
defective right earcup). To this end, we introduce ComVID, a video complaint
dataset containing 1,175 complaint videos and the corresponding descriptions,
also annotated with the emotional state of the complainer. Additionally, we
present a new complaint retention (CR) evaluation metric that discriminates the
proposed (CoD-V) task against standard video summary generation and description
tasks. To strengthen this initiative, we introduce a multimodal
Retrieval-Augmented Generation (RAG) embedded VideoLLaMA2-7b model, designed to
generate complaints while accounting for the user's emotional state. We conduct
a comprehensive evaluation of several Video Language Models on several tasks
(pre-trained and fine-tuned versions) with a range of established evaluation
metrics, including METEOR, perplexity, and the Coleman-Liau readability score,
among others. Our study lays the foundation for a new research direction to
provide a platform for users to express complaints through video. Dataset and
resources are available at: https://github.com/sarmistha-D/CoD-V.</details></td><td><details><summary>展开</summary>本文提出了一种新的投诉挖掘任务——视频投诉描述（CoD-V），旨在帮助用户通过视频表达投诉内容，并引入了一个包含1175条投诉视频及对应描述的数据集ComVID。作者提出了一种新的评估指标CR，并开发了一种基于检索增强生成（RAG）的多模态模型VideoLLaMA2-7b，用于生成考虑用户情感状态的投诉描述。研究通过多种评估指标对模型性能进行了全面验证，为该领域的新研究方向奠定了基础。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.19931v1">Documentation Retrieval Improves Planning Language Generation</a></td><td><details><summary>展开</summary>Certain strong LLMs have shown promise for zero-shot formal planning by
generating planning languages like PDDL. Yet, performance of most open-source
models under 50B parameters has been reported to be close to zero due to the
low-resource nature of these languages. We significantly improve their
performance via a series of lightweight pipelines that integrates documentation
retrieval with modular code generation and error refinement. With models like
Llama-4-Maverick, our best pipeline improves plan correctness from 0\% to over
80\% on the common BlocksWorld domain. However, while syntactic errors are
substantially reduced, semantic errors persist in more challenging domains,
revealing fundamental limitations in current models' reasoning
capabilities.\footnote{Our code and data can be found at
https://github.com/Nangxxxxx/PDDL-RAG</details></td><td><details><summary>展开</summary>该论文提出了一种通过整合文档检索、模块化代码生成和错误修正的轻量级流程，显著提升了中小型开源LLMs在零样本形式化规划任务中的表现（如生成PDDL规划语言），尤其在BlocksWorld领域将正确率从0%提升至80%以上，但指出模型在复杂领域的语义推理仍存在根本性局限。其方法核心涉及检索增强技术（代码库标注了PDDL-RAG）。</details></td></tr></tbody></table>

### 📅 2025-09-23
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.19218v1">HyKid: An Open MRI Dataset with Expert-Annotated Multi-Structure and Choroid Plexus in Pediatric Hydrocephalus</a></td><td><details><summary>展开</summary>Evaluation of hydrocephalus in children is challenging, and the related
research is limited by a lack of publicly available, expert-annotated datasets,
particularly those with segmentation of the choroid plexus. To address this, we
present HyKid, an open-source dataset from 48 pediatric patients with
hydrocephalus. 3D MRIs were provided with 1mm isotropic resolution, which was
reconstructed from routine low-resolution images using a slice-to-volume
algorithm. Manually corrected segmentations of brain tissues, including white
matter, grey matter, lateral ventricle, external CSF, and the choroid plexus,
were provided by an experienced neurologist. Additionally, structured data was
extracted from clinical radiology reports using a Retrieval-Augmented
Generation framework. The strong correlation between choroid plexus volume and
total CSF volume provided a potential biomarker for hydrocephalus evaluation,
achieving excellent performance in a predictive model (AUC = 0.87). The
proposed HyKid dataset provided a high-quality benchmark for neuroimaging
algorithms development, and it revealed the choroid plexus-related features in
hydrocephalus assessments. Our datasets are publicly available at
https://www.synapse.org/Synapse:syn68544889.</details></td><td><details><summary>展开</summary>这篇论文介绍了HyKid数据集，一个针对儿童脑积水的开源数据集，包含高分辨率3D MRI图像和专家手动校正的分割标注。研究利用RAG框架从临床放射学报告中提取结构化数据，并发现了脉络丛体积与脑脊液总量的相关性可作为脑积水评估的生物标志物，预测模型表现优异（AUC=0.87）。该数据集为神经影像算法开发提供了高质量基准。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.19209v1">A Knowledge Graph and a Tripartite Evaluation Framework Make Retrieval-Augmented Generation Scalable and Transparent</a></td><td><details><summary>展开</summary>Large Language Models (LLMs) have significantly enhanced conversational
Artificial Intelligence(AI) chatbots; however, domain-specific accuracy and the
avoidance of factual inconsistencies remain pressing challenges, particularly
for large datasets. Designing an effective chatbot with appropriate methods and
evaluating its effectiveness is among the challenges in this domain. This study
presents a Retrieval Augmented Generation (RAG) chatbot that harnesses a
knowledge graph and vector search retrieval to deliver precise, context-rich
responses in an exemplary use case from over high-volume engineering
project-related emails, thereby minimising the need for document chunking. A
central innovation of this work is the introduction of RAG Evaluation
(RAG-Eval), a novel chain-of-thought LLM-based tripartite evaluation framework
specifically developed to assess RAG applications. This framework operates in
parallel with the chatbot, jointly assessing the user's query, the retrieved
document, and the generated response, enabling a holistic evaluation across
multiple quality metrics like query relevance, factual accuracy, coverage,
coherence and fluency. The resulting scoring system is provided directly to
users as a confidence score (1 to 100%), enabling quick identification of
possible misaligned or incomplete answers. This proposed approach promotes
transparency and rapid verification by incorporating metadata email IDs,
timestamps into responses. Experimental comparisons against BERTScore and
G-EVAL for summarisation evaluation tasks confirm its effectiveness, and
empirical analysis also shows RAG-Eval reliably detects factual gaps and query
mismatches, thereby fostering trust in high demand, data centric environments.
These findings highlight a scalable path for developing accurate,
user-verifiable chatbots that bridge the gap between high-level conversational
fluency and factual accuracy.</details></td><td><details><summary>展开</summary>这篇论文提出了一种基于检索增强生成（RAG）的聊天机器人，结合知识图谱和向量搜索检索技术，从大规模工程相关邮件数据中生成精准且上下文丰富的回答，减少文档分块的需求。论文还创新地引入了RAG-Eval，一个基于大语言模型的三方评估框架，旨在评估RAG应用的查询相关性、事实准确性、覆盖范围等质量指标，并通过置信度分数和元数据增强透明度。实验证明该方法在高效性和可信度上优于BERTScore和G-EVAL。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.18868v1">Memory in Large Language Models: Mechanisms, Evaluation and Evolution</a></td><td><details><summary>展开</summary>Under a unified operational definition, we define LLM memory as a persistent
state written during pretraining, finetuning, or inference that can later be
addressed and that stably influences outputs. We propose a four-part taxonomy
(parametric, contextual, external, procedural/episodic) and a memory quadruple
(location, persistence, write/access path, controllability). We link mechanism,
evaluation, and governance via the chain write -> read -> inhibit/update. To
avoid distorted comparisons across heterogeneous setups, we adopt a
three-setting protocol (parametric only, offline retrieval, online retrieval)
that decouples capability from information availability on the same data and
timeline. On this basis we build a layered evaluation: parametric (closed-book
recall, edit differential, memorization/privacy), contextual (position curves
and the mid-sequence drop), external (answer correctness vs snippet
attribution/faithfulness), and procedural/episodic (cross-session consistency
and timeline replay, E MARS+). The framework integrates temporal governance and
leakage auditing (freshness hits, outdated answers, refusal slices) and
uncertainty reporting via inter-rater agreement plus paired tests with
multiple-comparison correction. For updating and forgetting, we present DMM
Gov: coordinating DAPT/TAPT, PEFT, model editing (ROME, MEND, MEMIT, SERAC),
and RAG to form an auditable loop covering admission thresholds, rollout,
monitoring, rollback, and change audits, with specs for timeliness, conflict
handling, and long-horizon consistency. Finally, we give four testable
propositions: minimum identifiability; a minimal evaluation card; causally
constrained editing with verifiable forgetting; and when retrieval with
small-window replay outperforms ultra-long-context reading. This yields a
reproducible, comparable, and governable coordinate system for research and
deployment.</details></td><td><details><summary>展开</summary>这篇论文提出了一个关于LLM记忆的统一操作定义和四部分分类法（参数化、上下文、外部、过程/情景），并设计了一个评估框架，其中包括外部记忆（与RAG相关）的评估标准，如答案正确性与片段归因/忠实性。论文还讨论了DMM Gov框架，协调包括RAG在内的多种技术形成一个可审计的循环，用于更新和遗忘。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.18667v1">TERAG: Token-Efficient Graph-Based Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Graph-based Retrieval-augmented generation (RAG) has become a widely studied
approach for improving the reasoning, accuracy, and factuality of Large
Language Models. However, many existing graph-based RAG systems overlook the
high cost associated with LLM token usage during graph construction, hindering
large-scale adoption. To address this, we propose TERAG, a simple yet effective
framework designed to build informative graphs at a significantly lower cost.
Inspired by HippoRAG, we incorporate Personalized PageRank (PPR) during the
retrieval phase, and we achieve at least 80% of the accuracy of widely used
graph-based RAG methods while consuming only 3%-11% of the output tokens.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为TERAG的低成本图结构检索增强生成框架，通过结合个性化PageRank（PPR）优化检索阶段，大幅减少LLM建图时的token消耗（降至3%-11%），同时保持主流图基RAG方法80%以上的准确性。</details></td></tr></tbody></table>

### 📅 2025-09-22
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.18054v1">A Knowledge Graph-based Retrieval-Augmented Generation Framework for Algorithm Selection in the Facility Layout Problem</a></td><td><details><summary>展开</summary>Selecting a solution algorithm for the Facility Layout Problem (FLP), an
NP-hard optimization problem with a multiobjective trade-off, is a complex task
that requires deep expert knowledge. The performance of a given algorithm
depends on specific problem characteristics such as its scale, objectives, and
constraints. This creates a need for a data-driven recommendation method to
guide algorithm selection in automated design systems. This paper introduces a
new recommendation method to make such expertise accessible, based on a
Knowledge Graph-based Retrieval-Augmented Generation (KG RAG) framework. To
address this, a domain-specific knowledge graph is constructed from published
literature. The method then employs a multi-faceted retrieval mechanism to
gather relevant evidence from this knowledge graph using three distinct
approaches, which include a precise graph-based search, flexible vector-based
search, and high-level cluster-based search. The retrieved evidence is utilized
by a Large Language Model (LLM) to generate algorithm recommendations with
data-driven reasoning. The proposed KG-RAG method is compared against a
commercial LLM chatbot with access to the knowledge base as a table, across a
series of diverse, real-world FLP test cases. Based on recommendation accuracy
and reasoning capability, the proposed method performed significantly better
than the commercial LLM chatbot.</details></td><td><details><summary>展开</summary>这篇论文提出了一种基于知识图谱的检索增强生成（KG-RAG）框架，用于为设施布局问题（FLP）推荐合适的算法。该方法通过构建领域特定的知识图谱，结合多方面的检索机制（包括基于图的精确搜索、基于向量的灵活搜索和基于聚类的高级搜索），利用大语言模型（LLM）生成算法推荐，并在真实FLP案例中验证了其优于商用LLM聊天机器人的性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.17788v1">One Agent to Serve All: a Lite-Adaptive Stylized AI Assistant for Millions of Multi-Style Official Accounts</a></td><td><details><summary>展开</summary>Conversational agents deployed in industrial-scale official account platforms
must generate responses that are both contextually grounded and stylistically
aligned-requirements that existing methods struggle to meet. Chain-of-thought
(CoT) prompting induces significant latency due to multi-turn reasoning;
per-account fine-tuning is computationally prohibitive; and long prompt-based
methods degrade the model's ability to grasp injected context and style. In
this paper, we propose WeStar, a lite-adaptive framework for stylized
contextual question answering that scales to millions of official accounts.
WeStar combines context-grounded generation via RAG with style-aware generation
using Parametric RAG (PRAG), where LoRA modules are dynamically activated per
style cluster. Our contributions are fourfold: (1) We introduce WeStar, a
unified framework capable of serving large volumes of official accounts with
minimal overhead. (2) We propose a multi-dimensional, cluster-based parameter
sharing scheme that enables compact style representation while preserving
stylistic diversity. (3) We develop a style-enhanced Direct Preference
Optimization (SeDPO) method to optimize each style cluster's parameters for
improved generation quality. (4) Experiments on a large-scale industrial
dataset validate the effectiveness and efficiency of WeStar, underscoring its
pracitical value in real-world deployment.</details></td><td><details><summary>展开</summary>这篇论文提出了WeStar框架，结合RAG和Parametric RAG（PRAG）技术，通过动态激活LoRA模块实现风格化上下文问答，旨在为海量官方账号提供低延迟、高适应性的生成解决方案。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.17671v1">Turk-LettuceDetect: A Hallucination Detection Models for Turkish RAG Applications</a></td><td><details><summary>展开</summary>The widespread adoption of Large Language Models (LLMs) has been hindered by
their tendency to hallucinate, generating plausible but factually incorrect
information. While Retrieval-Augmented Generation (RAG) systems attempt to
address this issue by grounding responses in external knowledge, hallucination
remains a persistent challenge, particularly for morphologically complex,
low-resource languages like Turkish. This paper introduces Turk-LettuceDetect,
the first suite of hallucination detection models specifically designed for
Turkish RAG applications. Building on the LettuceDetect framework, we formulate
hallucination detection as a token-level classification task and fine-tune
three distinct encoder architectures: a Turkish-specific ModernBERT,
TurkEmbed4STS, and multilingual EuroBERT. These models were trained on a
machine-translated version of the RAGTruth benchmark dataset containing 17,790
instances across question answering, data-to-text generation, and summarization
tasks. Our experimental results show that the ModernBERT-based model achieves
an F1-score of 0.7266 on the complete test set, with particularly strong
performance on structured tasks. The models maintain computational efficiency
while supporting long contexts up to 8,192 tokens, making them suitable for
real-time deployment. Comparative analysis reveals that while state-of-the-art
LLMs demonstrate high recall, they suffer from low precision due to
over-generation of hallucinated content, underscoring the necessity of
specialized detection mechanisms. By releasing our models and translated
dataset, this work addresses a critical gap in multilingual NLP and establishes
a foundation for developing more reliable and trustworthy AI applications for
Turkish and other languages.</details></td><td><details><summary>展开</summary>该论文针对土耳其语等低资源语言中RAG系统的幻觉问题，提出了首个土耳其语专用幻觉检测模型套件Turk-LettuceDetect。通过微调三种编码器架构并使用机器翻译的基准数据集进行训练，重点解决了问答、数据到文本生成和摘要任务中的幻觉检测问题，实验表明其模型在保持计算效率的同时有效提升了检测性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.17544v1">A Multimodal Conversational Assistant for the Characterization of Agricultural Plots from Geospatial Open Data</a></td><td><details><summary>展开</summary>The increasing availability of open Earth Observation (EO) and agricultural
datasets holds great potential for supporting sustainable land management.
However, their high technical entry barrier limits accessibility for non-expert
users. This study presents an open-source conversational assistant that
integrates multimodal retrieval and large language models (LLMs) to enable
natural language interaction with heterogeneous agricultural and geospatial
data. The proposed architecture combines orthophotos, Sentinel-2 vegetation
indices, and user-provided documents through retrieval-augmented generation
(RAG), allowing the system to flexibly determine whether to rely on multimodal
evidence, textual knowledge, or both in formulating an answer. To assess
response quality, we adopt an LLM-as-a-judge methodology using Qwen3-32B in a
zero-shot, unsupervised setting, applying direct scoring in a multi-dimensional
quantitative evaluation framework. Preliminary results show that the system is
capable of generating clear, relevant, and context-aware responses to
agricultural queries, while remaining reproducible and scalable across
geographic regions. The primary contributions of this work include an
architecture for fusing multimodal EO and textual knowledge sources, a
demonstration of lowering the barrier to access specialized agricultural
information through natural language interaction, and an open and reproducible
design.</details></td><td><details><summary>展开</summary>这篇论文提出了一种结合多模态检索与大语言模型的开源对话助手，利用RAG技术整合农业与地理空间数据，通过自然语言交互降低非专家用户使用门槛，并采用LLM评估方法验证响应质量。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.17486v1">AttnComp: Attention-Guided Adaptive Context Compression for Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Retrieval-augmented generation improves the factual accuracy of Large
Language Models (LLMs) by incorporating external context, but often suffers
from irrelevant retrieved content that hinders effectiveness. Context
compression addresses this issue by filtering out irrelevant information from
context before LLM generation. However, existing methods struggle to adaptively
adjust compression rates for different context, maintain low latency and
integrate information across multiple documents. To overcome these limitations,
We introduce AttnComp, an adaptive, efficient and context-aware compression
framework. By leveraging the attention mechanism of LLMs to identify relevant
information, AttnComp employs a Top-P compression algorithm to retain the
minimal set of documents whose cumulative attention weights exceeds a
predefined threshold. In addition to compression, AttnComp estimates response
confidence by assessing the overall relevance of the retrieved content,
enabling users to gauge response reliability. Experiments demonstrate that
AttnComp outperforms existing compression methods and uncompressed baselines,
achieving higher accuracy with substantial compression rates and lower latency.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为AttnComp的自适应、高效且上下文感知的压缩框架，用于解决RAG中检索内容可能无关导致效果下降的问题。该框架利用大语言模型的注意力机制识别相关信息，并通过Top-P压缩算法保留关键文档，同时还能评估响应置信度以提升可靠性，实验证明其性能优于现有压缩方法和未压缩基线。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.17395v1">FinDebate: Multi-Agent Collaborative Intelligence for Financial Analysis</a></td><td><details><summary>展开</summary>We introduce FinDebate, a multi-agent framework for financial analysis,
integrating collaborative debate with domain-specific Retrieval-Augmented
Generation (RAG). Five specialized agents, covering earnings, market,
sentiment, valuation, and risk, run in parallel to synthesize evidence into
multi-dimensional insights. To mitigate overconfidence and improve reliability,
we introduce a safe debate protocol that enables agents to challenge and refine
initial conclusions while preserving coherent recommendations. Experimental
results, based on both LLM-based and human evaluations, demonstrate the
framework's efficacy in producing high-quality analysis with calibrated
confidence levels and actionable investment strategies across multiple time
horizons.</details></td><td><details><summary>展开</summary>这篇文章介绍了一个名为FinDebate的多代理框架，用于金融分析，结合了协作辩论和特定领域的检索增强生成（RAG）。五个专业代理并行工作，将证据合成为多维度的见解，并通过安全辩论协议减少过度自信并提高可靠性。实验结果表明该框架能生成高质量的分析和可操作的投资策略。</details></td></tr></tbody></table>

### 📅 2025-09-21
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.17197v1">SignalLLM: A General-Purpose LLM Agent Framework for Automated Signal Processing</a></td><td><details><summary>展开</summary>Modern signal processing (SP) pipelines, whether model-based or data-driven,
often constrained by complex and fragmented workflow, rely heavily on expert
knowledge and manual engineering, and struggle with adaptability and
generalization under limited data. In contrast, Large Language Models (LLMs)
offer strong reasoning capabilities, broad general-purpose knowledge,
in-context learning, and cross-modal transfer abilities, positioning them as
powerful tools for automating and generalizing SP workflows. Motivated by these
potentials, we introduce SignalLLM, the first general-purpose LLM-based agent
framework for general SP tasks. Unlike prior LLM-based SP approaches that are
limited to narrow applications or tricky prompting, SignalLLM introduces a
principled, modular architecture. It decomposes high-level SP goals into
structured subtasks via in-context learning and domain-specific retrieval,
followed by hierarchical planning through adaptive retrieval-augmented
generation (RAG) and refinement; these subtasks are then executed through
prompt-based reasoning, cross-modal reasoning, code synthesis, model
invocation, or data-driven LLM-assisted modeling. Its generalizable design
enables the flexible selection of problem solving strategies across different
signal modalities, task types, and data conditions. We demonstrate the
versatility and effectiveness of SignalLLM through five representative tasks in
communication and sensing, such as radar target detection, human activity
recognition, and text compression. Experimental results show superior
performance over traditional and existing LLM-based methods, particularly in
few-shot and zero-shot settings.</details></td><td><details><summary>展开</summary>这篇论文介绍了SignalLLM，一个基于大型语言模型（LLM）的通用信号处理（SP）代理框架，它通过引入模块化架构和检索增强生成（RAG）技术，将高层SP目标分解为结构化的子任务，并结合领域特定检索、分层规划和多模态推理，实现了跨信号模态和任务类型的灵活问题解决。实验证明其在少样本和零样本设定下的优越性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.17066v1">RALLM-POI: Retrieval-Augmented LLM for Zero-shot Next POI Recommendation with Geographical Reranking</a></td><td><details><summary>展开</summary>Next point-of-interest (POI) recommendation predicts a user's next
destination from historical movements. Traditional models require intensive
training, while LLMs offer flexible and generalizable zero-shot solutions but
often generate generic or geographically irrelevant results due to missing
trajectory and spatial context. To address these issues, we propose RALLM-POI,
a framework that couples LLMs with retrieval-augmented generation and
self-rectification. We first propose a Historical Trajectory Retriever (HTR)
that retrieves relevant past trajectories to serve as contextual references,
which are then reranked by a Geographical Distance Reranker (GDR) for
prioritizing spatially relevant trajectories. Lastly, an Agentic LLM Rectifier
(ALR) is designed to refine outputs through self-reflection. Without additional
training, RALLM-POI achieves substantial accuracy gains across three real-world
Foursquare datasets, outperforming both conventional and LLM-based baselines.
Code is released at https://github.com/LKRcrocodile/RALLM-POI.</details></td><td><details><summary>展开</summary>该论文提出RALLM-POI框架，通过检索增强生成（RAG）和自矫正技术改进基于大语言模型（LLM）的下一个兴趣点（POI）推荐。框架包含历史轨迹检索器（HTR）、地理距离重排序器（GDR）和LLM代理矫正器（ALR），利用相关轨迹作为上下文输入LLM并自我优化输出，无需额外训练即显著提升推荐准确性，在Foursquare数据集上超越传统和LLM基线方法。</details></td></tr></tbody></table>

### 📅 2025-09-20
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.16780v1">Comparing RAG and GraphRAG for Page-Level Retrieval Question Answering on Math Textbook</a></td><td><details><summary>展开</summary>Technology-enhanced learning environments often help students retrieve
relevant learning content for questions arising during self-paced study. Large
language models (LLMs) have emerged as novel aids for information retrieval
during learning. While LLMs are effective for general-purpose
question-answering, they typically lack alignment with the domain knowledge of
specific course materials such as textbooks and slides. We investigate
Retrieval-Augmented Generation (RAG) and GraphRAG, a knowledge graph-enhanced
RAG approach, for page-level question answering in an undergraduate mathematics
textbook. While RAG has been effective for retrieving discrete, contextually
relevant passages, GraphRAG may excel in modeling interconnected concepts and
hierarchical knowledge structures. We curate a dataset of 477 question-answer
pairs, each tied to a distinct textbook page. We then compare the standard
embedding-based RAG methods to GraphRAG for evaluating both retrieval
accuracy-whether the correct page is retrieved-and generated answer quality via
F1 scores. Our findings show that embedding-based RAG achieves higher retrieval
accuracy and better F1 scores compared to GraphRAG, which tends to retrieve
excessive and sometimes irrelevant content due to its entity-based structure.
We also explored re-ranking the retrieved pages with LLM and observed mixed
results, including performance drop and hallucinations when dealing with larger
context windows. Overall, this study highlights both the promises and
challenges of page-level retrieval systems in educational contexts, emphasizing
the need for more refined retrieval methods to build reliable AI tutoring
solutions in providing reference page numbers.</details></td><td><details><summary>展开</summary>这篇论文研究了检索增强生成（RAG）和知识图谱增强的GraphRAG方法在本科数学教科书页级问答中的应用，比较了它们在检索准确性和生成答案质量上的表现，发现基于嵌入的RAG优于GraphRAG，并探讨了教育场景中检索系统的潜力与挑战。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.16584v1">From Scores to Steps: Diagnosing and Improving LLM Performance in Evidence-Based Medical Calculations</a></td><td><details><summary>展开</summary>Large language models (LLMs) have demonstrated promising performance on
medical benchmarks; however, their ability to perform medical calculations, a
crucial aspect of clinical decision-making, remains underexplored and poorly
evaluated. Existing benchmarks often assess only the final answer with a wide
numerical tolerance, overlooking systematic reasoning failures and potentially
causing serious clinical misjudgments. In this work, we revisit medical
calculation evaluation with a stronger focus on clinical trustworthiness.
First, we clean and restructure the MedCalc-Bench dataset and propose a new
step-by-step evaluation pipeline that independently assesses formula selection,
entity extraction, and arithmetic computation. Under this granular framework,
the accuracy of GPT-4o drops from 62.7% to 43.6%, revealing errors masked by
prior evaluations. Second, we introduce an automatic error analysis framework
that generates structured attribution for each failure mode. Human evaluation
confirms its alignment with expert judgment, enabling scalable and explainable
diagnostics. Finally, we propose a modular agentic pipeline, MedRaC, that
combines retrieval-augmented generation and Python-based code execution.
Without any fine-tuning, MedRaC improves the accuracy of different LLMs from
16.35% up to 53.19%. Our work highlights the limitations of current benchmark
practices and proposes a more clinically faithful methodology. By enabling
transparent and transferable reasoning evaluation, we move closer to making
LLM-based systems trustworthy for real-world medical applications.</details></td><td><details><summary>展开</summary>这篇论文探讨大语言模型（LLM）在医学计算任务中的性能问题，提出改进评估方法（MedCalc-Bench数据集和分步评估流程），发现现有评测掩盖系统性错误（如GPT-4o准确率从62.7%降至43.6%）。作者开发了自动错误分析框架，并提出结合检索增强生成（RAG）和Python代码执行的模块化流程MedRaC，显著提升不同LLM的准确率（最高达53.19%）。研究强调临床可信度评估的重要性，推动LLM在真实医疗场景的可靠应用。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.16508v1">Federated Learning with Ad-hoc Adapter Insertions: The Case of Soft-Embeddings for Training Classifier-as-Retriever</a></td><td><details><summary>展开</summary>When existing retrieval-augmented generation (RAG) solutions are intended to
be used for new knowledge domains, it is necessary to update their encoders,
which are taken to be pretrained large language models (LLMs). However, fully
finetuning these large models is compute- and memory-intensive, and even
infeasible when deployed on resource-constrained edge devices. We propose a
novel encoder architecture in this work that addresses this limitation by using
a frozen small language model (SLM), which satisfies the memory constraints of
edge devices, and inserting a small adapter network before the transformer
blocks of the SLM. The trainable adapter takes the token embeddings of the new
corpus and learns to produce enhanced soft embeddings for it, while requiring
significantly less compute power to update than full fine-tuning. We further
propose a novel retrieval mechanism by attaching a classifier head to the SLM
encoder, which is trained to learn a similarity mapping of the input embeddings
to their corresponding documents. Finally, to enable the online fine-tuning of
both (i) the encoder soft embeddings and (ii) the classifier-as-retriever on
edge devices, we adopt federated learning (FL) and differential privacy (DP) to
achieve an efficient, privacy-preserving, and product-grade training solution.
We conduct a theoretical analysis of our methodology, establishing convergence
guarantees under mild assumptions on gradient variance when deployed for
general smooth nonconvex loss functions. Through extensive numerical
experiments, we demonstrate (i) the efficacy of obtaining soft embeddings to
enhance the encoder, (ii) training a classifier to improve the retriever, and
(iii) the role of FL in achieving speedup.</details></td><td><details><summary>展开</summary>本文提出了一种适用于边缘设备的新型RAG编码器架构，采用冻结的小语言模型（SLM）和适配器网络来减少计算和内存需求，同时引入基于分类器的检索机制和联邦学习（FL）进行隐私保护和高效在线微调，理论分析和实验验证了方法的有效性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.16502v1">GRIL: Knowledge Graph Retrieval-Integrated Learning with Large Language Models</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) has significantly mitigated the
hallucinations of Large Language Models (LLMs) by grounding the generation with
external knowledge. Recent extensions of RAG to graph-based retrieval offer a
promising direction, leveraging the structural knowledge for multi-hop
reasoning. However, existing graph RAG typically decouples retrieval and
reasoning processes, which prevents the retriever from adapting to the
reasoning needs of the LLM. They also struggle with scalability when performing
multi-hop expansion over large-scale graphs, or depend heavily on annotated
ground-truth entities, which are often unavailable in open-domain settings. To
address these challenges, we propose a novel graph retriever trained end-to-end
with LLM, which features an attention-based growing and pruning mechanism,
adaptively navigating multi-hop relevant entities while filtering out noise.
Within the extracted subgraph, structural knowledge and semantic features are
encoded via soft tokens and the verbalized graph, respectively, which are
infused into the LLM together, thereby enhancing its reasoning capability and
facilitating interactive joint training of the graph retriever and the LLM
reasoner. Experimental results across three QA benchmarks show that our
approach consistently achieves state-of-the-art performance, validating the
strength of joint graph-LLM optimization for complex reasoning tasks. Notably,
our framework eliminates the need for predefined ground-truth entities by
directly optimizing the retriever using LLM logits as implicit feedback, making
it especially effective in open-domain settings.</details></td><td><details><summary>展开</summary>这篇论文提出了一种新颖的端到端训练方法，将基于图的检索器与大语言模型（LLM）联合优化，通过注意力机制动态导航多跳相关实体并过滤噪声，同时融合结构知识和语义特征以增强LLM的推理能力，显著提升了开放领域复杂问答任务的表现。</details></td></tr></tbody></table>

### 📅 2025-09-19
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.16112v1">CodeRAG: Finding Relevant and Necessary Knowledge for Retrieval-Augmented Repository-Level Code Completion</a></td><td><details><summary>展开</summary>Repository-level code completion automatically predicts the unfinished code
based on the broader information from the repository. Recent strides in Code
Large Language Models (code LLMs) have spurred the development of
repository-level code completion methods, yielding promising results.
Nevertheless, they suffer from issues such as inappropriate query construction,
single-path code retrieval, and misalignment between code retriever and code
LLM. To address these problems, we introduce CodeRAG, a framework tailored to
identify relevant and necessary knowledge for retrieval-augmented
repository-level code completion. Its core components include log probability
guided query construction, multi-path code retrieval, and preference-aligned
BestFit reranking. Extensive experiments on benchmarks ReccEval and CCEval
demonstrate that CodeRAG significantly and consistently outperforms
state-of-the-art methods. The implementation of CodeRAG is available at
https://github.com/KDEGroup/CodeRAG.</details></td><td><details><summary>展开</summary>这篇论文提出了一个名为CodeRAG的框架，旨在解决现有仓库级代码补全方法中存在的问题，如不恰当的查询构建、单一路径的代码检索以及代码检索器与大语言模型之间的不对齐。CodeRAG通过概率引导的查询构建、多路径代码检索和偏好对齐的BestFit重排序等核心组件，提升了检索增强的仓库级代码补全的性能。实验证明，CodeRAG在多个基准测试中显著优于现有方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.15883v1">RACap: Relation-Aware Prompting for Lightweight Retrieval-Augmented Image Captioning</a></td><td><details><summary>展开</summary>Recent retrieval-augmented image captioning methods incorporate external
knowledge to compensate for the limitations in comprehending complex scenes.
However, current approaches face challenges in relation modeling: (1) the
representation of semantic prompts is too coarse-grained to capture
fine-grained relationships; (2) these methods lack explicit modeling of image
objects and their semantic relationships. To address these limitations, we
propose RACap, a relation-aware retrieval-augmented model for image captioning,
which not only mines structured relation semantics from retrieval captions, but
also identifies heterogeneous objects from the image. RACap effectively
retrieves structured relation features that contain heterogeneous visual
information to enhance the semantic consistency and relational expressiveness.
Experimental results show that RACap, with only 10.8M trainable parameters,
achieves superior performance compared to previous lightweight captioning
models.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为RACap的关系感知检索增强图像描述生成模型，通过从检索到的描述中挖掘结构化关系语义并识别图像中的异构对象，以提升语义一致性和关系表达能力，实验显示其在轻量级模型中表现优异。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.15577v1">Relevance to Utility: Process-Supervised Rewrite for RAG</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation systems often suffer from a gap between
optimizing retrieval relevance and generative utility: retrieved documents may
be topically relevant but still lack the content needed for effective reasoning
during generation. While existing "bridge" modules attempt to rewrite the
retrieved text for better generation, we show how they fail to capture true
document utility. In this work, we propose R2U, with a key distinction of
directly optimizing to maximize the probability of generating a correct answer
through process supervision. As such direct observation is expensive, we also
propose approximating an efficient distillation pipeline by scaling the
supervision from LLMs, which helps the smaller rewriter model generalize
better. We evaluate our method across multiple open-domain question-answering
benchmarks. The empirical results demonstrate consistent improvements over
strong bridging baselines.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为R2U的方法，旨在解决RAG系统中检索相关性与生成效用之间的不一致问题。通过直接优化生成正确答案的概率，并利用LLM的监督信号来高效训练较小的重写模型，论文在多个开放域问答基准测试中展示了优于现有基线方法的性能。</details></td></tr></tbody></table>

### 📅 2025-09-18
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.15211v1">What's the Best Way to Retrieve Slides? A Comparative Study of Multimodal, Caption-Based, and Hybrid Retrieval Techniques</a></td><td><details><summary>展开</summary>Slide decks, serving as digital reports that bridge the gap between
presentation slides and written documents, are a prevalent medium for conveying
information in both academic and corporate settings. Their multimodal nature,
combining text, images, and charts, presents challenges for retrieval-augmented
generation systems, where the quality of retrieval directly impacts downstream
performance. Traditional approaches to slide retrieval often involve separate
indexing of modalities, which can increase complexity and lose contextual
information. This paper investigates various methodologies for effective slide
retrieval, including visual late-interaction embedding models like ColPali, the
use of visual rerankers, and hybrid retrieval techniques that combine dense
retrieval with BM25, further enhanced by textual rerankers and fusion methods
like Reciprocal Rank Fusion. A novel Vision-Language Models-based captioning
pipeline is also evaluated, demonstrating significantly reduced embedding
storage requirements compared to visual late-interaction techniques, alongside
comparable retrieval performance. Our analysis extends to the practical aspects
of these methods, evaluating their runtime performance and storage demands
alongside retrieval efficacy, thus offering practical guidance for the
selection and development of efficient and robust slide retrieval systems for
real-world applications.</details></td><td><details><summary>展开</summary>本文研究针对多模态幻灯片（包含文本、图像和图表）的高效检索方法，探讨了视觉延迟交互嵌入模型、视觉重排序器、混合检索技术（结合稠密检索与BM25）等方案，并提出基于视觉语言模型的标题生成流程，在保证检索性能的同时显著降低存储需求，为RAG系统中幻灯片检索的实际应用提供效能评估与开发指导。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.15159v1">AIP: Subverting Retrieval-Augmented Generation via Adversarial Instructional Prompt</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by
retrieving relevant documents from external sources to improve factual accuracy
and verifiability. However, this reliance introduces new attack surfaces within
the retrieval pipeline, beyond the LLM itself. While prior RAG attacks have
exposed such vulnerabilities, they largely rely on manipulating user queries,
which is often infeasible in practice due to fixed or protected user inputs.
This narrow focus overlooks a more realistic and stealthy vector: instructional
prompts, which are widely reused, publicly shared, and rarely audited. Their
implicit trust makes them a compelling target for adversaries to manipulate RAG
behavior covertly.
  We introduce a novel attack for Adversarial Instructional Prompt (AIP) that
exploits adversarial instructional prompts to manipulate RAG outputs by subtly
altering retrieval behavior. By shifting the attack surface to the
instructional prompts, AIP reveals how trusted yet seemingly benign interface
components can be weaponized to degrade system integrity. The attack is crafted
to achieve three goals: (1) naturalness, to evade user detection; (2) utility,
to encourage use of prompts; and (3) robustness, to remain effective across
diverse query variations. We propose a diverse query generation strategy that
simulates realistic linguistic variation in user queries, enabling the
discovery of prompts that generalize across paraphrases and rephrasings.
Building on this, a genetic algorithm-based joint optimization is developed to
evolve adversarial prompts by balancing attack success, clean-task utility, and
stealthiness. Experimental results show that AIP achieves up to 95.23% ASR
while preserving benign functionality. These findings uncover a critical and
previously overlooked vulnerability in RAG systems, emphasizing the need to
reassess the shared instructional prompts.</details></td><td><details><summary>展开</summary>这篇论文探讨了RAG系统中的新型攻击方式Adversarial Instructional Prompt (AIP)，通过操纵广泛复用且未被审计的指令提示（而非直接篡改用户查询），隐秘地改变检索行为以操控输出。研究提出基于生成多样查询和遗传算法的联合优化方法，揭示RAG中基于指令提示的安全漏洞，实验显示AIP攻击成功率高达95.23%且保持正常功能，强调了重新评估共享提示风险的必要性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.14956v1">Sentinel Agents for Secure and Trustworthy Agentic AI in Multi-Agent Systems</a></td><td><details><summary>展开</summary>This paper proposes a novel architectural framework aimed at enhancing
security and reliability in multi-agent systems (MAS). A central component of
this framework is a network of Sentinel Agents, functioning as a distributed
security layer that integrates techniques such as semantic analysis via large
language models (LLMs), behavioral analytics, retrieval-augmented verification,
and cross-agent anomaly detection. Such agents can potentially oversee
inter-agent communications, identify potential threats, enforce privacy and
access controls, and maintain comprehensive audit records. Complementary to the
idea of Sentinel Agents is the use of a Coordinator Agent. The Coordinator
Agent supervises policy implementation, and manages agent participation. In
addition, the Coordinator also ingests alerts from Sentinel Agents. Based on
these alerts, it can adapt policies, isolate or quarantine misbehaving agents,
and contain threats to maintain the integrity of the MAS ecosystem. This
dual-layered security approach, combining the continuous monitoring of Sentinel
Agents with the governance functions of Coordinator Agents, supports dynamic
and adaptive defense mechanisms against a range of threats, including prompt
injection, collusive agent behavior, hallucinations generated by LLMs, privacy
breaches, and coordinated multi-agent attacks. In addition to the architectural
design, we present a simulation study where 162 synthetic attacks of different
families (prompt injection, hallucination, and data exfiltration) were injected
into a multi-agent conversational environment. The Sentinel Agents successfully
detected the attack attempts, confirming the practical feasibility of the
proposed monitoring approach. The framework also offers enhanced system
observability, supports regulatory compliance, and enables policy evolution
over time.</details></td><td><details><summary>展开</summary>该论文提出了一种增强多智能体系统（MAS）安全性和可靠性的新型架构框架，其中包含利用大型语言模型（LLMs）进行语义分析、检索增强验证等技术。Sentinel Agents作为分布式安全层监控通信并识别威胁，Coordinator Agent则实施策略管理和威胁响应，并通过仿真验证了该框架对抗多种攻击（如提示注入、幻觉生成）的有效性。其检索增强验证（retrieval-augmented verification）技术明确体现了RAG的应用。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.14750v1">Enhancing Retrieval Augmentation via Adversarial Collaboration</a></td><td><details><summary>展开</summary>Retrieval-augmented Generation (RAG) is a prevalent approach for
domain-specific LLMs, yet it is often plagued by "Retrieval Hallucinations"--a
phenomenon where fine-tuned models fail to recognize and act upon poor-quality
retrieved documents, thus undermining performance. To address this, we propose
the Adversarial Collaboration RAG (AC-RAG) framework. AC-RAG employs two
heterogeneous agents: a generalist Detector that identifies knowledge gaps, and
a domain-specialized Resolver that provides precise solutions. Guided by a
moderator, these agents engage in an adversarial collaboration, where the
Detector's persistent questioning challenges the Resolver's expertise. This
dynamic process allows for iterative problem dissection and refined knowledge
retrieval. Extensive experiments show that AC-RAG significantly improves
retrieval accuracy and outperforms state-of-the-art RAG methods across various
vertical domains.</details></td><td><details><summary>展开</summary>该论文提出一种名为AC-RAG的新框架，通过引入对抗性协作机制（包含通用检测器和领域专家解析器两个异构代理），有效解决RAG中存在的"检索幻觉"问题，即模型无法识别低质量检索文档的缺陷。实验表明AC-RAG在检索准确性和垂直领域性能上超越现有先进方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.14623v1">Automating Modelica Module Generation Using Large Language Models: A Case Study on Building Control Description Language</a></td><td><details><summary>展开</summary>Dynamic energy systems and controls require advanced modeling frameworks to
design and test supervisory and fault tolerant strategies. Modelica is a widely
used equation based language, but developing control modules is labor intensive
and requires specialized expertise. This paper examines the use of large
language models (LLMs) to automate the generation of Control Description
Language modules in the Building Modelica Library as a case study. We developed
a structured workflow that combines standardized prompt scaffolds, library
aware grounding, automated compilation with OpenModelica, and human in the loop
evaluation. Experiments were carried out on four basic logic tasks (And, Or,
Not, and Switch) and five control modules (chiller enable/disable, bypass valve
control, cooling tower fan speed, plant requests, and relief damper control).
The results showed that GPT 4o failed to produce executable Modelica code in
zero shot mode, while Claude Sonnet 4 achieved up to full success for basic
logic blocks with carefully engineered prompts. For control modules, success
rates reached 83 percent, and failed outputs required medium level human repair
(estimated one to eight hours). Retrieval augmented generation often produced
mismatches in module selection (for example, And retrieved as Or), while a
deterministic hard rule search strategy avoided these errors. Human evaluation
also outperformed AI evaluation, since current LLMs cannot assess simulation
results or validate behavioral correctness. Despite these limitations, the LLM
assisted workflow reduced the average development time from 10 to 20 hours down
to 4 to 6 hours per module, corresponding to 40 to 60 percent time savings.
These results highlight both the potential and current limitations of LLM
assisted Modelica generation, and point to future research in pre simulation
validation, stronger grounding, and closed loop evaluation.</details></td><td><details><summary>展开</summary>这篇论文探讨了使用大语言模型（LLMs）和检索增强生成（RAG）技术自动化生成Modelica控制模块的方法，通过结合标准化提示框架、库感知基础、自动编译和人工评估，显著减少了开发时间，同时指出了RAG在模块选择上的局限性以及未来改进方向。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.14622v1">Adversarial Distilled Retrieval-Augmented Guarding Model for Online Malicious Intent Detection</a></td><td><details><summary>展开</summary>With the deployment of Large Language Models (LLMs) in interactive
applications, online malicious intent detection has become increasingly
critical. However, existing approaches fall short of handling diverse and
complex user queries in real time. To address these challenges, we introduce
ADRAG (Adversarial Distilled Retrieval-Augmented Guard), a two-stage framework
for robust and efficient online malicious intent detection. In the training
stage, a high-capacity teacher model is trained on adversarially perturbed,
retrieval-augmented inputs to learn robust decision boundaries over diverse and
complex user queries. In the inference stage, a distillation scheduler
transfers the teacher's knowledge into a compact student model, with a
continually updated knowledge base collected online. At deployment, the compact
student model leverages top-K similar safety exemplars retrieved from the
online-updated knowledge base to enable both online and real-time malicious
query detection. Evaluations across ten safety benchmarks demonstrate that
ADRAG, with a 149M-parameter model, achieves 98.5% of WildGuard-7B's
performance, surpasses GPT-4 by 3.3% and Llama-Guard-3-8B by 9.5% on
out-of-distribution detection, while simultaneously delivering up to 5.6x lower
latency at 300 queries per second (QPS) in real-time applications.</details></td><td><details><summary>展开</summary>这篇文章介绍了ADRAG（Adversarial Distilled Retrieval-Augmented Guard），一种结合检索增强生成（RAG）和对抗蒸馏的两阶段框架，用于实时在线恶意意图检测。通过训练阶段利用检索增强的对抗扰动输入训练教师模型，并在推理阶段将知识蒸馏到轻量级学生模型中，其在线更新的知识库支持实时检索Top-K相似安全示例，显著提升了恶意查询检测的性能和效率。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.14608v1">Enterprise AI Must Enforce Participant-Aware Access Control</a></td><td><details><summary>展开</summary>Large language models (LLMs) are increasingly deployed in enterprise settings
where they interact with multiple users and are trained or fine-tuned on
sensitive internal data. While fine-tuning enhances performance by
internalizing domain knowledge, it also introduces a critical security risk:
leakage of confidential training data to unauthorized users. These risks are
exacerbated when LLMs are combined with Retrieval-Augmented Generation (RAG)
pipelines that dynamically fetch contextual documents at inference time.
  We demonstrate data exfiltration attacks on AI assistants where adversaries
can exploit current fine-tuning and RAG architectures to leak sensitive
information by leveraging the lack of access control enforcement. We show that
existing defenses, including prompt sanitization, output filtering, system
isolation, and training-level privacy mechanisms, are fundamentally
probabilistic and fail to offer robust protection against such attacks.
  We take the position that only a deterministic and rigorous enforcement of
fine-grained access control during both fine-tuning and RAG-based inference can
reliably prevent the leakage of sensitive data to unauthorized recipients.
  We introduce a framework centered on the principle that any content used in
training, retrieval, or generation by an LLM is explicitly authorized for
\emph{all users involved in the interaction}. Our approach offers a simple yet
powerful paradigm shift for building secure multi-user LLM systems that are
grounded in classical access control but adapted to the unique challenges of
modern AI workflows. Our solution has been deployed in Microsoft Copilot
Tuning, a product offering that enables organizations to fine-tune models using
their own enterprise-specific data.</details></td><td><details><summary>展开</summary>这篇文章探讨了在企业环境中部署大语言模型（LLMs）和检索增强生成（RAG）管道时面临的数据安全风险，提出了一种基于细粒度访问控制的框架，以防止敏感信息泄露给未经授权的用户，并已在Microsoft Copilot Tuning中部署应用。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.14507v1">DeKeyNLU: Enhancing Natural Language to SQL Generation through Task Decomposition and Keyword Extraction</a></td><td><details><summary>展开</summary>Natural Language to SQL (NL2SQL) provides a new model-centric paradigm that
simplifies database access for non-technical users by converting natural
language queries into SQL commands. Recent advancements, particularly those
integrating Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT)
reasoning, have made significant strides in enhancing NL2SQL performance.
However, challenges such as inaccurate task decomposition and keyword
extraction by LLMs remain major bottlenecks, often leading to errors in SQL
generation. While existing datasets aim to mitigate these issues by fine-tuning
models, they struggle with over-fragmentation of tasks and lack of
domain-specific keyword annotations, limiting their effectiveness. To address
these limitations, we present DeKeyNLU, a novel dataset which contains 1,500
meticulously annotated QA pairs aimed at refining task decomposition and
enhancing keyword extraction precision for the RAG pipeline. Fine-tuned with
DeKeyNLU, we propose DeKeySQL, a RAG-based NL2SQL pipeline that employs three
distinct modules for user question understanding, entity retrieval, and
generation to improve SQL generation accuracy. We benchmarked multiple model
configurations within DeKeySQL RAG pipeline. Experimental results demonstrate
that fine-tuning with DeKeyNLU significantly improves SQL generation accuracy
on both BIRD (62.31% to 69.10%) and Spider (84.2% to 88.7%) dev datasets.</details></td><td><details><summary>展开</summary>这篇论文提出DeKeyNLU数据集和DeKeySQL管道，通过改进任务分解和关键词提取增强RAG在自然语言转SQL（NL2SQL）中的性能，实验显示其显著提升了BIRD和Spider数据集上的SQL生成准确率。</details></td></tr></tbody></table>

### 📅 2025-09-17
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.14436v1">When Content is Goliath and Algorithm is David: The Style and Semantic Effects of Generative Search Engine</a></td><td><details><summary>展开</summary>Generative search engines (GEs) leverage large language models (LLMs) to
deliver AI-generated summaries with website citations, establishing novel
traffic acquisition channels while fundamentally altering the search engine
optimization landscape. To investigate the distinctive characteristics of GEs,
we collect data through interactions with Google's generative and conventional
search platforms, compiling a dataset of approximately ten thousand websites
across both channels. Our empirical analysis reveals that GEs exhibit
preferences for citing content characterized by significantly higher
predictability for underlying LLMs and greater semantic similarity among
selected sources. Through controlled experiments utilizing retrieval augmented
generation (RAG) APIs, we demonstrate that these citation preferences emerge
from intrinsic LLM tendencies to favor content aligned with their generative
expression patterns. Motivated by applications of LLMs to optimize website
content, we conduct additional experimentation to explore how LLM-based content
polishing by website proprietors alters AI summaries, finding that such
polishing paradoxically enhances information diversity within AI summaries.
Finally, to assess the user-end impact of LLM-induced information increases, we
design a generative search engine and recruit Prolific participants to conduct
a randomized controlled experiment involving an information-seeking and writing
task. We find that higher-educated users exhibit minimal changes in their final
outputs' information diversity but demonstrate significantly reduced task
completion time when original sites undergo polishing. Conversely,
lower-educated users primarily benefit through enhanced information density in
their task outputs while maintaining similar completion times across
experimental groups.</details></td><td><details><summary>展开</summary>该论文研究生成式搜索引擎（GEs）的特点及其引用偏好，发现GEs倾向于引用与底层LLM生成表达模式一致的内容，并通过RAG API实验验证了这一偏好源自LLM的内在倾向。此外，论文还探讨了网站所有者通过LLM优化内容对AI摘要的影响，并评估了不同教育背景用户在使用GEs时的表现差异。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.14435v1">Causal-Counterfactual RAG: The Integration of Causal-Counterfactual Reasoning into RAG</a></td><td><details><summary>展开</summary>Large language models (LLMs) have transformed natural language processing
(NLP), enabling diverse applications by integrating large-scale pre-trained
knowledge. However, their static knowledge limits dynamic reasoning over
external information, especially in knowledge-intensive domains.
Retrieval-Augmented Generation (RAG) addresses this challenge by combining
retrieval mechanisms with generative modeling to improve contextual
understanding. Traditional RAG systems suffer from disrupted contextual
integrity due to text chunking and over-reliance on semantic similarity for
retrieval, often resulting in shallow and less accurate responses. We propose
Causal-Counterfactual RAG, a novel framework that integrates explicit causal
graphs representing cause-effect relationships into the retrieval process and
incorporates counterfactual reasoning grounded on the causal structure. Unlike
conventional methods, our framework evaluates not only direct causal evidence
but also the counterfactuality of associated causes, combining results from
both to generate more robust, accurate, and interpretable answers. By
leveraging causal pathways and associated hypothetical scenarios,
Causal-Counterfactual RAG preserves contextual coherence, reduces
hallucination, and enhances reasoning fidelity.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为Causal-Counterfactual RAG的新框架，通过将显式因果图整合到检索过程中并引入基于因果结构的反事实推理，解决了传统RAG系统因文本分块和过度依赖语义相似性而导致的上下文不连贯和回答浅显的问题，从而生成更准确、鲁棒且可解释的答案。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.13978v1">LLM Agents for Interactive Workflow Provenance: Reference Architecture and Evaluation Methodology</a></td><td><details><summary>展开</summary>Modern scientific discovery increasingly relies on workflows that process
data across the Edge, Cloud, and High Performance Computing (HPC) continuum.
Comprehensive and in-depth analyses of these data are critical for hypothesis
validation, anomaly detection, reproducibility, and impactful findings.
Although workflow provenance techniques support such analyses, at large scale,
the provenance data become complex and difficult to analyze. Existing systems
depend on custom scripts, structured queries, or static dashboards, limiting
data interaction. In this work, we introduce an evaluation methodology,
reference architecture, and open-source implementation that leverages
interactive Large Language Model (LLM) agents for runtime data analysis. Our
approach uses a lightweight, metadata-driven design that translates natural
language into structured provenance queries. Evaluations across LLaMA, GPT,
Gemini, and Claude, covering diverse query classes and a real-world chemistry
workflow, show that modular design, prompt tuning, and Retrieval-Augmented
Generation (RAG) enable accurate and insightful LLM agent responses beyond
recorded provenance.</details></td><td><details><summary>展开</summary>该论文提出了一种利用交互式大语言模型（LLM）代理进行运行时数据分析的方法，采用轻量级、以元数据驱动的设计将自然语言转换为结构化的溯源查询，并通过对比实验（涵盖多种LLM模型及实际化学工作流）证明，其模块化设计、提示调优及检索增强生成（RAG）技术能显著提升LLM代理响应的准确性和洞察力，超越了传统记录的溯源数据能力。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.13930v1">Linguistic Nepotism: Trading-off Quality for Language Preference in Multilingual RAG</a></td><td><details><summary>展开</summary>Multilingual Retrieval-Augmented Generation (mRAG) systems enable language
models to answer knowledge-intensive queries with citation-supported responses
across languages. While such systems have been proposed, an open questions is
whether the mixture of different document languages impacts generation and
citation in unintended ways. To investigate, we introduce a controlled
methodology using model internals to measure language preference while holding
other factors such as document relevance constant. Across eight languages and
six open-weight models, we find that models preferentially cite English sources
when queries are in English, with this bias amplified for lower-resource
languages and for documents positioned mid-context. Crucially, we find that
models sometimes trade-off document relevance for language preference,
indicating that citation choices are not always driven by informativeness
alone. Our findings shed light on how language models leverage multilingual
context and influence citation behavior.</details></td><td><details><summary>展开</summary>本文研究多语言检索增强生成（mRAG）系统中语言偏好对生成和引用的影响，发现模型倾向于引用英文来源，且可能牺牲文档相关性而选择语言偏好，揭示了语言模型在多语言语境中的引用行为特点。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.13772v1">Who Taught the Lie? Responsibility Attribution for Poisoned Knowledge in Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) integrates external knowledge into large
language models to improve response quality. However, recent work has shown
that RAG systems are highly vulnerable to poisoning attacks, where malicious
texts are inserted into the knowledge database to influence model outputs.
While several defenses have been proposed, they are often circumvented by more
adaptive or sophisticated attacks.
  This paper presents RAGOrigin, a black-box responsibility attribution
framework designed to identify which texts in the knowledge database are
responsible for misleading or incorrect generations. Our method constructs a
focused attribution scope tailored to each misgeneration event and assigns a
responsibility score to each candidate text by evaluating its retrieval
ranking, semantic relevance, and influence on the generated response. The
system then isolates poisoned texts using an unsupervised clustering method. We
evaluate RAGOrigin across seven datasets and fifteen poisoning attacks,
including newly developed adaptive poisoning strategies and multi-attacker
scenarios. Our approach outperforms existing baselines in identifying poisoned
content and remains robust under dynamic and noisy conditions. These results
suggest that RAGOrigin provides a practical and effective solution for tracing
the origins of corrupted knowledge in RAG systems.</details></td><td><details><summary>展开</summary>本文提出RAGOrigin框架，针对RAG系统中知识库中毒攻击导致错误生成的问题，通过黑盒责任溯源方法分析检索排序、语义相关性和生成响应影响，识别和隔离恶意文本，并在多数据集和攻击场景下验证其优于现有基线。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.13702v1">DSCC-HS: A Dynamic Self-Reinforcing Framework for Hallucination Suppression in Large Language Models</a></td><td><details><summary>展开</summary>Large Language Model (LLM) hallucination is a significant barrier to their
reliable deployment. Current methods like Retrieval-Augmented Generation (RAG)
are often reactive. We introduce **Dynamic Self-reinforcing Calibration for
Hallucination Suppression (DSCC-HS)**, a novel, proactive framework that
intervenes during autoregressive decoding. Inspired by dual-process cognitive
theory, DSCC-HS uses a compact proxy model, trained in adversarial roles as a
Factual Alignment Proxy (FAP) and a Hallucination Detection Proxy (HDP). During
inference, these proxies dynamically steer a large target model by injecting a
real-time steering vector, which is the difference between FAP and HDP logits,
at each decoding step. This plug-and-play approach requires no modification to
the target model. Our experiments on TruthfulQA and BioGEN show DSCC-HS
achieves state-of-the-art performance. On TruthfulQA, it reached a 99.2%
Factual Consistency Rate (FCR). On the long-form BioGEN benchmark, it attained
the highest FActScore of 46.50. These results validate DSCC-HS as a principled
and efficient solution for enhancing LLM factuality.</details></td><td><details><summary>展开</summary>该论文提出了一种名为DSCC-HS的新型主动式框架，通过动态自我强化校准来抑制LLM的幻觉问题，采用双代理模型（FAP和HDP）在自回归解码过程中实时修正目标模型的输出。尽管属于RAG相关研究（提到RAG作为现有方法对比），但其核心创新点在于不依赖外部检索的主动干预机制，实验证明在TruthfulQA和BioGEN基准中显著提升了生成内容的真实性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.13683v1">Improving Context Fidelity via Native Retrieval-Augmented Reasoning</a></td><td><details><summary>展开</summary>Large language models (LLMs) often struggle with context fidelity, producing
inconsistent answers when responding to questions based on provided
information. Existing approaches either rely on expensive supervised
fine-tuning to generate evidence post-answer or train models to perform web
searches without necessarily improving utilization of the given context. We
propose CARE, a novel native retrieval-augmented reasoning framework that
teaches LLMs to explicitly integrate in-context evidence within their reasoning
process with the model's own retrieval capabilities. Our method requires
limited labeled evidence data while significantly enhancing both retrieval
accuracy and answer generation performance through strategically retrieved
in-context tokens in the reasoning chain. Extensive experiments on multiple
real-world and counterfactual QA benchmarks demonstrate that our approach
substantially outperforms supervised fine-tuning, traditional
retrieval-augmented generation methods, and external retrieval solutions. This
work represents a fundamental advancement in making LLMs more accurate,
reliable, and efficient for knowledge-intensive tasks.</details></td><td><details><summary>展开</summary>这篇论文提出了CARE框架，通过让大语言模型（LLMs）在推理过程中显式整合上下文证据并结合自身检索能力，改进了传统检索增强生成（RAG）方法，显著提升了检索准确性和答案生成性能。实验表明，该方法在多项QA基准测试中优于监督微调和外部检索方案，增强了LLMs在知识密集型任务中的准确性和可靠性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.13626v1">Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval</a></td><td><details><summary>展开</summary>Access to reliable mental health information is vital for early help-seeking,
yet expanding knowledge bases is resource-intensive and often misaligned with
user needs. This results in poor performance of retrieval systems when
presented concerns are not covered or expressed in informal or contextualized
language. We present an AI-based gap-informed framework for corpus augmentation
that authentically identifies underrepresented topics (gaps) by overlaying
naturalistic user data such as forum posts in order to prioritize expansions
based on coverage and usefulness. In a case study, we compare Directed
(gap-informed augmentations) with Non-Directed augmentation (random additions),
evaluating the relevance and usefulness of retrieved information across four
retrieval-augmented generation (RAG) pipelines. Directed augmentation achieved
near-optimal performance with modest expansions--requiring only a 42% increase
for Query Transformation, 74% for Reranking and Hierarchical, and 318% for
Baseline--to reach ~95% of the performance of an exhaustive reference corpus.
In contrast, Non-Directed augmentation required substantially larger and thus
practically infeasible expansions to achieve comparable performance (232%,
318%, 403%, and 763%, respectively). These results show that strategically
targeted corpus growth can reduce content creation demands while sustaining
high retrieval and provision quality, offering a scalable approach for building
trusted health information repositories and supporting generative AI
applications in high-stakes domains.</details></td><td><details><summary>展开</summary>该论文提出了一种基于AI的框架，通过识别未充分覆盖的主题（缺口）来增强语料库，并评估了其在四种检索增强生成（RAG）管道中的效果，结果显示定向增强能以较小的扩展达到接近最优的检索性能。</details></td></tr></tbody></table>

### 📅 2025-09-16
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.12765v1">InfoGain-RAG: Boosting Retrieval-Augmented Generation via Document Information Gain-based Reranking and Filtering</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) has emerged as a promising approach to
address key limitations of Large Language Models (LLMs), such as hallucination,
outdated knowledge, and lacking reference. However, current RAG frameworks
often struggle with identifying whether retrieved documents meaningfully
contribute to answer generation. This shortcoming makes it difficult to filter
out irrelevant or even misleading content, which notably impacts the final
performance. In this paper, we propose Document Information Gain (DIG), a novel
metric designed to quantify the contribution of retrieved documents to correct
answer generation. DIG measures a document's value by computing the difference
of LLM's generation confidence with and without the document augmented.
Further, we introduce InfoGain-RAG, a framework that leverages DIG scores to
train a specialized reranker, which prioritizes each retrieved document from
exact distinguishing and accurate sorting perspectives. This approach can
effectively filter out irrelevant documents and select the most valuable ones
for better answer generation. Extensive experiments across various models and
benchmarks demonstrate that InfoGain-RAG can significantly outperform existing
approaches, on both single and multiple retrievers paradigm. Specifically on
NaturalQA, it achieves the improvements of 17.9%, 4.5%, 12.5% in exact match
accuracy against naive RAG, self-reflective RAG and modern ranking-based RAG
respectively, and even an average of 15.3% increment on advanced proprietary
model GPT-4o across all datasets. These results demonstrate the feasibility of
InfoGain-RAG as it can offer a reliable solution for RAG in multiple
applications.</details></td><td><details><summary>展开</summary>该论文提出了一种名为“文档信息增益（DIG）”的新指标，用于量化检索到的文档对生成正确答案的贡献，并进一步介绍了基于DIG的InfoGain-RAG框架，该框架通过训练专门的重新排序模型来优先选择最有价值的文档，显著提升了RAG的性能。实验结果表明，该方法在多个基准测试中优于现有方法。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.12743v1">Zero-shot Graph Reasoning via Retrieval Augmented Framework with LLMs</a></td><td><details><summary>展开</summary>We propose a new, training-free method, Graph Reasoning via Retrieval
Augmented Framework (GRRAF), that harnesses retrieval-augmented generation
(RAG) alongside the code-generation capabilities of large language models
(LLMs) to address a wide range of graph reasoning tasks. In GRRAF, the target
graph is stored in a graph database, and the LLM is prompted to generate
executable code queries that retrieve the necessary information. This approach
circumvents the limitations of existing methods that require extensive
finetuning or depend on predefined algorithms, and it incorporates an error
feedback loop with a time-out mechanism to ensure both correctness and
efficiency. Experimental evaluations on the GraphInstruct dataset reveal that
GRRAF achieves 100% accuracy on most graph reasoning tasks, including cycle
detection, bipartite graph checks, shortest path computation, and maximum flow,
while maintaining consistent token costs regardless of graph sizes. Imperfect
but still very high performance is observed on subgraph matching. Notably,
GRRAF scales effectively to large graphs with up to 10,000 nodes.</details></td><td><details><summary>展开</summary>该论文介绍了一种名为GRRAF的新型免训练方法，利用检索增强生成（RAG）技术和大型语言模型（LLMs）的代码生成能力来解决广泛的图推理任务。GRRAF通过将目标图存储在图形数据库中，并提示LLM生成可执行的代码查询来检索必要信息，从而避免了现有方法需要大量微调或依赖预定义算法的限制。实验结果显示，GRRAF在大多数图推理任务上实现了100%的准确率，并能有效扩展到包含多达10,000个节点的大型图中。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.12653v1">Beyond Artificial Misalignment: Detecting and Grounding Semantic-Coordinated Multimodal Manipulations</a></td><td><details><summary>展开</summary>The detection and grounding of manipulated content in multimodal data has
emerged as a critical challenge in media forensics. While existing benchmarks
demonstrate technical progress, they suffer from misalignment artifacts that
poorly reflect real-world manipulation patterns: practical attacks typically
maintain semantic consistency across modalities, whereas current datasets
artificially disrupt cross-modal alignment, creating easily detectable
anomalies. To bridge this gap, we pioneer the detection of
semantically-coordinated manipulations where visual edits are systematically
paired with semantically consistent textual descriptions. Our approach begins
with constructing the first Semantic-Aligned Multimodal Manipulation (SAMM)
dataset, generated through a two-stage pipeline: 1) applying state-of-the-art
image manipulations, followed by 2) generation of contextually-plausible
textual narratives that reinforce the visual deception. Building on this
foundation, we propose a Retrieval-Augmented Manipulation Detection and
Grounding (RamDG) framework. RamDG commences by harnessing external knowledge
repositories to retrieve contextual evidence, which serves as the auxiliary
texts and encoded together with the inputs through our image forgery grounding
and deep manipulation detection modules to trace all manipulations. Extensive
experiments demonstrate our framework significantly outperforms existing
methods, achieving 2.06\% higher detection accuracy on SAMM compared to
state-of-the-art approaches. The dataset and code are publicly available at
https://github.com/shen8424/SAMM-RamDG-CAP.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为RAMDG的检索增强多模态篡改检测与定位框架，通过构建语义对齐的多模态篡改数据集（SAMM）并利用外部知识库检索辅助证据，显著提升了篡改检测的准确率。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.12589v1">Redefining CX with Agentic AI: Minerva CQ Case Study</a></td><td><details><summary>展开</summary>Despite advances in AI for contact centers, customer experience (CX)
continues to suffer from high average handling time (AHT), low first-call
resolution, and poor customer satisfaction (CSAT). A key driver is the
cognitive load on agents, who must navigate fragmented systems, troubleshoot
manually, and frequently place customers on hold. Existing AI-powered
agent-assist tools are often reactive driven by static rules, simple prompting,
or retrieval-augmented generation (RAG) without deeper contextual reasoning. We
introduce Agentic AI goal-driven, autonomous, tool-using systems that
proactively support agents in real time. Unlike conventional approaches,
Agentic AI identifies customer intent, triggers modular workflows, maintains
evolving context, and adapts dynamically to conversation state. This paper
presents a case study of Minerva CQ, a real-time Agent Assist product deployed
in voice-based customer support. Minerva CQ integrates real-time transcription,
intent and sentiment detection, entity recognition, contextual retrieval,
dynamic customer profiling, and partial conversational summaries enabling
proactive workflows and continuous context-building. Deployed in live
production, Minerva CQ acts as an AI co-pilot, delivering measurable
improvements in agent efficiency and customer experience across multiple
deployments.</details></td><td><details><summary>展开</summary>这篇论文介绍了Agentic AI在客服中心的应用，特别是Minerva CQ产品，它结合了实时转录、意图识别和检索增强生成（RAG）等技术，通过动态上下文和工作流提升客服代理效率及客户体验。尽管RAG是现有技术之一，但文章重点强调其超越传统RAG的自主性和实时性能力。</details></td></tr></tbody></table>

### 📅 2025-09-15
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.12382v1">LLM-as-a-Judge: Rapid Evaluation of Legal Document Recommendation for Retrieval-Augmented Generation</a></td><td><details><summary>展开</summary>The evaluation bottleneck in recommendation systems has become particularly
acute with the rise of Generative AI, where traditional metrics fall short of
capturing nuanced quality dimensions that matter in specialized domains like
legal research. Can we trust Large Language Models to serve as reliable judges
of their own kind? This paper investigates LLM-as-a-Judge as a principled
approach to evaluating Retrieval-Augmented Generation systems in legal
contexts, where the stakes of recommendation quality are exceptionally high.
  We tackle two fundamental questions that determine practical viability: which
inter-rater reliability metrics best capture the alignment between LLM and
human assessments, and how do we conduct statistically sound comparisons
between competing systems? Through systematic experimentation, we discover that
traditional agreement metrics like Krippendorff's alpha can be misleading in
the skewed distributions typical of AI system evaluations. Instead, Gwet's AC2
and rank correlation coefficients emerge as more robust indicators for judge
selection, while the Wilcoxon Signed-Rank Test with Benjamini-Hochberg
corrections provides the statistical rigor needed for reliable system
comparisons.
  Our findings suggest a path toward scalable, cost-effective evaluation that
maintains the precision demanded by legal applications, transforming what was
once a human-intensive bottleneck into an automated, yet statistically
principled, evaluation framework.</details></td><td><details><summary>展开</summary>这篇论文探讨了在推荐系统中利用大语言模型（LLM）作为评估工具的可行性，特别是在法律检索与生成（RAG）领域。研究重点关注如何选择可信的指标（如Gwet's AC2和秩相关系数）和统计方法（如Wilcoxon Signed-Rank Test）来对齐LLM与人类评估结果，从而为高风险的RAG系统提供可扩展且精准的自动化评估框架。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.12168v1">RAGs to Riches: RAG-like Few-shot Learning for Large Language Model Role-playing</a></td><td><details><summary>展开</summary>Role-playing Large language models (LLMs) are increasingly deployed in
high-stakes domains such as healthcare, education, and governance, where
failures can directly impact user trust and well-being. A cost effective
paradigm for LLM role-playing is few-shot learning, but existing approaches
often cause models to break character in unexpected and potentially harmful
ways, especially when interacting with hostile users. Inspired by
Retrieval-Augmented Generation (RAG), we reformulate LLM role-playing into a
text retrieval problem and propose a new prompting framework called
RAGs-to-Riches, which leverages curated reference demonstrations to condition
LLM responses. We evaluate our framework with LLM-as-a-judge preference voting
and introduce two novel token-level ROUGE metrics: Intersection over Output
(IOO) to quantity how much an LLM improvises and Intersection over References
(IOR) to measure few-shot demonstrations utilization rate during the evaluation
tasks. When simulating interactions with a hostile user, our prompting strategy
incorporates in its responses during inference an average of 35% more tokens
from the reference demonstrations. As a result, across 453 role-playing
interactions, our models are consistently judged as being more authentic, and
remain in-character more often than zero-shot and in-context Learning (ICL)
methods. Our method presents a scalable strategy for building robust,
human-aligned LLM role-playing frameworks.</details></td><td><details><summary>展开</summary>本文提出了一种名为RAGs-to-Riches的提示框架，将大语言模型（LLM）的角色扮演重新构建为文本检索问题，通过利用精心策划的参考演示来调节LLM的响应。该框架在对抗性用户互动中表现更优，能更有效地利用参考演示，提高角色的真实性和一致性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.12086v1">SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation</a></td><td><details><summary>展开</summary>Approximate Nearest Neighbor Search (ANNS) plays a critical role in
applications such as search engines, recommender systems, and RAG for LLMs.
Vector quantization (VQ), a crucial technique for ANNS, is commonly used to
reduce space overhead and accelerate distance computations. However, despite
significant research advances, state-of-the-art VQ methods still face
challenges in balancing encoding efficiency and quantization accuracy. To
address these limitations, we propose a novel VQ method called SAQ. To improve
accuracy, SAQ employs a new dimension segmentation technique to strategically
partition PCA-projected vectors into segments along their dimensions. By
prioritizing leading dimension segments with larger magnitudes, SAQ allocates
more bits to high-impact segments, optimizing the use of the available space
quota. An efficient dynamic programming algorithm is developed to optimize
dimension segmentation and bit allocation, ensuring minimal quantization error.
To speed up vector encoding, SAQ devises a code adjustment technique to first
quantize each dimension independently and then progressively refine quantized
vectors using a coordinate-descent-like approach to avoid exhaustive
enumeration. Extensive experiments demonstrate SAQ's superiority over classical
methods (e.g., PQ, PCA) and recent state-of-the-art approaches (e.g., LVQ,
Extended RabitQ). SAQ achieves up to 80% reduction in quantization error and
accelerates encoding speed by over 80x compared to Extended RabitQ.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为SAQ的新型向量量化方法，旨在改进近似最近邻搜索（ANNS）中的编码效率和量化精度平衡问题，通过维度分割和动态编程优化技术显著降低量化误差并加速编码速度，直接关联并优化了RAG技术中检索环节的核心性能。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.12042v1">FinGEAR: Financial Mapping-Guided Enhanced Answer Retrieval</a></td><td><details><summary>展开</summary>Financial disclosures such as 10-K filings present challenging retrieval
problems due to their length, regulatory section hierarchy, and domain-specific
language, which standard retrieval-augmented generation (RAG) models underuse.
We introduce FinGEAR (Financial Mapping-Guided Enhanced Answer Retrieval), a
retrieval framework tailored to financial documents. FinGEAR combines a finance
lexicon for Item-level guidance (FLAM), dual hierarchical indices for
within-Item search (Summary Tree and Question Tree), and a two-stage
cross-encoder reranker. This design aligns retrieval with disclosure structure
and terminology, enabling fine-grained, query-aware context selection.
Evaluated on full 10-Ks with queries aligned to the FinQA dataset, FinGEAR
delivers consistent gains in precision, recall, F1, and relevancy, improving F1
by up to 56.7% over flat RAG, 12.5% over graph-based RAGs, and 217.6% over
prior tree-based systems, while also increasing downstream answer accuracy with
a fixed reader. By jointly modeling section hierarchy and domain lexicon
signals, FinGEAR improves retrieval fidelity and provides a practical
foundation for high-stakes financial analysis.</details></td><td><details><summary>展开</summary>这篇论文介绍了FinGEAR，一个针对金融文档（如10-K文件）优化的检索框架，通过结合金融词汇表（FLAM）、双重层次索引和两阶段交叉编码器重排器，改进了传统RAG模型在金融领域的检索效果，显著提升了精确率、召回率和下游答案准确性。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.11947v1">A GPU-Accelerated RAG-Based Telegram Assistant for Supporting Parallel Processing Students</a></td><td><details><summary>展开</summary>This project addresses a critical pedagogical need: offering students
continuous, on-demand academic assistance beyond conventional reception hours.
I present a domain-specific Retrieval-Augmented Generation (RAG) system powered
by a quantized Mistral-7B Instruct model and deployed as a Telegram bot. The
assistant enhances learning by delivering real-time, personalized responses
aligned with the "Introduction to Parallel Processing" course materials. GPU
acceleration significantly improves inference latency, enabling practical
deployment on consumer hardware. This approach demonstrates how consumer GPUs
can enable affordable, private, and effective AI tutoring for HPC education.</details></td><td><details><summary>展开</summary>本文介绍了一个面向教育领域的RAG系统，基于量化版Mistral-7B Instruct模型构建，通过Telegram机器人提供并行处理课程的实时个性化学习支持，利用GPU加速实现消费级硬件部署，展示了低成本高效的AI辅导方案。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.11937v1">MMORE: Massive Multimodal Open RAG & Extraction</a></td><td><details><summary>展开</summary>We introduce MMORE, an open-source pipeline for Massive Multimodal Open
RetrievalAugmented Generation and Extraction, designed to ingest, transform,
and retrieve knowledge from heterogeneous document formats at scale. MMORE
supports more than fifteen file types, including text, tables, images, emails,
audio, and video, and processes them into a unified format to enable downstream
applications for LLMs. The architecture offers modular, distributed processing,
enabling scalable parallelization across CPUs and GPUs. On processing
benchmarks, MMORE demonstrates a 3.8-fold speedup over single-node baselines
and 40% higher accuracy than Docling on scanned PDFs. The pipeline integrates
hybrid dense-sparse retrieval and supports both interactive APIs and batch RAG
endpoints. Evaluated on PubMedQA, MMORE-augmented medical LLMs improve
biomedical QA accuracy with increasing retrieval depth. MMORE provides a
robust, extensible foundation for deploying task-agnostic RAG systems on
diverse, real-world multimodal data. The codebase is available at
https://github.com/swiss-ai/mmore.</details></td><td><details><summary>展开</summary>MMORE是一个开源的多模态检索增强生成（RAG）系统，支持处理多种文档格式（如文本、表格、图像等），并将其统一处理以供大语言模型使用。该系统通过分布式处理提高了效率和准确性，集成了混合检索方法，并在医疗QA任务中展现了性能提升。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.11687v1">A Dynamic Knowledge Update-Driven Model with Large Language Models for Fake News Detection</a></td><td><details><summary>展开</summary>As the Internet and social media evolve rapidly, distinguishing credible news
from a vast amount of complex information poses a significant challenge. Due to
the suddenness and instability of news events, the authenticity labels of news
can potentially shift as events develop, making it crucial for fake news
detection to obtain the latest event updates. Existing methods employ
retrieval-augmented generation to fill knowledge gaps, but they suffer from
issues such as insufficient credibility of retrieved content and interference
from noisy information. We propose a dynamic knowledge update-driven model for
fake news detection (DYNAMO), which leverages knowledge graphs to achieve
continuous updating of new knowledge and integrates with large language models
to fulfill dual functions: news authenticity detection and verification of new
knowledge correctness, solving the two key problems of ensuring the
authenticity of new knowledge and deeply mining news semantics. Specifically,
we first construct a news-domain-specific knowledge graph. Then, we use Monte
Carlo Tree Search to decompose complex news and verify them step by step.
Finally, we extract and update new knowledge from verified real news texts and
reasoning paths. Experimental results demonstrate that DYNAMO achieves the best
performance on two real-world datasets.</details></td><td><details><summary>展开</summary>这篇论文提出了一种名为DYNAMO的假新闻检测模型，通过结合知识图谱的动态更新与大语言模型，解决了现有检索增强生成方法中检索内容可信度不足和噪声干扰的问题。模型利用新闻领域特定的知识图谱，通过蒙特卡洛树搜索逐步分解和验证复杂新闻，同时从已验证的真实新闻中提取和更新知识，实现了新闻真实性检测与新知识正确性验证的双重功能。实验结果表明DYNAMO在两个真实数据集上表现最佳。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.11645v1">Adapting and Evaluating Multimodal Large Language Models for Adolescent Idiopathic Scoliosis Self-Management: A Divide and Conquer Framework</a></td><td><details><summary>展开</summary>This study presents the first comprehensive evaluation of Multimodal Large
Language Models (MLLMs) for Adolescent Idiopathic Scoliosis (AIS)
self-management. We constructed a database of approximately 3,000
anteroposterior X-rays with diagnostic texts and evaluated five MLLMs through a
`Divide and Conquer' framework consisting of a visual question-answering task,
a domain knowledge assessment task, and a patient education counseling
assessment task. Our investigation revealed limitations of MLLMs' ability in
interpreting complex spinal radiographs and comprehending AIS care knowledge.
To address these, we pioneered enhancing MLLMs with spinal keypoint prompting
and compiled an AIS knowledge base for retrieval augmented generation (RAG),
respectively. Results showed varying effectiveness of visual prompting across
different architectures, while RAG substantially improved models' performances
on the knowledge assessment task. Our findings indicate current MLLMs are far
from capable in realizing personalized assistant in AIS care. The greatest
challenge lies in their abilities to obtain accurate detections of spinal
deformity locations (best accuracy: 0.55) and directions (best accuracy: 0.13).</details></td><td><details><summary>展开</summary>该研究评估了多模态大语言模型(MLLMs)在青少年特发性脊柱侧凸(AIS)自我管理中的应用，发现模型在解读复杂脊柱X光片和理解AIS护理知识方面存在局限，并通过引入脊柱关键点提示和构建AIS知识库结合检索增强生成(RAG)技术来提升模型性能，结果显示RAG显著改善了模型的知识评估任务表现。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.14267v1">Graph-Enhanced Retrieval-Augmented Question Answering for E-Commerce Customer Support</a></td><td><details><summary>展开</summary>E-Commerce customer support requires quick and accurate answers grounded in
product data and past support cases. This paper develops a novel
retrieval-augmented generation (RAG) framework that uses knowledge graphs (KGs)
to improve the relevance of the answer and the factual grounding. We examine
recent advances in knowledge-augmented RAG and chatbots based on large language
models (LLM) in customer support, including Microsoft's GraphRAG and hybrid
retrieval architectures. We then propose a new answer synthesis algorithm that
combines structured subgraphs from a domain-specific KG with text documents
retrieved from support archives, producing more coherent and grounded
responses. We detail the architecture and knowledge flow of our system, provide
comprehensive experimental evaluation, and justify its design in real-time
support settings. Our implementation demonstrates 23\% improvement in factual
accuracy and 89\% user satisfaction in e-Commerce QA scenarios.</details></td><td><details><summary>展开</summary>该论文提出了一种新颖的基于知识图谱（KG）的RAG框架，旨在提升电子商务客服回答的相关性和事实依据，通过结合结构化子图和文本检索生成更连贯的响应，实验表明其实现23%的事实准确性提升和89%的用户满意度。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.11552v2">HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking</a></td><td><details><summary>展开</summary>Retrieval-Augmented Generation (RAG) enhances the response capabilities of
language models by integrating external knowledge sources. However, document
chunking as an important part of RAG system often lacks effective evaluation
tools. This paper first analyzes why existing RAG evaluation benchmarks are
inadequate for assessing document chunking quality, specifically due to
evidence sparsity. Based on this conclusion, we propose HiCBench, which
includes manually annotated multi-level document chunking points, synthesized
evidence-dense quetion answer(QA) pairs, and their corresponding evidence
sources. Additionally, we introduce the HiChunk framework, a multi-level
document structuring framework based on fine-tuned LLMs, combined with the
Auto-Merge retrieval algorithm to improve retrieval quality. Experiments
demonstrate that HiCBench effectively evaluates the impact of different
chunking methods across the entire RAG pipeline. Moreover, HiChunk achieves
better chunking quality within reasonable time consumption, thereby enhancing
the overall performance of RAG systems.</details></td><td><details><summary>展开</summary>这篇论文聚焦于RAG系统中文档分块（chunking）评估的不足，提出带有手动标注多级分块点的评估基准HiCBench和证据密集型QA数据集，同时设计了基于微调LLMs的多级文档结构化框架HiChunk及Auto-Merge检索算法，实验证明其能有效提升分块质量和RAG整体性能。</details></td></tr></tbody></table>

### 📅 2025-09-14
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody><tr><td><a href="http://arxiv.org/abs/2509.11376v1">Intelligent Reservoir Decision Support: An Integrated Framework Combining Large Language Models, Advanced Prompt Engineering, and Multimodal Data Fusion for Real-Time Petroleum Operations</a></td><td><details><summary>展开</summary>The petroleum industry faces unprecedented challenges in reservoir
management, requiring rapid integration of complex multimodal datasets for
real-time decision support. This study presents a novel integrated framework
combining state-of-the-art large language models (GPT-4o, Claude 4 Sonnet,
Gemini 2.5 Pro) with advanced prompt engineering techniques and multimodal data
fusion for comprehensive reservoir analysis. The framework implements
domain-specific retrieval-augmented generation (RAG) with over 50,000 petroleum
engineering documents, chain-of-thought reasoning, and few-shot learning for
rapid field adaptation. Multimodal integration processes seismic
interpretations, well logs, and production data through specialized AI models
with vision transformers. Field validation across 15 diverse reservoir
environments demonstrates exceptional performance: 94.2% reservoir
characterization accuracy, 87.6% production forecasting precision, and 91.4%
well placement optimization success rate. The system achieves sub-second
response times while maintaining 96.2% safety reliability with no high-risk
incidents during evaluation. Economic analysis reveals 62-78% cost reductions
(mean 72%) relative to traditional methods with 8-month payback period.
Few-shot learning reduces field adaptation time by 72%, while automated prompt
optimization achieves 89% improvement in reasoning quality. The framework
processed real-time data streams with 96.2% anomaly detection accuracy and
reduced environmental incidents by 45%. We provide detailed experimental
protocols, baseline comparisons, ablation studies, and statistical significance
testing to ensure reproducibility. This research demonstrates practical
integration of cutting-edge AI technologies with petroleum domain expertise for
enhanced operational efficiency, safety, and economic performance.</details></td><td><details><summary>展开</summary>该论文提出了一种结合大型语言模型、多模态数据融合和领域特定检索增强生成（RAG）技术的集成框架，用于石油行业的储层管理。通过整合超过50,000份石油工程文档的RAG系统、多模态数据处理（如地震解释、测井数据和生产数据）以及链式推理和少样本学习，显著提高了储层表征、产量预测和井位优化的准确性和效率，同时降低了成本和安全风险。实证结果表明该系统在多个性能指标上表现卓越。</details></td></tr><tr><td><a href="http://arxiv.org/abs/2509.14265v1">Evolution of Kernels: Automated RISC-V Kernel Optimization with Large Language Models</a></td><td><details><summary>展开</summary>Automated kernel design is critical for overcoming software ecosystem
barriers in emerging hardware platforms like RISC-V. While large language
models (LLMs) have shown promise for automated kernel optimization,
demonstrating success in CUDA domains with comprehensive technical documents
and mature codebases, their effectiveness remains unproven for reference-scarce
domains like RISC-V. We present Evolution of Kernels (EoK), a novel LLM-based
evolutionary program search framework that automates kernel design for domains
with limited reference material. EoK mitigates reference scarcity by mining and
formalizing reusable optimization ideas (general design principles + actionable
thoughts) from established kernel libraries' development histories; it then
guides parallel LLM explorations using these ideas, enriched via
Retrieval-Augmented Generation (RAG) with RISC-V-specific context, prioritizing
historically effective techniques. Empirically, EoK achieves a median 1.27x
speedup, surpassing human experts on all 80 evaluated kernel design tasks and
improving upon prior LLM-based automated kernel design methods by 20%. These
results underscore the viability of incorporating human experience into
emerging domains and highlight the immense potential of LLM-based automated
kernel optimization.</details></td><td><details><summary>展开</summary>该论文提出了一个名为EoK（Evolution of Kernels）的基于大语言模型的进化程序搜索框架，用于在RISC-V等参考资源稀缺的领域自动化内核设计。EoK通过从已有内核库的开发历史中挖掘和形式化可重用的优化思想，并利用检索增强生成（RAG）结合RISC-V特定上下文来指导并行的大语言模型探索，从而在80项内核设计任务中实现了中位数1.27倍的加速，超越人类专家和先前基于大语言模型的方法。</details></td></tr></tbody></table>
