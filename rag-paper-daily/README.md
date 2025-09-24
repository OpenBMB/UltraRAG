# 📚 RAG Paper Daily

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
