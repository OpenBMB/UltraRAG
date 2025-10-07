# 📚 RAG Paper Daily

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
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody></tbody></table>

### 📅 2025-10-04
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody></tbody></table>

### 📅 2025-10-03
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody></tbody></table>

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
<table style='width:100%;'><colgroup><col><col><col></colgroup><thead><tr><th>title</th><th>abstract</th><th>summary</th></tr></thead><tbody></tbody></table>

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
