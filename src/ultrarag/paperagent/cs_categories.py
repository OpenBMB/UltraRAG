"""
arXiv CS (Computer Science) 分类标签系统

提供完整的 CS 子分类定义，用于用户画像和论文检索。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class CSCategory:
    """CS 子分类数据模型"""
    code: str  # 分类代码，如 "cs.CL"
    name: str  # 英文名称
    name_zh: str  # 中文名称
    description: str  # 分类描述
    keywords: List[str]  # 关键词列表，用于匹配

# CS 子分类完整定义
CS_SUBCATEGORIES: Dict[str, CSCategory] = {
    "cs.AI": CSCategory(
        code="cs.AI",
        name="Artificial Intelligence",
        name_zh="人工智能",
        description="Covers all areas of AI except Vision, Robotics, Machine Learning, Multiagent Systems, and Computation and Language",
        keywords=["artificial intelligence", "AI", "knowledge representation", "reasoning", "planning"]
    ),
    "cs.CL": CSCategory(
        code="cs.CL",
        name="Computation and Language",
        name_zh="计算语言学/NLP",
        description="Natural language processing, computational linguistics, speech recognition",
        keywords=["NLP", "natural language", "language model", "LLM", "text generation", "machine translation", "sentiment analysis"]
    ),
    "cs.CV": CSCategory(
        code="cs.CV",
        name="Computer Vision and Pattern Recognition",
        name_zh="计算机视觉",
        description="Image processing, computer vision, pattern recognition",
        keywords=["computer vision", "image", "object detection", "segmentation", "CNN", "visual"]
    ),
    "cs.LG": CSCategory(
        code="cs.LG",
        name="Machine Learning",
        name_zh="机器学习",
        description="Machine learning algorithms and applications",
        keywords=["machine learning", "deep learning", "neural network", "training", "optimization", "transformer"]
    ),
    "cs.IR": CSCategory(
        code="cs.IR",
        name="Information Retrieval",
        name_zh="信息检索",
        description="Information retrieval, search engines, recommender systems",
        keywords=["information retrieval", "search", "retrieval", "RAG", "recommendation", "ranking"]
    ),
    "cs.RO": CSCategory(
        code="cs.RO",
        name="Robotics",
        name_zh="机器人学",
        description="Robotics, control, and automation",
        keywords=["robotics", "robot", "control", "manipulation", "navigation", "autonomous"]
    ),
    "cs.SE": CSCategory(
        code="cs.SE",
        name="Software Engineering",
        name_zh="软件工程",
        description="Software development, testing, maintenance",
        keywords=["software engineering", "code generation", "testing", "debugging", "programming"]
    ),
    "cs.DB": CSCategory(
        code="cs.DB",
        name="Databases",
        name_zh="数据库",
        description="Database management, query processing, data storage",
        keywords=["database", "SQL", "query", "data management", "storage"]
    ),
    "cs.DC": CSCategory(
        code="cs.DC",
        name="Distributed, Parallel, and Cluster Computing",
        name_zh="分布式计算",
        description="Distributed systems, parallel computing, cloud computing",
        keywords=["distributed", "parallel", "cloud", "cluster", "scalability"]
    ),
    "cs.CR": CSCategory(
        code="cs.CR",
        name="Cryptography and Security",
        name_zh="密码学与安全",
        description="Cryptography, security, privacy",
        keywords=["security", "cryptography", "privacy", "encryption", "authentication"]
    ),
    "cs.NE": CSCategory(
        code="cs.NE",
        name="Neural and Evolutionary Computing",
        name_zh="神经与进化计算",
        description="Neural networks, genetic algorithms, evolutionary computation",
        keywords=["neural network", "evolutionary", "genetic algorithm", "neuroevolution"]
    ),
    "cs.HC": CSCategory(
        code="cs.HC",
        name="Human-Computer Interaction",
        name_zh="人机交互",
        description="HCI, user interfaces, user experience",
        keywords=["HCI", "human-computer", "interface", "user experience", "UX", "interaction"]
    ),
    "cs.MA": CSCategory(
        code="cs.MA",
        name="Multiagent Systems",
        name_zh="多智能体系统",
        description="Multi-agent systems, agent-based modeling",
        keywords=["multi-agent", "agent", "cooperation", "coordination", "swarm"]
    ),
    "cs.GT": CSCategory(
        code="cs.GT",
        name="Computer Science and Game Theory",
        name_zh="计算机博弈论",
        description="Game theory, mechanism design, computational economics",
        keywords=["game theory", "mechanism design", "auction", "economics", "strategic"]
    ),
    "cs.SI": CSCategory(
        code="cs.SI",
        name="Social and Information Networks",
        name_zh="社交与信息网络",
        description="Social networks, information diffusion, network analysis",
        keywords=["social network", "graph", "network analysis", "community", "influence"]
    ),
    "cs.CY": CSCategory(
        code="cs.CY",
        name="Computers and Society",
        name_zh="计算机与社会",
        description="Social implications of technology, ethics, policy",
        keywords=["ethics", "society", "policy", "bias", "fairness", "AI ethics"]
    ),
    "cs.SD": CSCategory(
        code="cs.SD",
        name="Sound",
        name_zh="声音处理",
        description="Audio processing, speech synthesis, music",
        keywords=["audio", "sound", "speech", "music", "TTS", "voice"]
    ),
    "cs.MM": CSCategory(
        code="cs.MM",
        name="Multimedia",
        name_zh="多媒体",
        description="Multimedia systems, video processing",
        keywords=["multimedia", "video", "streaming", "compression"]
    ),
    "cs.GR": CSCategory(
        code="cs.GR",
        name="Graphics",
        name_zh="图形学",
        description="Computer graphics, rendering, 3D modeling",
        keywords=["graphics", "rendering", "3D", "visualization", "animation"]
    ),
    "cs.AR": CSCategory(
        code="cs.AR",
        name="Hardware Architecture",
        name_zh="硬件架构",
        description="Computer architecture, hardware design",
        keywords=["architecture", "hardware", "processor", "GPU", "accelerator"]
    ),
    "cs.OS": CSCategory(
        code="cs.OS",
        name="Operating Systems",
        name_zh="操作系统",
        description="Operating systems, system software",
        keywords=["operating system", "kernel", "scheduler", "memory management"]
    ),
    "cs.NI": CSCategory(
        code="cs.NI",
        name="Networking and Internet Architecture",
        name_zh="网络与互联网",
        description="Computer networks, protocols, internet",
        keywords=["network", "internet", "protocol", "routing", "wireless"]
    ),
    "cs.PL": CSCategory(
        code="cs.PL",
        name="Programming Languages",
        name_zh="编程语言",
        description="Programming language design and implementation",
        keywords=["programming language", "compiler", "type system", "semantics"]
    ),
    "cs.LO": CSCategory(
        code="cs.LO",
        name="Logic in Computer Science",
        name_zh="计算机逻辑",
        description="Formal methods, verification, logic",
        keywords=["logic", "formal methods", "verification", "theorem proving"]
    ),
    "cs.DS": CSCategory(
        code="cs.DS",
        name="Data Structures and Algorithms",
        name_zh="数据结构与算法",
        description="Algorithms, data structures, complexity",
        keywords=["algorithm", "data structure", "complexity", "optimization"]
    ),
    "cs.CC": CSCategory(
        code="cs.CC",
        name="Computational Complexity",
        name_zh="计算复杂性",
        description="Computational complexity theory",
        keywords=["complexity", "P vs NP", "hardness", "reduction"]
    ),
    "cs.FL": CSCategory(
        code="cs.FL",
        name="Formal Languages and Automata Theory",
        name_zh="形式语言与自动机",
        description="Formal languages, automata, grammars",
        keywords=["automata", "formal language", "grammar", "finite state"]
    ),
    "cs.CG": CSCategory(
        code="cs.CG",
        name="Computational Geometry",
        name_zh="计算几何",
        description="Geometric algorithms, spatial data structures",
        keywords=["computational geometry", "geometric", "spatial", "mesh"]
    ),
    "cs.SC": CSCategory(
        code="cs.SC",
        name="Symbolic Computation",
        name_zh="符号计算",
        description="Computer algebra, symbolic mathematics",
        keywords=["symbolic", "algebra", "mathematical", "symbolic AI"]
    ),
    "cs.NA": CSCategory(
        code="cs.NA",
        name="Numerical Analysis",
        name_zh="数值分析",
        description="Numerical methods, scientific computing",
        keywords=["numerical", "scientific computing", "simulation", "numerical methods"]
    ),
    "cs.CE": CSCategory(
        code="cs.CE",
        name="Computational Engineering, Finance, and Science",
        name_zh="计算工程/金融/科学",
        description="Computational methods in engineering, finance, science",
        keywords=["computational", "simulation", "modeling", "engineering"]
    ),
    "cs.IT": CSCategory(
        code="cs.IT",
        name="Information Theory",
        name_zh="信息论",
        description="Information theory, coding theory",
        keywords=["information theory", "coding", "entropy", "channel"]
    ),
    "cs.SY": CSCategory(
        code="cs.SY",
        name="Systems and Control",
        name_zh="系统与控制",
        description="Control systems, dynamic systems",
        keywords=["control", "system", "dynamic", "feedback", "stability"]
    ),
}

# 热门研究方向分组
POPULAR_RESEARCH_AREAS = {
    "大语言模型与NLP": ["cs.CL", "cs.AI", "cs.LG"],
    "计算机视觉": ["cs.CV", "cs.GR", "cs.MM"],
    "机器学习": ["cs.LG", "cs.NE", "cs.AI"],
    "信息检索与推荐": ["cs.IR", "cs.DB", "cs.SI"],
    "机器人与自动化": ["cs.RO", "cs.SY", "cs.AI"],
    "安全与隐私": ["cs.CR", "cs.CY"],
    "软件工程": ["cs.SE", "cs.PL", "cs.LO"],
    "系统与网络": ["cs.DC", "cs.NI", "cs.OS", "cs.AR"],
    "多智能体系统": ["cs.MA", "cs.AI", "cs.GT"],
}


def get_category(code: str) -> Optional[CSCategory]:
    """获取分类信息"""
    return CS_SUBCATEGORIES.get(code)


def get_all_categories() -> List[CSCategory]:
    """获取所有分类"""
    return list(CS_SUBCATEGORIES.values())


def get_categories_by_codes(codes: List[str]) -> List[CSCategory]:
    """根据代码列表获取分类"""
    return [CS_SUBCATEGORIES[c] for c in codes if c in CS_SUBCATEGORIES]


def search_categories(keyword: str) -> List[CSCategory]:
    """根据关键词搜索相关分类"""
    keyword = keyword.lower()
    results = []
    for cat in CS_SUBCATEGORIES.values():
        # 搜索名称、描述、关键词
        if (keyword in cat.name.lower() or 
            keyword in cat.name_zh or 
            keyword in cat.description.lower() or
            any(keyword in kw.lower() for kw in cat.keywords)):
            results.append(cat)
    return results


def get_category_display_list() -> List[Dict]:
    """获取分类列表用于前端显示"""
    return [
        {
            "code": cat.code,
            "name": cat.name,
            "name_zh": cat.name_zh,
            "description": cat.description,
        }
        for cat in CS_SUBCATEGORIES.values()
    ]

