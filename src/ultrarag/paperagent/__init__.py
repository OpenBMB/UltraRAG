"""
PaperAgent - 学术论文研究助手

提供以下功能：
- arXiv 论文搜索、获取和下载
- CS 分类标签系统
- 用户画像管理
- 知识库管理
- 研究报告生成
"""

from .arxiv_crawler import (
    ArxivCrawler,
    Paper,
    SearchResult,
    search_arxiv,
    get_arxiv_paper,
    download_arxiv_pdf,
)

from .cs_categories import (
    CSCategory,
    CS_SUBCATEGORIES,
    POPULAR_RESEARCH_AREAS,
    get_category,
    get_all_categories,
    get_categories_by_codes,
    search_categories,
    get_category_display_list,
)

from .user_profile import (
    UserProfile,
    ResearchInterest,
    PaperInteraction,
    UserProfileManager,
    get_profile_manager,
)

from .knowledge_manager import (
    PaperMetadata,
    KnowledgeManager,
    get_knowledge_manager,
)

from .report_manager import (
    ResearchReport,
    ReportManager,
    get_report_manager,
)

__all__ = [
    # 爬虫
    "ArxivCrawler",
    "Paper",
    "SearchResult",
    "search_arxiv",
    "get_arxiv_paper",
    "download_arxiv_pdf",
    # 分类
    "CSCategory",
    "CS_SUBCATEGORIES",
    "POPULAR_RESEARCH_AREAS",
    "get_category",
    "get_all_categories",
    "get_categories_by_codes",
    "search_categories",
    "get_category_display_list",
    # 用户画像
    "UserProfile",
    "ResearchInterest",
    "PaperInteraction",
    "UserProfileManager",
    "get_profile_manager",
    # 知识管理
    "PaperMetadata",
    "KnowledgeManager",
    "get_knowledge_manager",
    # 报告管理
    "ResearchReport",
    "ReportManager",
    "get_report_manager",
]

__version__ = "0.1.0"
