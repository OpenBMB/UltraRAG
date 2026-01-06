"""
PaperAgent MCP Server

提供论文搜索、用户画像管理、报告生成等工具。
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加 src 到路径
SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrarag.server import UltraRAG_MCP_Server

app = UltraRAG_MCP_Server("paperagent")


# 默认用户 ID（简化版，单用户模式）
DEFAULT_USER_ID = "default"

# ==================== 初始化工具 ====================

@app.tool()
def paperagent_init() -> Dict[str, bool]:
    """初始化 PaperAgent 会话（单用户模式）"""
    from ultrarag.paperagent.user_profile import get_profile_manager
    from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
    from ultrarag.paperagent.report_manager import get_report_manager
    
    # 确保管理器已初始化
    get_profile_manager()
    get_knowledge_manager()
    get_report_manager()
    
    app.logger.info(f"[paperagent] Initialized")
    return {"initialized": True}


# ==================== 分类与兴趣工具 ====================

@app.tool(output="None->cs_categories")
def get_cs_categories() -> Dict[str, List[Dict]]:
    """获取所有 CS 分类"""
    from ultrarag.paperagent.cs_categories import get_category_display_list
    
    categories = get_category_display_list()
    return {"cs_categories": categories}


@app.tool(output="category->onboarding_papers")
def get_onboarding_papers(
    category: str,
    count: int = 10
) -> Dict[str, List[Dict]]:
    """获取用于 onboarding 的论文"""
    from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
    
    km = get_knowledge_manager()
    papers = km.get_onboarding_papers(category, count)
    
    return {
        "onboarding_papers": [p.to_dict() for p in papers]
    }


@app.tool(output="user_id,categories,paper_ids->profile")
def set_user_interests(
    user_id: str,
    categories: List[str],
    paper_ids: List[str] = None
) -> Dict[str, Dict]:
    """设置用户初始研究兴趣"""
    from ultrarag.paperagent.user_profile import get_profile_manager
    
    pm = get_profile_manager()
    profile = pm.set_initial_interests(user_id, categories, paper_ids)
    
    app.logger.info(f"[paperagent] Set interests for {user_id}: {categories}")
    
    return {"profile": profile.to_dict()}


@app.tool(output="user_id->profile")
def get_user_profile(user_id: str) -> Dict[str, Optional[Dict]]:
    """获取用户画像"""
    from ultrarag.paperagent.user_profile import get_profile_manager
    
    pm = get_profile_manager()
    profile = pm.get_profile(user_id)
    
    return {"profile": profile.to_dict() if profile else None}


# ==================== 论文搜索工具 ====================

@app.tool(output="query,category,max_results->papers")
def search_papers(
    query: str,
    category: str = "",
    max_results: int = 20
) -> Dict[str, List[Dict]]:
    """搜索论文"""
    from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
    
    km = get_knowledge_manager()
    papers = km.search_papers(
        query=query,
        category=category if category else None,
        max_results=max_results
    )
    
    return {"papers": [p.to_dict() for p in papers]}


@app.tool(output="user_id,max_results->recommended_papers")
def get_recommended_papers(
    user_id: str,
    max_results: int = 20
) -> Dict[str, List[Dict]]:
    """获取推荐论文"""
    from ultrarag.paperagent.user_profile import get_profile_manager
    from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
    
    pm = get_profile_manager()
    km = get_knowledge_manager()
    
    profile = pm.get_profile(user_id)
    if not profile:
        return {"recommended_papers": []}
    
    # 获取用户兴趣
    interests = profile.get_top_interests(5)
    keywords = list(profile.keyword_preferences.keys())[:10]
    
    # 排除已交互的论文
    exclude_ids = [pi.arxiv_id for pi in profile.paper_interactions]
    
    papers = km.get_recommended_papers(
        interests=interests,
        keywords=keywords,
        exclude_ids=exclude_ids,
        max_results=max_results
    )
    
    return {"recommended_papers": [p.to_dict() for p in papers]}


@app.tool(output="categories,max_per_category->trending_papers")
def get_trending_papers(
    categories: List[str],
    max_per_category: int = 10
) -> Dict[str, Dict[str, List[Dict]]]:
    """获取热门论文"""
    from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
    
    km = get_knowledge_manager()
    trending = km.get_trending_papers(categories, max_per_category)
    
    result = {}
    for cat, papers in trending.items():
        result[cat] = [p.to_dict() for p in papers]
    
    return {"trending_papers": result}


@app.tool(output="arxiv_ids->papers")
def get_papers_by_ids(arxiv_ids: List[str]) -> Dict[str, List[Dict]]:
    """根据 ID 获取论文"""
    from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
    
    km = get_knowledge_manager()
    papers = km.get_papers(arxiv_ids)
    
    return {"papers": [p.to_dict() for p in papers]}


# ==================== 用户交互工具 ====================

@app.tool(output="user_id,arxiv_id,interaction_type->success")
def record_paper_interaction(
    user_id: str,
    arxiv_id: str,
    title: str = "",
    categories: List[str] = None,
    interaction_type: str = "read"
) -> Dict[str, bool]:
    """记录论文交互"""
    from ultrarag.paperagent.user_profile import get_profile_manager
    
    pm = get_profile_manager()
    pm.record_paper_interaction(
        user_id=user_id,
        arxiv_id=arxiv_id,
        title=title,
        categories=categories or [],
        interaction_type=interaction_type
    )
    
    return {"success": True}


@app.tool(output="user_id,summary,keywords,categories->success")
def update_profile_from_conversation(
    user_id: str,
    summary: str,
    keywords: List[str] = None,
    categories: List[str] = None
) -> Dict[str, bool]:
    """从对话更新用户画像"""
    from ultrarag.paperagent.user_profile import get_profile_manager
    
    pm = get_profile_manager()
    pm.update_from_conversation(
        user_id=user_id,
        conversation_summary=summary,
        extracted_keywords=keywords,
        mentioned_categories=categories
    )
    
    app.logger.info(f"[paperagent] Updated profile from conversation for {user_id}")
    
    return {"success": True}


# ==================== 报告工具 ====================

@app.tool(output="content,topics->report")
def create_report(
    content: Any = "",
    topics: Any = None,
    title: str = "",
    categories: List[str] = None,
    keywords: List[str] = None
) -> Dict[str, Dict]:
    """创建研究报告（单用户模式）
    
    content 和 topics 可以是字符串或列表（pipeline 输出通常是列表）
    title 如果不提供，会从 topics 自动生成
    自动从 ArxivCitationRegistry 获取引用的论文信息
    """
    from ultrarag.paperagent.report_manager import get_report_manager
    
    # 处理 content（可能是列表）
    if isinstance(content, list):
        content = content[0] if content else ""
    content = str(content) if content else ""
    
    # 处理 topics（可能是字符串或嵌套列表）
    if isinstance(topics, str):
        topics = [topics]
    elif isinstance(topics, list) and topics:
        # 可能是 [["topic1"]] 这样的嵌套列表
        if isinstance(topics[0], list):
            topics = topics[0]
        # 确保都是字符串
        topics = [str(t) for t in topics if t]
    else:
        topics = []
    
    # 如果没有 title，从 topics 生成
    if not title and topics:
        title = f"{topics[0]} - 研究报告"
    elif not title:
        title = "研究报告"
    
    # 自动从注册表获取引用的论文
    cited_papers = ArxivCitationRegistry.get_all_cited_papers()
    app.logger.info(f"[paperagent] Auto-retrieved {len(cited_papers)} cited papers for report")
    
    rm = get_report_manager()
    report = rm.create_report(
        user_id=DEFAULT_USER_ID,
        title=title,
        content=content,
        topics=topics if topics else None,
        categories=categories,
        keywords=keywords,
        cited_papers=cited_papers
    )
    
    app.logger.info(f"[paperagent] Created report {report.report_id}")
    
    return {"report": report.to_dict()}


@app.tool(output="report_id,content,status->report")
def update_report(
    report_id: str,
    content: str,
    status: str = None
) -> Dict[str, Optional[Dict]]:
    """更新报告内容"""
    from ultrarag.paperagent.report_manager import get_report_manager
    
    rm = get_report_manager()
    report = rm.update_report_content(report_id, content, status)
    
    return {"report": report.to_dict() if report else None}


@app.tool(output="report_id->report")
def get_report(report_id: str) -> Dict[str, Optional[Dict]]:
    """获取报告"""
    from ultrarag.paperagent.report_manager import get_report_manager
    
    rm = get_report_manager()
    report = rm.get_report(report_id)
    
    return {"report": report.to_dict() if report else None}


@app.tool(output="user_id,status,limit->reports")
def list_reports(
    user_id: str = "",
    status: str = "",
    limit: int = 50
) -> Dict[str, List[Dict]]:
    """列出报告"""
    from ultrarag.paperagent.report_manager import get_report_manager
    
    rm = get_report_manager()
    reports = rm.list_reports(
        user_id=user_id if user_id else None,
        status=status if status else None,
        limit=limit
    )
    
    return {"reports": reports}


@app.tool(output="report_id->markdown")
def export_report(report_id: str) -> Dict[str, Optional[str]]:
    """导出报告为 Markdown"""
    from ultrarag.paperagent.report_manager import get_report_manager
    
    rm = get_report_manager()
    markdown = rm.export_report_markdown(report_id)
    
    return {"markdown": markdown}


@app.tool(output="report_id->success")
def complete_report(report_id: str) -> Dict[str, bool]:
    """标记报告为完成"""
    from ultrarag.paperagent.report_manager import get_report_manager
    
    rm = get_report_manager()
    report = rm.complete_report(report_id)
    
    return {"success": report is not None}


# ==================== 语料库工具 ====================

@app.tool(output="categories,query,max_papers,corpus_name->corpus_path")
def build_paper_corpus(
    categories: List[str] = None,
    query: str = "",
    max_papers: int = 500,
    corpus_name: str = "default"
) -> Dict[str, str]:
    """构建论文语料库"""
    from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
    
    km = get_knowledge_manager()
    path = km.build_corpus(
        categories=categories,
        query=query,
        max_papers=max_papers,
        corpus_name=corpus_name
    )
    
    app.logger.info(f"[paperagent] Built corpus: {path}")
    
    return {"corpus_path": str(path)}


@app.tool(output="None->corpora")
def list_paper_corpora() -> Dict[str, List[Dict]]:
    """列出所有论文语料库"""
    from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
    
    km = get_knowledge_manager()
    corpora = km.list_corpora()
    
    return {"corpora": corpora}


@app.tool(output="None->stats")
def get_knowledge_stats() -> Dict[str, Dict]:
    """获取知识库统计信息"""
    from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
    
    km = get_knowledge_manager()
    stats = km.get_stats()
    
    return {"stats": stats}


# ==================== 高级研究工具 ====================

@app.tool(output="topic->research_context")
def prepare_research_context(topic: List[str]) -> Dict[str, Dict]:
    """准备研究上下文（单用户模式）"""
    from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
    
    km = get_knowledge_manager()
    
    # 搜索相关论文
    topic = topic[0] if isinstance(topic, list) else topic
    papers = km.search_papers(topic, max_results=20)
    
    app.logger.info(f"[paperagent] Prepared context for topic: {topic}, found {len(papers)} papers")
    
    return {
        "research_context": {
            "topic": topic,
            "relevant_papers": [p.to_dict() for p in papers[:10]],
        }
    }


# ==================== 在线检索工具（用于 Pipeline）====================

@app.tool(output="keywords_ls->ret_psg_ls,psg_ids_ls")
def arxiv_online_search(
    keywords_ls: List[List[str]],
    max_per_keyword: int = 5
) -> Dict[str, Any]:
    """
    在线搜索 arXiv 论文（用于替代 Milvus 检索）
    
    输入: keywords_ls - 每个 query 对应的关键词列表，如 [["RAG", "retrieval"], ["LLM", "agent"]]
    输出: ret_psg_ls - 检索到的段落列表（与 retriever 格式兼容）
          psg_ids_ls - 段落 ID 列表
    """
    from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
    
    km = get_knowledge_manager()
    
    all_passages = []
    all_ids = []
    
    for keywords in keywords_ls:
        batch_passages = []
        batch_ids = []
        seen_ids = set()
        
        for keyword in keywords:
            if not keyword or not keyword.strip():
                continue
            
            try:
                papers = km.search_papers(
                    query=keyword.strip(),
                    max_results=max_per_keyword
                )
                
                for paper in papers:
                    if paper.arxiv_id in seen_ids:
                        continue
                    seen_ids.add(paper.arxiv_id)
                    
                    # 注册论文到引用注册表，获取引用 ID
                    paper_info = {
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "authors": paper.authors[:5],
                        "abstract": paper.abstract[:500] if paper.abstract else "",
                        "categories": paper.categories,
                        "published": paper.published,
                        "arxiv_url": f"https://arxiv.org/abs/{paper.arxiv_id}"
                    }
                    ArxivCitationRegistry.register_paper(paper.arxiv_id, paper_info)
                    
                    # 格式化为段落格式（与 retriever 兼容）
                    passage = f"【论文】{paper.title}\n\n" \
                              f"【作者】{', '.join(paper.authors[:5])}\n" \
                              f"【分类】{', '.join(paper.categories)}\n" \
                              f"【发布日期】{paper.published}\n" \
                              f"【arXiv ID】{paper.arxiv_id}\n\n" \
                              f"【摘要】{paper.abstract}"
                    
                    batch_passages.append(passage)
                    batch_ids.append(paper.arxiv_id)
                    
            except Exception as e:
                app.logger.warning(f"[paperagent] Search failed for '{keyword}': {e}")
                continue
        
        all_passages.append(batch_passages)
        all_ids.append(batch_ids)
    
    app.logger.info(f"[paperagent] Online search completed: {sum(len(p) for p in all_passages)} papers found")
    
    return {
        "ret_psg_ls": all_passages,
        "psg_ids_ls": all_ids
    }


# ==================== arXiv Citation Registry ====================

class ArxivCitationRegistry:
    """arXiv 论文引用注册表，用于跟踪论文引用"""
    _id_counter: int = 0  # 全局计数器
    _arxiv_id_to_citation: Dict[str, int] = {}  # arxiv_id -> citation_id
    _citation_to_paper: Dict[int, Dict] = {}  # citation_id -> paper_info
    
    @classmethod
    def reset(cls):
        """重置注册表"""
        cls._id_counter = 0
        cls._arxiv_id_to_citation = {}
        cls._citation_to_paper = {}
    
    @classmethod
    def register_paper(cls, arxiv_id: str, paper_info: Dict) -> int:
        """注册论文并分配引用 ID"""
        if arxiv_id in cls._arxiv_id_to_citation:
            return cls._arxiv_id_to_citation[arxiv_id]
        
        cls._id_counter += 1
        citation_id = cls._id_counter
        cls._arxiv_id_to_citation[arxiv_id] = citation_id
        cls._citation_to_paper[citation_id] = paper_info
        return citation_id
    
    @classmethod
    def get_citation_id(cls, arxiv_id: str) -> Optional[int]:
        """获取论文的引用 ID"""
        return cls._arxiv_id_to_citation.get(arxiv_id)
    
    @classmethod
    def get_paper(cls, citation_id: int) -> Optional[Dict]:
        """根据引用 ID 获取论文信息"""
        return cls._citation_to_paper.get(citation_id)
    
    @classmethod
    def get_all_cited_papers(cls) -> List[Dict]:
        """获取所有已引用的论文列表（按引用 ID 排序）"""
        papers = []
        for citation_id in sorted(cls._citation_to_paper.keys()):
            paper = cls._citation_to_paper[citation_id].copy()
            paper['citation_id'] = citation_id
            papers.append(paper)
        return papers


@app.tool(output="ret_psg_ls,psg_ids_ls->retrieved_info_ls,ret_psg")
def process_arxiv_passages_with_citation(
    ret_psg_ls: List[List[str]],
    psg_ids_ls: List[List[str]] = None,
    top_k: int = 20
) -> Dict[str, Any]:
    """
    处理 arXiv 检索结果并添加引用 ID（替代 surveycpm_process_passages_with_citation）
    
    输入格式: ret_psg_ls - List[List[str]]，每个元素是一个 batch 的段落列表
             psg_ids_ls - 对应的 arxiv_id 列表
    输出:
        - retrieved_info_ls: 格式化的段落字符串列表（用于 prompt）
        - ret_psg: 带引用 ID 的段落列表（用于前端渲染）
    """
    retrieved_info_ls = []
    ret_psg_output = []
    
    for query_idx, passages in enumerate(ret_psg_ls):
        if not passages:
            retrieved_info_ls.append("")
            ret_psg_output.append([])
            continue
        
        # 获取对应的 arxiv_ids（如果有）
        arxiv_ids = psg_ids_ls[query_idx] if psg_ids_ls and query_idx < len(psg_ids_ls) else []
        
        seen_ids = set()
        all_passages = []
        
        for idx, psg in enumerate(passages[:top_k]):
            arxiv_id = arxiv_ids[idx] if idx < len(arxiv_ids) else None
            
            # 使用 arxiv_id 去重（如果有）
            if arxiv_id:
                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)
                
                # 获取已注册的引用 ID
                citation_id = ArxivCitationRegistry.get_citation_id(arxiv_id)
                if citation_id is None:
                    # 如果没有注册，说明是漏掉的，跳过
                    app.logger.warning(f"[paperagent] Paper {arxiv_id} not registered in citation registry")
                    continue
            else:
                # 没有 arxiv_id 的情况，使用内容哈希
                content_hash = hash(psg)
                if content_hash in seen_ids:
                    continue
                seen_ids.add(content_hash)
                citation_id = len(all_passages) + 1
            
            cited_psg = f"[{citation_id}] {psg}"
            all_passages.append(cited_psg)
        
        info = "\n\n".join(all_passages).strip()
        retrieved_info_ls.append(info)
        ret_psg_output.append(all_passages)
    
    app.logger.info(f"[paperagent] Processed {sum(len(p) for p in ret_psg_output)} passages with citations")
    
    return {
        "retrieved_info_ls": retrieved_info_ls,
        "ret_psg": ret_psg_output
    }


@app.tool()
def reset_arxiv_citation_registry() -> Dict[str, bool]:
    """重置 arXiv 引用注册表"""
    ArxivCitationRegistry.reset()
    app.logger.info("[paperagent] Citation registry reset")
    return {"success": True}


@app.tool()
def get_cited_papers() -> Dict[str, List[Dict]]:
    """获取当前所有已引用的论文信息（用于报告生成）"""
    cited_papers = ArxivCitationRegistry.get_all_cited_papers()
    app.logger.info(f"[paperagent] Retrieved {len(cited_papers)} cited papers")
    return {"cited_papers": cited_papers}


@app.tool(output="query_ls->ret_psg_ls,psg_ids_ls")
def arxiv_search_by_queries(
    query_ls: List[str],
    max_results: int = 10
) -> Dict[str, Any]:
    """
    按查询列表在线搜索 arXiv 论文
    
    输入: query_ls - 查询列表，如 ["RAG 检索增强生成", "大语言模型应用"]
    输出: ret_psg_ls - 检索到的段落列表
          psg_ids_ls - 段落 ID 列表
    """
    from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
    
    km = get_knowledge_manager()
    
    all_passages = []
    all_ids = []
    
    for query in query_ls:
        passages = []
        ids = []
        
        if query and query.strip():
            try:
                papers = km.search_papers(
                    query=query.strip(),
                    max_results=max_results
                )
                
                for paper in papers:
                    passage = f"【论文】{paper.title}\n\n" \
                              f"【作者】{', '.join(paper.authors[:5])}\n" \
                              f"【分类】{', '.join(paper.categories)}\n" \
                              f"【发布日期】{paper.published}\n" \
                              f"【arXiv ID】{paper.arxiv_id}\n\n" \
                              f"【摘要】{paper.abstract}"
                    
                    passages.append(passage)
                    ids.append(paper.arxiv_id)
                    
            except Exception as e:
                app.logger.warning(f"[paperagent] Search failed for '{query}': {e}")
        
        all_passages.append(passages)
        all_ids.append(ids)
    
    app.logger.info(f"[paperagent] Query search completed: {sum(len(p) for p in all_passages)} papers found")
    
    return {
        "ret_psg_ls": all_passages,
        "psg_ids_ls": all_ids
    }


@app.tool(output="papers_ls->formatted_passages_ls")
def format_papers_as_passages(
    papers_ls: List[List[Dict]]
) -> Dict[str, List[List[str]]]:
    """将论文列表格式化为段落列表（用于 citation）"""
    formatted = []
    
    for papers in papers_ls:
        batch = []
        for paper in papers:
            passage = f"【论文】{paper.get('title', 'Unknown')}\n\n" \
                      f"【作者】{', '.join(paper.get('authors', [])[:5])}\n" \
                      f"【分类】{', '.join(paper.get('categories', []))}\n" \
                      f"【发布日期】{paper.get('published', 'Unknown')}\n" \
                      f"【arXiv ID】{paper.get('arxiv_id', 'Unknown')}\n\n" \
                      f"【摘要】{paper.get('abstract', '')}"
            batch.append(passage)
        formatted.append(batch)
    
    return {"formatted_passages_ls": formatted}


if __name__ == "__main__":
    app.run(transport="stdio")

