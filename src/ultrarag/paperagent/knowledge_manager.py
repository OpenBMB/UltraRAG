"""
知识管理模块

管理 arXiv 论文的爬取、存储、索引和检索。
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib

from .arxiv_crawler import ArxivCrawler, Paper, SearchResult
from .cs_categories import CS_SUBCATEGORIES, get_category

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """论文元数据，用于快速检索"""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    primary_category: str
    published: str
    keywords: List[str] = field(default_factory=list)  # 提取的关键词
    citation_count: int = 0
    embedding_id: Optional[str] = None  # 在向量库中的 ID
    
    def to_dict(self) -> Dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "primary_category": self.primary_category,
            "published": self.published,
            "keywords": self.keywords,
            "citation_count": self.citation_count,
            "embedding_id": self.embedding_id,
        }
    
    @classmethod
    def from_paper(cls, paper: Paper) -> "PaperMetadata":
        """从 Paper 对象创建元数据"""
        return cls(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            abstract=paper.abstract,
            authors=paper.authors,
            categories=paper.categories,
            primary_category=paper.primary_category,
            published=paper.published.isoformat(),
            keywords=[],
        )
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PaperMetadata":
        return cls(**data)
    
    def to_corpus_entry(self) -> Dict:
        """转换为语料库条目格式"""
        return {
            "id": self.arxiv_id,
            "contents": f"Title: {self.title}\n\nAbstract: {self.abstract}",
            "metadata": {
                "arxiv_id": self.arxiv_id,
                "title": self.title,
                "authors": self.authors,
                "categories": self.categories,
                "published": self.published,
            }
        }


class KnowledgeManager:
    """知识管理器"""
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        crawler: Optional[ArxivCrawler] = None
    ):
        if storage_dir is None:
            # 从 src/ultrarag/paperagent/ 向上 4 级到项目根目录
            storage_dir = Path(__file__).parent.parent.parent.parent / "data" / "paper_knowledge"
        
        self.storage_dir = Path(storage_dir)
        self.metadata_dir = self.storage_dir / "metadata"
        self.corpus_dir = self.storage_dir / "corpus"
        self.pdf_dir = self.storage_dir / "pdfs"
        self.index_dir = self.storage_dir / "index"
        
        # 创建目录
        for d in [self.metadata_dir, self.corpus_dir, self.pdf_dir, self.index_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.crawler = crawler or ArxivCrawler()
        
        # 元数据缓存
        self._metadata_cache: Dict[str, PaperMetadata] = {}
        self._load_metadata_cache()
    
    def _load_metadata_cache(self):
        """加载元数据缓存"""
        index_file = self.metadata_dir / "index.jsonl"
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        meta = PaperMetadata.from_dict(data)
                        self._metadata_cache[meta.arxiv_id] = meta
                    except:
                        continue
        logger.info(f"Loaded {len(self._metadata_cache)} papers to cache")
    
    def _save_metadata(self, meta: PaperMetadata):
        """保存单个元数据"""
        self._metadata_cache[meta.arxiv_id] = meta
        
        # 追加写入索引文件
        index_file = self.metadata_dir / "index.jsonl"
        with open(index_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta.to_dict(), ensure_ascii=False) + "\n")
    
    def _rebuild_index(self):
        """重建索引文件"""
        index_file = self.metadata_dir / "index.jsonl"
        with open(index_file, "w", encoding="utf-8") as f:
            for meta in self._metadata_cache.values():
                f.write(json.dumps(meta.to_dict(), ensure_ascii=False) + "\n")
    
    # ========== 论文获取 ==========
    
    def search_papers(
        self,
        query: str,
        category: Optional[str] = None,
        max_results: int = 20,
        sort_by: str = "relevance"
    ) -> List[PaperMetadata]:
        """搜索论文"""
        results = self.crawler.search(
            query=query,
            category=category,
            max_results=max_results,
            sort_by=sort_by
        )
        
        papers = []
        for paper in results.papers:
            meta = PaperMetadata.from_paper(paper)
            # 检查是否已存在
            if meta.arxiv_id not in self._metadata_cache:
                self._save_metadata(meta)
            else:
                meta = self._metadata_cache[meta.arxiv_id]
            papers.append(meta)
        
        return papers
    
    async def search_papers_async(
        self,
        query: str,
        category: Optional[str] = None,
        max_results: int = 20,
        sort_by: str = "relevance"
    ) -> List[PaperMetadata]:
        """异步搜索论文"""
        results = await self.crawler.search_async(
            query=query,
            category=category,
            max_results=max_results,
            sort_by=sort_by
        )
        
        papers = []
        for paper in results.papers:
            meta = PaperMetadata.from_paper(paper)
            if meta.arxiv_id not in self._metadata_cache:
                self._save_metadata(meta)
            else:
                meta = self._metadata_cache[meta.arxiv_id]
            papers.append(meta)
        
        return papers
    
    def get_recent_papers(
        self,
        category: str,
        days: int = 7,
        max_results: int = 50
    ) -> List[PaperMetadata]:
        """获取某分类最近的论文"""
        return self.search_papers(
            query="",
            category=category,
            max_results=max_results,
            sort_by="submittedDate"
        )
    
    def get_trending_papers(
        self,
        categories: List[str],
        max_per_category: int = 10
    ) -> Dict[str, List[PaperMetadata]]:
        """获取热门论文（按分类）"""
        trending = {}
        for cat in categories:
            papers = self.get_recent_papers(cat, days=7, max_results=max_per_category)
            trending[cat] = papers
        return trending
    
    def get_paper(self, arxiv_id: str) -> Optional[PaperMetadata]:
        """获取单篇论文"""
        if arxiv_id in self._metadata_cache:
            return self._metadata_cache[arxiv_id]
        
        paper = self.crawler.get_paper(arxiv_id)
        if paper:
            meta = PaperMetadata.from_paper(paper)
            self._save_metadata(meta)
            return meta
        return None
    
    def get_papers(self, arxiv_ids: List[str]) -> List[PaperMetadata]:
        """批量获取论文"""
        results = []
        missing_ids = []
        
        for aid in arxiv_ids:
            if aid in self._metadata_cache:
                results.append(self._metadata_cache[aid])
            else:
                missing_ids.append(aid)
        
        if missing_ids:
            papers = self.crawler.get_papers(missing_ids)
            for paper in papers:
                meta = PaperMetadata.from_paper(paper)
                self._save_metadata(meta)
                results.append(meta)
        
        return results
    
    # ========== 语料库管理 ==========
    
    def build_corpus(
        self,
        categories: List[str] = None,
        query: str = None,
        max_papers: int = 500,
        corpus_name: str = "default"
    ) -> Path:
        """构建论文语料库"""
        papers = []
        
        if categories:
            for cat in categories:
                cat_papers = self.search_papers(
                    query=query or "",
                    category=cat,
                    max_results=max_papers // len(categories),
                    sort_by="submittedDate"
                )
                papers.extend(cat_papers)
        elif query:
            papers = self.search_papers(
                query=query,
                max_results=max_papers,
                sort_by="relevance"
            )
        
        # 写入语料库文件
        corpus_path = self.corpus_dir / f"{corpus_name}.jsonl"
        with open(corpus_path, "w", encoding="utf-8") as f:
            for meta in papers:
                entry = meta.to_corpus_entry()
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.info(f"Built corpus with {len(papers)} papers: {corpus_path}")
        return corpus_path
    
    def get_corpus_path(self, corpus_name: str = "default") -> Optional[Path]:
        """获取语料库路径"""
        path = self.corpus_dir / f"{corpus_name}.jsonl"
        return path if path.exists() else None
    
    def list_corpora(self) -> List[Dict]:
        """列出所有语料库"""
        corpora = []
        for path in self.corpus_dir.glob("*.jsonl"):
            # 计算条目数
            with open(path, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
            corpora.append({
                "name": path.stem,
                "path": str(path),
                "count": count,
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            })
        return corpora
    
    # ========== 论文推荐 ==========
    
    def get_recommended_papers(
        self,
        interests: List[str],
        keywords: List[str] = None,
        exclude_ids: List[str] = None,
        max_results: int = 20
    ) -> List[PaperMetadata]:
        """基于兴趣获取推荐论文"""
        all_papers = []
        exclude_set = set(exclude_ids or [])
        
        # 按分类获取
        for interest in interests[:3]:  # 限制查询数量
            papers = self.search_papers(
                query="",
                category=interest,
                max_results=max_results // 2,
                sort_by="submittedDate"
            )
            all_papers.extend(papers)
        
        # 按关键词获取
        if keywords:
            for kw in keywords[:3]:
                papers = self.search_papers(
                    query=kw,
                    max_results=max_results // 3,
                    sort_by="relevance"
                )
                all_papers.extend(papers)
        
        # 去重并排除
        seen = set()
        unique_papers = []
        for p in all_papers:
            if p.arxiv_id not in seen and p.arxiv_id not in exclude_set:
                seen.add(p.arxiv_id)
                unique_papers.append(p)
        
        return unique_papers[:max_results]
    
    def get_onboarding_papers(
        self,
        categories: Optional[List[str]] = None,
        max_results: int = 20,
        refresh: bool = False
    ) -> List[PaperMetadata]:
        """获取用于 onboarding 的代表性论文
        
        Args:
            categories: 分类列表，如果为空则获取热门分类
            max_results: 最大返回数量
            refresh: 是否强制从 arXiv 刷新获取新论文（会获取不同的论文）
        """
        import random
        
        all_papers = []
        
        # 如果没有指定分类，使用热门分类
        if not categories:
            categories = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.IR"]
        
        # 每个分类获取的数量
        per_cat = max(max_results // len(categories), 5)
        
        for cat in categories:
            if refresh:
                # 强制刷新：使用 ArxivCrawler 获取论文
                try:
                    from ultrarag.paperagent.arxiv_crawler import ArxivCrawler
                    crawler = ArxivCrawler()
                    
                    # 混合策略：随机选择排序方式，增加多样性
                    sort_options = [
                        ("relevance", "descending"),      # 相关性（经典/热门）
                        ("submittedDate", "descending"),  # 最新提交
                        ("lastUpdatedDate", "descending") # 最近更新
                    ]
                    sort_by, sort_order = random.choice(sort_options)
                    
                    # 随机起始位置，获取不同的论文
                    start_offset = random.randint(0, 50)
                    
                    result = crawler.search(
                        query="",
                        category=cat,
                        max_results=per_cat + 10,  # 多取一些用于随机选择
                        start=start_offset,
                        sort_by=sort_by,
                        sort_order=sort_order
                    )
                    
                    # 将 ArxivPaper 转换为 PaperMetadata
                    cat_papers = []
                    for paper in result.papers:
                        metadata = PaperMetadata(
                            arxiv_id=paper.arxiv_id,
                            title=paper.title,
                            authors=paper.authors,
                            abstract=paper.abstract,
                            categories=paper.categories,
                            published=paper.published.isoformat() if paper.published else "",
                            updated=paper.updated.isoformat() if paper.updated else "",
                            pdf_url=paper.pdf_url,
                            arxiv_url=paper.arxiv_url,
                        )
                        cat_papers.append(metadata)
                    
                    # 随机打乱并选取
                    random.shuffle(cat_papers)
                    all_papers.extend(cat_papers[:per_cat])
                    
                except Exception as e:
                    logger.error(f"Failed to refresh papers for {cat}: {e}")
                    # 回退到缓存搜索
                    papers = self.search_papers(
                        query="",
                        category=cat,
                        max_results=per_cat,
                        sort_by="submittedDate"
                    )
                    all_papers.extend(papers)
            else:
                papers = self.search_papers(
                    query="",
                    category=cat,
                    max_results=per_cat,
                    sort_by="submittedDate"
                )
                all_papers.extend(papers)
        
        # 去重
        seen = set()
        unique_papers = []
        for p in all_papers:
            if p.arxiv_id not in seen:
                seen.add(p.arxiv_id)
                unique_papers.append(p)
        
        # 刷新时再次打乱顺序，确保每次显示不同
        if refresh:
            random.shuffle(unique_papers)
        
        return unique_papers[:max_results]
    
    # ========== 统计信息 ==========
    
    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        return {
            "total_papers": len(self._metadata_cache),
            "corpora": self.list_corpora(),
            "categories": self._get_category_stats(),
        }
    
    def _get_category_stats(self) -> Dict[str, int]:
        """获取分类统计"""
        stats = {}
        for meta in self._metadata_cache.values():
            for cat in meta.categories:
                stats[cat] = stats.get(cat, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))


# 单例
_knowledge_manager: Optional[KnowledgeManager] = None

def get_knowledge_manager() -> KnowledgeManager:
    """获取知识管理器单例"""
    global _knowledge_manager
    if _knowledge_manager is None:
        _knowledge_manager = KnowledgeManager()
    return _knowledge_manager

