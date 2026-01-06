"""
arXiv 论文爬取器

提供从 arXiv 搜索、获取和下载论文的功能。
使用 arXiv API: https://arxiv.org/help/api/
"""

import asyncio
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlencode

import aiohttp
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# arXiv API 相关常量
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"
ARXIV_ABS_BASE = "https://arxiv.org/abs"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# arXiv 分类映射
ARXIV_CATEGORIES = {
    "cs": "Computer Science",
    "math": "Mathematics",
    "physics": "Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics",
    "eess": "Electrical Engineering and Systems Science",
    "econ": "Economics",
}


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class Paper:
    """arXiv 论文数据模型"""
    
    arxiv_id: str  # arXiv ID，如 "2301.00001"
    title: str  # 论文标题
    abstract: str  # 摘要
    authors: List[str]  # 作者列表
    categories: List[str]  # 分类，如 ["cs.CL", "cs.AI"]
    primary_category: str  # 主分类
    published: datetime  # 发布日期
    updated: datetime  # 更新日期
    pdf_url: str  # PDF 下载链接
    html_url: str  # arXiv 页面链接
    comment: Optional[str] = None  # 作者备注
    journal_ref: Optional[str] = None  # 期刊引用
    doi: Optional[str] = None  # DOI
    
    def __str__(self) -> str:
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += f" et al. ({len(self.authors)} authors)"
        return f"[{self.arxiv_id}] {self.title}\n  Authors: {authors_str}\n  Published: {self.published.strftime('%Y-%m-%d')}"
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "primary_category": self.primary_category,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "pdf_url": self.pdf_url,
            "html_url": self.html_url,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "doi": self.doi,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        """从字典创建 Paper 对象"""
        return cls(
            arxiv_id=data["arxiv_id"],
            title=data["title"],
            abstract=data["abstract"],
            authors=data["authors"],
            categories=data["categories"],
            primary_category=data["primary_category"],
            published=datetime.fromisoformat(data["published"]) if isinstance(data["published"], str) else data["published"],
            updated=datetime.fromisoformat(data["updated"]) if isinstance(data["updated"], str) else data["updated"],
            pdf_url=data["pdf_url"],
            html_url=data["html_url"],
            comment=data.get("comment"),
            journal_ref=data.get("journal_ref"),
            doi=data.get("doi"),
        )


@dataclass
class SearchResult:
    """搜索结果数据模型"""
    
    query: str  # 搜索查询
    total_results: int  # 总结果数
    start_index: int  # 起始索引
    papers: List[Paper] = field(default_factory=list)  # 论文列表
    
    def __str__(self) -> str:
        return f"SearchResult(query='{self.query}', total={self.total_results}, returned={len(self.papers)})"
    
    def __iter__(self):
        return iter(self.papers)
    
    def __len__(self):
        return len(self.papers)


# ============================================================================
# 爬虫类
# ============================================================================

class ArxivCrawler:
    """
    arXiv 论文爬取器
    
    支持功能：
    - 根据关键词搜索论文
    - 根据 arXiv ID 获取论文信息
    - 下载论文 PDF
    - 批量下载论文
    - 异步操作支持
    
    使用示例:
        ```python
        crawler = ArxivCrawler()
        
        # 搜索论文
        results = crawler.search("large language model", max_results=10)
        for paper in results:
            print(paper)
        
        # 获取特定论文
        paper = crawler.get_paper("2301.00001")
        
        # 下载 PDF
        crawler.download_pdf(paper, "./papers/")
        
        # 异步批量下载
        await crawler.download_papers_async(results.papers, "./papers/")
        ```
    """
    
    def __init__(
        self,
        request_interval: float = 3.0,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        初始化爬取器
        
        Args:
            request_interval: 请求间隔（秒），arXiv 建议至少 3 秒
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.request_interval = request_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request_time = 0.0
    
    def _wait_for_rate_limit(self):
        """等待以遵守速率限制"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_request_time = time.time()
    
    async def _async_wait_for_rate_limit(self):
        """异步等待以遵守速率限制"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_interval:
            await asyncio.sleep(self.request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _parse_arxiv_id(self, id_or_url: str) -> str:
        """
        解析 arXiv ID
        
        支持格式：
        - 2301.00001
        - arxiv:2301.00001
        - https://arxiv.org/abs/2301.00001
        - https://arxiv.org/pdf/2301.00001.pdf
        """
        id_str = id_or_url.strip()
        
        # 处理 URL
        if "arxiv.org" in id_str:
            match = re.search(r"/(?:abs|pdf)/([^\s/]+?)(?:\.pdf)?$", id_str)
            if match:
                id_str = match.group(1)
        
        # 移除 arxiv: 前缀
        if id_str.lower().startswith("arxiv:"):
            id_str = id_str[6:]
        
        return id_str
    
    def _parse_entry(self, entry: ET.Element) -> Paper:
        """解析 XML entry 为 Paper 对象"""
        # 获取 ID
        id_elem = entry.find(f"{ATOM_NS}id")
        arxiv_id = self._parse_arxiv_id(id_elem.text if id_elem is not None else "")
        
        # 获取标题
        title_elem = entry.find(f"{ATOM_NS}title")
        title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""
        
        # 获取摘要
        summary_elem = entry.find(f"{ATOM_NS}summary")
        abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
        
        # 获取作者
        authors = []
        for author in entry.findall(f"{ATOM_NS}author"):
            name_elem = author.find(f"{ATOM_NS}name")
            if name_elem is not None:
                authors.append(name_elem.text)
        
        # 获取分类
        categories = []
        for category in entry.findall(f"{ATOM_NS}category"):
            term = category.get("term")
            if term:
                categories.append(term)
        
        # 获取主分类
        primary_elem = entry.find(f"{ARXIV_NS}primary_category")
        primary_category = primary_elem.get("term", "") if primary_elem is not None else (categories[0] if categories else "")
        
        # 获取日期
        published_elem = entry.find(f"{ATOM_NS}published")
        published_str = published_elem.text if published_elem is not None else ""
        published = datetime.fromisoformat(published_str.replace("Z", "+00:00")) if published_str else datetime.now()
        
        updated_elem = entry.find(f"{ATOM_NS}updated")
        updated_str = updated_elem.text if updated_elem is not None else ""
        updated = datetime.fromisoformat(updated_str.replace("Z", "+00:00")) if updated_str else published
        
        # 获取备注
        comment_elem = entry.find(f"{ARXIV_NS}comment")
        comment = comment_elem.text if comment_elem is not None else None
        
        # 获取期刊引用
        journal_elem = entry.find(f"{ARXIV_NS}journal_ref")
        journal_ref = journal_elem.text if journal_elem is not None else None
        
        # 获取 DOI
        doi_elem = entry.find(f"{ARXIV_NS}doi")
        doi = doi_elem.text if doi_elem is not None else None
        
        # 构建 URL
        pdf_url = f"{ARXIV_PDF_BASE}/{arxiv_id}.pdf"
        html_url = f"{ARXIV_ABS_BASE}/{arxiv_id}"
        
        return Paper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            categories=categories,
            primary_category=primary_category,
            published=published,
            updated=updated,
            pdf_url=pdf_url,
            html_url=html_url,
            comment=comment,
            journal_ref=journal_ref,
            doi=doi,
        )
    
    def _build_search_query(
        self,
        query: str,
        id_list: Optional[List[str]] = None,
        start: int = 0,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> str:
        """构建搜索查询 URL"""
        params = {
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        
        if id_list:
            params["id_list"] = ",".join(id_list)
        
        if query:
            params["search_query"] = query
        
        return f"{ARXIV_API_BASE}?{urlencode(params)}"
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        category: Optional[str] = None,
    ) -> SearchResult:
        """
        搜索 arXiv 论文
        
        Args:
            query: 搜索查询，支持 arXiv 查询语法
                - 简单搜索: "large language model"
                - 标题搜索: "ti:attention"
                - 作者搜索: "au:hinton"
                - 摘要搜索: "abs:transformer"
                - 组合搜索: "ti:attention AND au:vaswani"
            max_results: 最大返回结果数
            start: 起始索引（用于分页）
            sort_by: 排序方式，可选 "relevance", "lastUpdatedDate", "submittedDate"
            sort_order: 排序顺序，可选 "ascending", "descending"
            category: 限制分类，如 "cs.CL", "cs.AI"
        
        Returns:
            SearchResult: 搜索结果对象
        """
        if category:
            if query:
                query = f"({query}) AND cat:{category}"
            else:
                query = f"cat:{category}"
        
        url = self._build_search_query(query, None, start, max_results, sort_by, sort_order)
        
        self._wait_for_rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.request_interval)
        
        root = ET.fromstring(response.content)
        
        total_elem = None
        for child in root:
            if "totalResults" in child.tag:
                total_elem = child
                break
        total_results = int(total_elem.text) if total_elem is not None else 0
        
        papers = []
        for entry in root.findall(f"{ATOM_NS}entry"):
            try:
                paper = self._parse_entry(entry)
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse entry: {e}")
        
        return SearchResult(
            query=query,
            total_results=total_results,
            start_index=start,
            papers=papers,
        )
    
    async def search_async(
        self,
        query: str,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        category: Optional[str] = None,
    ) -> SearchResult:
        """异步搜索 arXiv 论文"""
        if category:
            if query:
                query = f"({query}) AND cat:{category}"
            else:
                query = f"cat:{category}"
        
        url = self._build_search_query(query, None, start, max_results, sort_by, sort_order)
        
        await self._async_wait_for_rate_limit()
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                        response.raise_for_status()
                        content = await response.read()
                        break
                except aiohttp.ClientError as e:
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(self.request_interval)
        
        root = ET.fromstring(content)
        
        total_elem = None
        for child in root:
            if "totalResults" in child.tag:
                total_elem = child
                break
        total_results = int(total_elem.text) if total_elem is not None else 0
        
        papers = []
        for entry in root.findall(f"{ATOM_NS}entry"):
            try:
                paper = self._parse_entry(entry)
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse entry: {e}")
        
        return SearchResult(
            query=query,
            total_results=total_results,
            start_index=start,
            papers=papers,
        )
    
    def get_paper(self, arxiv_id: str) -> Optional[Paper]:
        """
        根据 arXiv ID 获取论文信息
        
        Args:
            arxiv_id: arXiv ID 或 URL
        
        Returns:
            Paper: 论文对象，如果未找到则返回 None
        """
        arxiv_id = self._parse_arxiv_id(arxiv_id)
        url = self._build_search_query("", [arxiv_id])
        
        self._wait_for_rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.request_interval)
        
        root = ET.fromstring(response.content)
        entries = root.findall(f"{ATOM_NS}entry")
        
        if not entries:
            return None
        
        return self._parse_entry(entries[0])
    
    async def get_paper_async(self, arxiv_id: str) -> Optional[Paper]:
        """异步根据 arXiv ID 获取论文信息"""
        arxiv_id = self._parse_arxiv_id(arxiv_id)
        url = self._build_search_query("", [arxiv_id])
        
        await self._async_wait_for_rate_limit()
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                        response.raise_for_status()
                        content = await response.read()
                        break
                except aiohttp.ClientError as e:
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(self.request_interval)
        
        root = ET.fromstring(content)
        entries = root.findall(f"{ATOM_NS}entry")
        
        if not entries:
            return None
        
        return self._parse_entry(entries[0])
    
    def get_papers(self, arxiv_ids: List[str]) -> List[Paper]:
        """
        批量获取论文信息
        
        Args:
            arxiv_ids: arXiv ID 列表
        
        Returns:
            List[Paper]: 论文列表
        """
        arxiv_ids = [self._parse_arxiv_id(aid) for aid in arxiv_ids]
        url = self._build_search_query("", arxiv_ids, max_results=len(arxiv_ids))
        
        self._wait_for_rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.request_interval)
        
        root = ET.fromstring(response.content)
        
        papers = []
        for entry in root.findall(f"{ATOM_NS}entry"):
            try:
                paper = self._parse_entry(entry)
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse entry: {e}")
        
        return papers
    
    def download_pdf(
        self,
        paper: Union[Paper, str],
        output_dir: Union[str, Path],
        filename: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        """
        下载论文 PDF
        
        Args:
            paper: Paper 对象或 arXiv ID
            output_dir: 输出目录
            filename: 自定义文件名（不含扩展名）
            overwrite: 是否覆盖已存在的文件
        
        Returns:
            Path: 下载的文件路径
        """
        if isinstance(paper, str):
            paper = self.get_paper(paper)
            if paper is None:
                raise ValueError(f"Paper not found: {paper}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{paper.arxiv_id.replace('/', '_')}"
        
        output_path = output_dir / f"{filename}.pdf"
        
        if output_path.exists() and not overwrite:
            logger.info(f"File already exists: {output_path}")
            return output_path
        
        self._wait_for_rate_limit()
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(paper.pdf_url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded: {output_path}")
                return output_path
                
            except requests.RequestException as e:
                logger.warning(f"Download failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.request_interval)
        
        return output_path
    
    async def download_pdf_async(
        self,
        paper: Union[Paper, str],
        output_dir: Union[str, Path],
        filename: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        """异步下载论文 PDF"""
        if isinstance(paper, str):
            paper = await self.get_paper_async(paper)
            if paper is None:
                raise ValueError(f"Paper not found: {paper}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{paper.arxiv_id.replace('/', '_')}"
        
        output_path = output_dir / f"{filename}.pdf"
        
        if output_path.exists() and not overwrite:
            logger.info(f"File already exists: {output_path}")
            return output_path
        
        await self._async_wait_for_rate_limit()
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                try:
                    async with session.get(
                        paper.pdf_url,
                        timeout=aiohttp.ClientTimeout(total=self.timeout * 3)
                    ) as response:
                        response.raise_for_status()
                        content = await response.read()
                        
                        with open(output_path, "wb") as f:
                            f.write(content)
                        
                        logger.info(f"Downloaded: {output_path}")
                        return output_path
                        
                except aiohttp.ClientError as e:
                    logger.warning(f"Download failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(self.request_interval)
        
        return output_path
    
    def download_papers(
        self,
        papers: List[Union[Paper, str]],
        output_dir: Union[str, Path],
        overwrite: bool = False,
        show_progress: bool = True,
    ) -> List[Path]:
        """
        批量下载论文 PDF
        
        Args:
            papers: Paper 对象或 arXiv ID 列表
            output_dir: 输出目录
            overwrite: 是否覆盖已存在的文件
            show_progress: 是否显示进度条
        
        Returns:
            List[Path]: 下载的文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = []
        iterator = tqdm(papers, desc="Downloading papers") if show_progress else papers
        
        for paper in iterator:
            try:
                path = self.download_pdf(paper, output_dir, overwrite=overwrite)
                downloaded.append(path)
            except Exception as e:
                logger.error(f"Failed to download paper: {e}")
        
        return downloaded
    
    async def download_papers_async(
        self,
        papers: List[Union[Paper, str]],
        output_dir: Union[str, Path],
        overwrite: bool = False,
        concurrency: int = 3,
        show_progress: bool = True,
    ) -> List[Path]:
        """
        异步批量下载论文 PDF
        
        Args:
            papers: Paper 对象或 arXiv ID 列表
            output_dir: 输出目录
            overwrite: 是否覆盖已存在的文件
            concurrency: 并发下载数（注意 arXiv 的速率限制）
            show_progress: 是否显示进度条
        
        Returns:
            List[Path]: 下载的文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def download_with_semaphore(paper):
            async with semaphore:
                try:
                    path = await self.download_pdf_async(paper, output_dir, overwrite=overwrite)
                    return path
                except Exception as e:
                    logger.error(f"Failed to download paper: {e}")
                    return None
        
        tasks = [download_with_semaphore(paper) for paper in papers]
        
        if show_progress:
            from tqdm.asyncio import tqdm_asyncio
            results = await tqdm_asyncio.gather(*tasks, desc="Downloading papers")
        else:
            results = await asyncio.gather(*tasks)
        
        downloaded = [r for r in results if r is not None]
        return downloaded
    
    def search_by_author(
        self,
        author: str,
        max_results: int = 10,
        sort_by: str = "submittedDate",
    ) -> SearchResult:
        """
        根据作者搜索论文
        
        Args:
            author: 作者名称
            max_results: 最大返回结果数
            sort_by: 排序方式
        
        Returns:
            SearchResult: 搜索结果
        """
        query = f"au:{author}"
        return self.search(query, max_results=max_results, sort_by=sort_by)
    
    def search_by_title(
        self,
        title: str,
        max_results: int = 10,
    ) -> SearchResult:
        """
        根据标题搜索论文
        
        Args:
            title: 标题关键词
            max_results: 最大返回结果数
        
        Returns:
            SearchResult: 搜索结果
        """
        query = f"ti:{title}"
        return self.search(query, max_results=max_results)
    
    def search_recent(
        self,
        category: str,
        max_results: int = 20,
    ) -> SearchResult:
        """
        获取某分类的最新论文
        
        Args:
            category: 分类，如 "cs.CL", "cs.AI"
            max_results: 最大返回结果数
        
        Returns:
            SearchResult: 搜索结果
        """
        return self.search(
            query="",
            category=category,
            max_results=max_results,
            sort_by="submittedDate",
            sort_order="descending",
        )


# ============================================================================
# 便捷函数
# ============================================================================

def search_arxiv(query: str, max_results: int = 10, **kwargs) -> SearchResult:
    """便捷函数：搜索 arXiv 论文"""
    crawler = ArxivCrawler()
    return crawler.search(query, max_results=max_results, **kwargs)


def get_arxiv_paper(arxiv_id: str) -> Optional[Paper]:
    """便捷函数：获取单篇论文"""
    crawler = ArxivCrawler()
    return crawler.get_paper(arxiv_id)


def download_arxiv_pdf(
    arxiv_id: str,
    output_dir: Union[str, Path] = ".",
    **kwargs
) -> Path:
    """便捷函数：下载论文 PDF"""
    crawler = ArxivCrawler()
    return crawler.download_pdf(arxiv_id, output_dir, **kwargs)
