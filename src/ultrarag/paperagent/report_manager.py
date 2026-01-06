"""
报告管理模块

管理研究报告的生成、存储和检索。
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResearchReport:
    """研究报告数据模型"""
    report_id: str
    user_id: str
    title: str
    content: str  # Markdown 格式的报告内容
    
    # 报告元信息
    topics: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # 引用的论文
    cited_papers: List[Dict] = field(default_factory=list)
    
    # 时间戳
    created_at: str = ""
    updated_at: str = ""
    
    # 状态
    status: str = "draft"  # draft, completed, archived
    
    # 摘要（用于预览）
    summary: str = ""
    
    # 报告类型
    report_type: str = "hotspot"  # hotspot, idea, survey
    
    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "topics": self.topics,
            "categories": self.categories,
            "keywords": self.keywords,
            "cited_papers": self.cited_papers,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "summary": self.summary,
            "report_type": self.report_type,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ResearchReport":
        return cls(**data)
    
    def get_preview(self, max_length: int = 200) -> str:
        """获取报告预览"""
        if self.summary:
            return self.summary[:max_length]
        # 从内容中提取
        content = self.content.replace("#", "").replace("*", "").strip()
        return content[:max_length] + "..." if len(content) > max_length else content


class ReportManager:
    """报告管理器"""
    
    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir is None:
            # 从 src/ultrarag/paperagent/ 向上 4 级到项目根目录
            storage_dir = Path(__file__).parent.parent.parent.parent / "data" / "reports"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[ReportManager] Storage dir: {self.storage_dir}")
        
        # 索引文件
        self.index_path = self.storage_dir / "index.json"
        self._index: Dict[str, Dict] = {}
        self._load_index()
    
    def _load_index(self):
        """加载索引"""
        if self.index_path.exists():
            try:
                self._index = json.loads(self.index_path.read_text(encoding="utf-8"))
            except:
                self._index = {}
    
    def _save_index(self):
        """保存索引"""
        self.index_path.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def _get_report_path(self, report_id: str) -> Path:
        """获取报告文件路径"""
        return self.storage_dir / f"{report_id}.json"
    
    def create_report(
        self,
        user_id: str,
        title: str,
        content: str = "",
        topics: List[str] = None,
        categories: List[str] = None,
        keywords: List[str] = None,
        cited_papers: List[Dict] = None,
        report_type: str = "hotspot",
    ) -> ResearchReport:
        """创建新报告"""
        now = datetime.now().isoformat()
        report_id = f"rpt_{int(datetime.now().timestamp())}_{str(uuid.uuid4())[:8]}"
        
        report = ResearchReport(
            report_id=report_id,
            user_id=user_id,
            title=title,
            content=content,
            topics=topics or [],
            categories=categories or [],
            keywords=keywords or [],
            cited_papers=cited_papers or [],
            created_at=now,
            updated_at=now,
            status="completed",
            summary=self._generate_summary(content) if content else "",
            report_type=report_type,
        )
        
        self.save_report(report)
        return report
    
    def save_report(self, report: ResearchReport):
        """保存报告"""
        report.updated_at = datetime.now().isoformat()
        
        # 保存完整报告
        path = self._get_report_path(report.report_id)
        path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        # 更新索引
        self._index[report.report_id] = {
            "report_id": report.report_id,
            "user_id": report.user_id,
            "title": report.title,
            "topics": report.topics,
            "status": report.status,
            "created_at": report.created_at,
            "updated_at": report.updated_at,
            "summary": report.get_preview(100),
            "report_type": report.report_type,
        }
        self._save_index()
    
    def get_report(self, report_id: str) -> Optional[ResearchReport]:
        """获取报告"""
        path = self._get_report_path(report_id)
        if not path.exists():
            return None
        
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ResearchReport.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load report {report_id}: {e}")
            return None
    
    def delete_report(self, report_id: str) -> bool:
        """删除报告"""
        path = self._get_report_path(report_id)
        if path.exists():
            path.unlink()
            if report_id in self._index:
                del self._index[report_id]
                self._save_index()
            return True
        return False
    
    def list_reports(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """列出报告"""
        reports = list(self._index.values())
        
        # 过滤
        if user_id:
            reports = [r for r in reports if r["user_id"] == user_id]
        if status:
            reports = [r for r in reports if r["status"] == status]
        
        # 排序（按更新时间倒序）
        reports.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return reports[:limit]
    
    def update_report_content(
        self,
        report_id: str,
        content: str,
        status: str = None
    ) -> Optional[ResearchReport]:
        """更新报告内容"""
        report = self.get_report(report_id)
        if not report:
            return None
        
        report.content = content
        report.summary = self._generate_summary(content)
        if status:
            report.status = status
        
        self.save_report(report)
        return report
    
    def complete_report(self, report_id: str) -> Optional[ResearchReport]:
        """标记报告为完成"""
        report = self.get_report(report_id)
        if not report:
            return None
        
        report.status = "completed"
        self.save_report(report)
        return report
    
    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """生成报告摘要"""
        if not content:
            return ""
        
        # 简单提取：去掉 Markdown 标记，取前 N 个字符
        lines = content.split("\n")
        text_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # 去掉 Markdown 格式
                line = line.replace("*", "").replace("_", "").replace("`", "")
                text_lines.append(line)
        
        text = " ".join(text_lines)
        return text[:max_length] + "..." if len(text) > max_length else text
    
    def search_reports(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """搜索报告"""
        query_lower = query.lower()
        results = []
        
        for info in self._index.values():
            if user_id and info["user_id"] != user_id:
                continue
            
            # 简单的关键词匹配
            if (query_lower in info["title"].lower() or
                query_lower in info.get("summary", "").lower() or
                any(query_lower in t.lower() for t in info.get("topics", []))):
                results.append(info)
        
        # 按相关性排序（简单实现：标题匹配优先）
        results.sort(
            key=lambda x: (
                query_lower in x["title"].lower(),
                x.get("updated_at", "")
            ),
            reverse=True
        )
        
        return results[:limit]
    
    def export_report_markdown(self, report_id: str) -> Optional[str]:
        """导出报告为 Markdown"""
        report = self.get_report(report_id)
        if not report:
            return None
        
        # 构建完整的 Markdown
        lines = [
            f"# {report.title}",
            "",
            f"*生成时间: {report.created_at}*",
            "",
        ]
        
        if report.topics:
            lines.append(f"**主题**: {', '.join(report.topics)}")
            lines.append("")
        
        if report.categories:
            lines.append(f"**分类**: {', '.join(report.categories)}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append(report.content)
        
        if report.cited_papers:
            lines.append("")
            lines.append("## 参考文献")
            lines.append("")
            for i, paper in enumerate(report.cited_papers, 1):
                arxiv_id = paper.get("arxiv_id", "")
                title = paper.get("title", "Unknown")
                authors = paper.get("authors", [])
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += " et al."
                lines.append(f"[{i}] {author_str}. *{title}*. arXiv:{arxiv_id}")
        
        return "\n".join(lines)


# 单例
_report_manager: Optional[ReportManager] = None

def get_report_manager() -> ReportManager:
    """获取报告管理器单例"""
    global _report_manager
    if _report_manager is None:
        _report_manager = ReportManager()
    return _report_manager

