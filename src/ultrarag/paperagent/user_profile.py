"""
用户画像模块

管理用户的研究兴趣、偏好和历史记录。
支持基于对话更新用户画像。
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ResearchInterest:
    """研究兴趣数据模型"""
    category: str  # CS 分类代码
    weight: float = 1.0  # 兴趣权重 (0-1)
    keywords: List[str] = field(default_factory=list)  # 具体关键词
    last_updated: str = ""  # 最后更新时间
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ResearchInterest":
        return cls(**data)


@dataclass
class PaperInteraction:
    """论文交互记录"""
    arxiv_id: str
    title: str
    categories: List[str]
    interaction_type: str  # "selected", "read", "saved", "cited"
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PaperInteraction":
        return cls(**data)


@dataclass 
class UserProfile:
    """用户画像数据模型"""
    user_id: str
    name: str = ""
    created_at: str = ""
    updated_at: str = ""
    
    # 研究兴趣
    research_interests: List[ResearchInterest] = field(default_factory=list)
    
    # 初始选择的论文 (onboarding 时选的 5 篇)
    initial_papers: List[str] = field(default_factory=list)  # arxiv_ids
    
    # 论文交互历史
    paper_interactions: List[PaperInteraction] = field(default_factory=list)
    
    # 关键词偏好（从对话和论文中提取）
    keyword_preferences: Dict[str, float] = field(default_factory=dict)
    
    # 生成的报告历史
    report_history: List[Dict] = field(default_factory=list)
    
    # 对话摘要（记忆）
    conversation_summaries: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "research_interests": [ri.to_dict() for ri in self.research_interests],
            "initial_papers": self.initial_papers,
            "paper_interactions": [pi.to_dict() for pi in self.paper_interactions],
            "keyword_preferences": self.keyword_preferences,
            "report_history": self.report_history,
            "conversation_summaries": self.conversation_summaries,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "UserProfile":
        return cls(
            user_id=data["user_id"],
            name=data.get("name", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            research_interests=[
                ResearchInterest.from_dict(ri) for ri in data.get("research_interests", [])
            ],
            initial_papers=data.get("initial_papers", []),
            paper_interactions=[
                PaperInteraction.from_dict(pi) for pi in data.get("paper_interactions", [])
            ],
            keyword_preferences=data.get("keyword_preferences", {}),
            report_history=data.get("report_history", []),
            conversation_summaries=data.get("conversation_summaries", []),
        )
    
    def get_top_interests(self, n: int = 5) -> List[str]:
        """获取权重最高的 N 个研究兴趣分类"""
        sorted_interests = sorted(
            self.research_interests, 
            key=lambda x: x.weight, 
            reverse=True
        )
        return [ri.category for ri in sorted_interests[:n]]
    
    def get_all_keywords(self) -> List[str]:
        """获取所有关键词"""
        keywords = set()
        for ri in self.research_interests:
            keywords.update(ri.keywords)
        return list(keywords)


class UserProfileManager:
    """用户画像管理器"""
    
    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir is None:
            # 从 src/ultrarag/paperagent/ 向上 4 级到项目根目录
            storage_dir = Path(__file__).parent.parent.parent.parent / "data" / "user_profiles"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, UserProfile] = {}
    
    def _get_profile_path(self, user_id: str) -> Path:
        """获取用户画像文件路径"""
        # 使用 hash 避免特殊字符问题
        safe_id = hashlib.md5(user_id.encode()).hexdigest()[:16]
        return self.storage_dir / f"{safe_id}.json"
    
    def create_profile(self, user_id: str, name: str = "") -> UserProfile:
        """创建新用户画像"""
        now = datetime.now().isoformat()
        profile = UserProfile(
            user_id=user_id,
            name=name,
            created_at=now,
            updated_at=now,
        )
        self.save_profile(profile)
        return profile
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        if user_id in self._cache:
            return self._cache[user_id]
        
        path = self._get_profile_path(user_id)
        if not path.exists():
            return None
        
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            profile = UserProfile.from_dict(data)
            self._cache[user_id] = profile
            return profile
        except Exception as e:
            logger.error(f"Failed to load profile for {user_id}: {e}")
            return None
    
    def get_or_create_profile(self, user_id: str, name: str = "") -> UserProfile:
        """获取或创建用户画像"""
        profile = self.get_profile(user_id)
        if profile is None:
            profile = self.create_profile(user_id, name)
        return profile
    
    def save_profile(self, profile: UserProfile):
        """保存用户画像"""
        profile.updated_at = datetime.now().isoformat()
        path = self._get_profile_path(profile.user_id)
        path.write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        self._cache[profile.user_id] = profile
    
    def delete_profile(self, user_id: str) -> bool:
        """删除用户画像"""
        path = self._get_profile_path(user_id)
        if path.exists():
            path.unlink()
            if user_id in self._cache:
                del self._cache[user_id]
            return True
        return False
    
    def list_profiles(self) -> List[str]:
        """列出所有用户 ID"""
        profiles = []
        for path in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                profiles.append(data["user_id"])
            except:
                continue
        return profiles
    
    # ========== 画像更新方法 ==========
    
    def set_initial_interests(
        self, 
        user_id: str, 
        categories: List[str],
        paper_ids: List[str] = None
    ) -> UserProfile:
        """设置用户初始研究兴趣（onboarding）"""
        profile = self.get_or_create_profile(user_id)
        now = datetime.now().isoformat()
        
        # 设置研究兴趣
        profile.research_interests = [
            ResearchInterest(
                category=cat,
                weight=1.0,
                keywords=[],
                last_updated=now
            )
            for cat in categories
        ]
        
        # 记录初始选择的论文
        if paper_ids:
            profile.initial_papers = paper_ids
            for pid in paper_ids:
                profile.paper_interactions.append(
                    PaperInteraction(
                        arxiv_id=pid,
                        title="",  # 后续可补充
                        categories=[],
                        interaction_type="selected",
                        timestamp=now
                    )
                )
        
        self.save_profile(profile)
        return profile
    
    def update_from_conversation(
        self, 
        user_id: str, 
        conversation_summary: str,
        extracted_keywords: List[str] = None,
        mentioned_categories: List[str] = None
    ) -> UserProfile:
        """从对话中更新用户画像"""
        profile = self.get_or_create_profile(user_id)
        now = datetime.now().isoformat()
        
        # 添加对话摘要
        profile.conversation_summaries.append({
            "summary": conversation_summary,
            "timestamp": now,
            "keywords": extracted_keywords or [],
        })
        
        # 限制历史长度
        if len(profile.conversation_summaries) > 20:
            profile.conversation_summaries = profile.conversation_summaries[-20:]
        
        # 更新关键词偏好
        if extracted_keywords:
            for kw in extracted_keywords:
                kw_lower = kw.lower()
                profile.keyword_preferences[kw_lower] = (
                    profile.keyword_preferences.get(kw_lower, 0) + 0.1
                )
        
        # 更新分类权重
        if mentioned_categories:
            for cat in mentioned_categories:
                found = False
                for ri in profile.research_interests:
                    if ri.category == cat:
                        ri.weight = min(1.0, ri.weight + 0.1)
                        ri.last_updated = now
                        found = True
                        break
                if not found:
                    profile.research_interests.append(
                        ResearchInterest(
                            category=cat,
                            weight=0.5,
                            keywords=[],
                            last_updated=now
                        )
                    )
        
        self.save_profile(profile)
        return profile
    
    def record_paper_interaction(
        self,
        user_id: str,
        arxiv_id: str,
        title: str,
        categories: List[str],
        interaction_type: str
    ) -> UserProfile:
        """记录论文交互"""
        profile = self.get_or_create_profile(user_id)
        now = datetime.now().isoformat()
        
        profile.paper_interactions.append(
            PaperInteraction(
                arxiv_id=arxiv_id,
                title=title,
                categories=categories,
                interaction_type=interaction_type,
                timestamp=now
            )
        )
        
        # 限制历史长度
        if len(profile.paper_interactions) > 100:
            profile.paper_interactions = profile.paper_interactions[-100:]
        
        # 根据交互更新兴趣权重
        weight_delta = {
            "selected": 0.2,
            "read": 0.1,
            "saved": 0.15,
            "cited": 0.25,
        }.get(interaction_type, 0.05)
        
        for cat in categories:
            found = False
            for ri in profile.research_interests:
                if ri.category == cat:
                    ri.weight = min(1.0, ri.weight + weight_delta)
                    ri.last_updated = now
                    found = True
                    break
            if not found:
                profile.research_interests.append(
                    ResearchInterest(
                        category=cat,
                        weight=weight_delta,
                        keywords=[],
                        last_updated=now
                    )
                )
        
        self.save_profile(profile)
        return profile
    
    def add_report_to_history(
        self,
        user_id: str,
        report_id: str,
        title: str,
        topics: List[str],
        summary: str
    ) -> UserProfile:
        """添加报告到历史"""
        profile = self.get_or_create_profile(user_id)
        now = datetime.now().isoformat()
        
        profile.report_history.append({
            "report_id": report_id,
            "title": title,
            "topics": topics,
            "summary": summary,
            "created_at": now,
        })
        
        # 限制历史长度
        if len(profile.report_history) > 50:
            profile.report_history = profile.report_history[-50:]
        
        self.save_profile(profile)
        return profile
    
    def get_recommendation_context(self, user_id: str) -> Dict:
        """获取用于推荐的上下文信息"""
        profile = self.get_profile(user_id)
        if not profile:
            return {}
        
        return {
            "top_categories": profile.get_top_interests(5),
            "all_keywords": profile.get_all_keywords(),
            "keyword_preferences": dict(
                sorted(
                    profile.keyword_preferences.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]
            ),
            "recent_papers": [
                pi.arxiv_id for pi in profile.paper_interactions[-10:]
            ],
            "conversation_context": [
                cs["summary"] for cs in profile.conversation_summaries[-5:]
            ],
        }


# 单例
_profile_manager: Optional[UserProfileManager] = None

def get_profile_manager() -> UserProfileManager:
    """获取用户画像管理器单例"""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = UserProfileManager()
    return _profile_manager

