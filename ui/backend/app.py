from __future__ import annotations

import logging
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory, Response

from . import pipeline_manager as pm

import uuid
import threading
from datetime import datetime

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
EXAMPLES_DIR = BASE_DIR.parent.parent / "examples"
KB_TASKS = {}

def _run_kb_background(task_id, pipeline_name, target_file, output_dir, collection_name, index_mode, chunk_params=None):
    LOGGER.info(f"Task {task_id} started: {pipeline_name}")
    try:
        result = pm.run_kb_pipeline_tool(
            pipeline_name=pipeline_name,
            target_file_path=target_file,
            output_dir=output_dir,
            collection_name=collection_name,
            index_mode=index_mode,
            chunk_params=chunk_params,
        )
        KB_TASKS[task_id]["status"] = "success"
        KB_TASKS[task_id]["result"] = result
        KB_TASKS[task_id]["completed_at"] = datetime.now().isoformat()
        LOGGER.info(f"Task {task_id} completed successfully.")
        
    except Exception as e:
        LOGGER.error(f"Task {task_id} failed: {e}", exc_info=True)
        KB_TASKS[task_id]["status"] = "failed"
        KB_TASKS[task_id]["error"] = str(e)

def create_app(admin_mode: bool = False) -> Flask:
    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
    app.config["ADMIN_MODE"] = admin_mode

    @app.errorhandler(pm.PipelineManagerError)
    def handle_pipeline_error(err: pm.PipelineManagerError):
        LOGGER.error(f"Pipeline error: {err}")
        return jsonify({"error": str(err)}), 400

    @app.errorhandler(Exception)
    def handle_generic_error(err: Exception):
        LOGGER.error(f"System error: {err}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "details": str(err)}), 500

    @app.route("/favicon.svg")
    def favicon():
        return send_from_directory(os.path.join(app.static_folder), "favicon.svg", mimetype="image/svg+xml")

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/config/mode", methods=["GET"])
    def get_app_mode():
        """Return the application mode (admin or chat-only)"""
        return jsonify({
            "admin_mode": app.config.get("ADMIN_MODE", False)
        })

    @app.route("/api/templates", methods=["GET"])
    def list_templates():
        templates = []
        for f in sorted(EXAMPLES_DIR.glob("*.yaml")):
            try:
                templates.append({"name": f.stem, "content": f.read_text(encoding="utf-8")})
            except Exception:
                continue
        return jsonify(templates)

    @app.route("/api/servers", methods=["GET"])
    def servers():
        return jsonify(pm.list_servers())

    @app.route("/api/tools", methods=["GET"])
    def tools():
        return jsonify([
            {
                "id": tool.identifier,
                "server": tool.server,
                "tool": tool.tool,
                "kind": tool.kind,
                "input": tool.input_spec,
                "output": tool.output_spec,
            }
            for tool in pm.list_server_tools()
        ])

    @app.route("/api/pipelines", methods=["GET"])
    def list_pipelines():
        return jsonify(pm.list_pipelines())

    @app.route("/api/pipelines", methods=["POST"])
    def save_pipeline():
        payload = request.get_json(force=True)
        return jsonify(pm.save_pipeline(payload))

    @app.route("/api/pipelines/<string:name>", methods=["GET"])
    def get_pipeline(name: str):
        return jsonify(pm.load_pipeline(name))

    @app.route("/api/pipelines/<string:name>", methods=["DELETE"])
    def delete_pipeline(name: str):
        pm.delete_pipeline(name)
        return jsonify({"status": "deleted"})

    @app.route("/api/pipelines/<string:name>/parameters", methods=["GET"])
    def get_parameters(name: str):
        return jsonify(pm.load_parameters(name))

    @app.route("/api/pipelines/<string:name>/parameters", methods=["PUT"])
    def save_parameters(name: str):
        payload = request.get_json(force=True)
        pm.save_parameters(name, payload)
        return jsonify({"status": "saved"})

    @app.route("/api/pipelines/<string:name>/build", methods=["POST"])
    def build_pipeline(name: str):
        return jsonify(pm.build(name))

    @app.route("/api/pipelines/<string:name>/demo/start", methods=["POST"])
    def start_demo_session(name: str):
        payload = request.get_json(force=True) or {}
        session_id = payload.get("session_id")
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        return jsonify(pm.start_demo_session(name, session_id))

    @app.route("/api/pipelines/demo/stop", methods=["POST"])
    def stop_demo_session():
        payload = request.get_json(force=True) or {}
        session_id = payload.get("session_id")
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        return jsonify(pm.stop_demo_session(session_id))

    @app.route("/api/pipelines/<string:name>/chat", methods=["POST"])
    def chat_pipeline(name: str):
        payload = request.get_json(force=True)
        question = payload.get("question", "")
        session_id = payload.get("session_id")
        dynamic_params = payload.get("dynamic_params", {})

        selected_collection = dynamic_params.get("collection_name")

        try:
            kb_config = pm.load_kb_config()
            milvus_global_config = kb_config.get("milvus", {})
            
            retriever_params = {
                "index_backend": "milvus", 
                "index_backend_configs": {
                    "milvus": milvus_global_config 
                }
            }
            
            if selected_collection:
                retriever_params["collection_name"] = selected_collection
                print(f"debug: Chat using collection override: {selected_collection}")
            
            dynamic_params["retriever"] = retriever_params
            
            if "collection_name" in dynamic_params:
                del dynamic_params["collection_name"]
                
        except Exception as e:
            print(f"Warning: Failed to construct retriever config: {e}")

        if not session_id:
            return jsonify({"error": "session_id missing. Please start engine first."}), 400
        
        return Response(
            pm.chat_demo_stream(name, question, session_id, dynamic_params),
            mimetype='text/event-stream'
        )
    
    @app.route("/api/pipelines/chat/stop", methods=["POST"])
    def stop_chat_generation():
        payload = request.get_json(force=True) or {}
        session_id = payload.get("session_id")
        if not session_id:
            return jsonify({"error": "session_id required"}), 400
        return jsonify(pm.interrupt_chat(session_id))

    # ===== 后台聊天任务 API =====
    
    @app.route("/api/pipelines/<string:name>/chat/background", methods=["POST"])
    def start_background_chat(name: str):
        """启动后台聊天任务"""
        payload = request.get_json(force=True)
        question = payload.get("question", "")
        session_id = payload.get("session_id")
        dynamic_params = payload.get("dynamic_params", {})
        user_id = payload.get("user_id", "") or request.headers.get("X-User-ID", "")
        
        if not question:
            return jsonify({"error": "question is required"}), 400
        if not session_id:
            return jsonify({"error": "session_id is required. Please start engine first."}), 400
        
        # 处理 collection_name
        selected_collection = dynamic_params.get("collection_name")
        try:
            kb_config = pm.load_kb_config()
            milvus_global_config = kb_config.get("milvus", {})
            
            retriever_params = {
                "index_backend": "milvus",
                "index_backend_configs": {
                    "milvus": milvus_global_config
                }
            }
            
            if selected_collection:
                retriever_params["collection_name"] = selected_collection
            
            dynamic_params["retriever"] = retriever_params
            
            if "collection_name" in dynamic_params:
                del dynamic_params["collection_name"]
                
        except Exception as e:
            LOGGER.warning(f"Failed to construct retriever config: {e}")
        
        try:
            task_id = pm.run_background_chat(name, question, session_id, dynamic_params, user_id=user_id)
            return jsonify({
                "status": "started",
                "task_id": task_id,
                "message": "Task started in background"
            }), 202
        except Exception as e:
            LOGGER.error(f"Failed to start background chat: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/background-tasks", methods=["GET"])
    def list_background_tasks():
        """列出当前用户的后台任务"""
        limit = request.args.get("limit", 20, type=int)
        user_id = request.args.get("user_id", "") or request.headers.get("X-User-ID", "")
        LOGGER.info(f"Listing background tasks for user_id: '{user_id}'")
        tasks = pm.list_background_tasks(limit, user_id=user_id)
        LOGGER.info(f"Found {len(tasks)} tasks for user_id: '{user_id}'")
        return jsonify(tasks)
    
    @app.route("/api/background-tasks/<string:task_id>", methods=["GET"])
    def get_background_task(task_id: str):
        """获取单个后台任务状态"""
        user_id = request.args.get("user_id", "") or request.headers.get("X-User-ID", "")
        task = pm.get_background_task(task_id, user_id=user_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        return jsonify(task)
    
    @app.route("/api/background-tasks/<string:task_id>", methods=["DELETE"])
    def delete_background_task(task_id: str):
        """删除后台任务"""
        user_id = request.args.get("user_id", "") or request.headers.get("X-User-ID", "")
        success = pm.delete_background_task(task_id, user_id=user_id)
        if success:
            return jsonify({"status": "deleted", "task_id": task_id})
        return jsonify({"error": "Task not found"}), 404
    
    @app.route("/api/background-tasks/clear-completed", methods=["POST"])
    def clear_completed_tasks():
        """清理当前用户已完成的后台任务"""
        payload = request.get_json(force=True) if request.is_json else {}
        user_id = payload.get("user_id", "") or request.headers.get("X-User-ID", "")
        count = pm.clear_completed_background_tasks(user_id=user_id)
        return jsonify({"status": "cleared", "count": count})

    @app.route("/api/system/shutdown", methods=["POST"])
    def shutdown():
        LOGGER.info("Shutdown requested")
        func = request.environ.get("werkzeug.server.shutdown")
        if func:
            threading.Timer(0.2, func).start()
            return jsonify({"status": "shutting-down", "mode": "graceful"})
        
        threading.Timer(0.5, os._exit, args=(0,)).start()
        return jsonify({"status": "shutting-down", "mode": "force"})

    @app.route("/api/kb/config", methods=["GET"])
    def get_kb_config():
        return jsonify(pm.load_kb_config())
    
    @app.route("/api/kb/config", methods=["POST"])
    def save_kb_config():
        try:
            payload = request.get_json(force=True)
            pm.save_kb_config(payload)
            return jsonify({"status": "saved"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/kb/files", methods=["GET"])
    def list_kb_files():
        return jsonify(pm.list_kb_files())

    @app.route("/api/kb/files/inspect", methods=["GET"])
    def inspect_kb_folder():
        category = request.args.get("category", "raw") #
        folder_name = request.args.get("name")
        
        if not folder_name:
            return jsonify({"error": "Folder name required"}), 400
            
        base_dir = {
            "raw": pm.KB_RAW_DIR,
            "corpus": pm.KB_CORPUS_DIR, 
            "chunks": pm.KB_CHUNKS_DIR
        }.get(category)
        
        target_path = base_dir / folder_name
        
        if not target_path.exists() or not target_path.is_dir():
            return jsonify({"error": "Folder not found"}), 404
            
        files = []
        for f in sorted(target_path.glob("*")):
            if f.is_file() and not f.name.startswith("."):
                files.append({
                    "name": f.name,
                    "size": f.stat().st_size
                })
                
        return jsonify({"files": files})

    @app.route("/api/kb/upload", methods=["POST"])
    def upload_kb_file():
        files = request.files.getlist('file') 
        
        if not files or files[0].filename == '':
            return jsonify({"error": "No selected files"}), 400
            
        try:
            result = pm.upload_kb_files_batch(files)
            return jsonify(result)
        except Exception as e:
            LOGGER.error(f"Upload failed: {e}")
            return jsonify({"error": str(e)}), 500
                
    @app.route("/api/kb/files/<string:category>/<string:filename>", methods=["DELETE"])
    def delete_kb_file(category: str, filename: str):
        if category not in ["raw", "corpus", "chunks", "collection", "index"]:
            return jsonify({"error": "Invalid category"}), 400
        
        try:
            return jsonify(pm.delete_kb_file(category, filename))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/kb/staging/clear", methods=["POST"])
    def clear_staging_area():
        """清空暂存区：删除 raw, corpus, chunks 三个目录中的所有文件"""
        try:
            result = pm.clear_staging_area()
            return jsonify(result)
        except Exception as e:
            LOGGER.error(f"Failed to clear staging area: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/kb/run", methods=["POST"])
    def run_kb_task():
        payload = request.get_json(force=True)
        pipeline_name = payload.get("pipeline_name")
        target_file = payload.get("target_file") 
        
        collection_name = payload.get("collection_name")
        index_mode = payload.get("index_mode", "append")

        chunk_params = {
            "chunk_backend": payload.get("chunk_backend", "token"),
            "tokenizer_or_token_counter": payload.get("tokenizer_or_token_counter", "gpt2"),
            "chunk_size": payload.get("chunk_size", 500),
            "use_title": payload.get("use_title", True)
        }

        if not pipeline_name or not target_file:
            return jsonify({"error": "Missing pipeline_name or target_file"}), 400

        output_dir = ""
        if pipeline_name == "build_text_corpus":
            output_dir = str(pm.KB_CORPUS_DIR)
        elif pipeline_name == "corpus_chunk":
            output_dir = str(pm.KB_CHUNKS_DIR)
        elif pipeline_name == "milvus_index":
            output_dir = "" 
        
        task_id = str(uuid.uuid4())
        KB_TASKS[task_id] = {
            "status": "running",
            "pipeline": pipeline_name,
            "created_at": datetime.now().isoformat()
        }

        thread = threading.Thread(
            target=_run_kb_background,
            args=(task_id, pipeline_name, target_file, output_dir, collection_name, index_mode, chunk_params),
            daemon=True # 设置为守护线程，主程序退出时它也会自动退出，防止挂起
        )
        thread.start()

        return jsonify({
            "status": "submitted",
            "task_id": task_id,
            "message": "Task started in background"
        }), 202

    @app.route("/api/kb/status/<string:task_id>", methods=["GET"])
    def get_kb_task_status(task_id: str):
        task = KB_TASKS.get(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        
        return jsonify(task)

    # ===== PaperAgent API =====
    
    @app.route("/api/paperagent/categories", methods=["GET"])
    def get_cs_categories():
        """获取所有 CS 分类"""
        try:
            from ultrarag.paperagent.cs_categories import get_category_display_list
            categories = get_category_display_list()
            return jsonify({"categories": categories})
        except Exception as e:
            LOGGER.error(f"Failed to get CS categories: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/search", methods=["GET", "POST"])
    def search_papers():
        """搜索论文"""
        try:
            if request.method == "POST":
                payload = request.get_json(force=True)
                query = payload.get("query", "")
                category = payload.get("category", "")
                max_results = payload.get("max_results", 20)
            else:
                query = request.args.get("query", "")
                category = request.args.get("category", "")
                max_results = request.args.get("max_results", 20, type=int)
            
            from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
            km = get_knowledge_manager()
            papers = km.search_papers(
                query=query,
                category=category if category else None,
                max_results=max_results
            )
            
            return jsonify({
                "papers": [p.to_dict() for p in papers],
                "total": len(papers)
            })
        except Exception as e:
            LOGGER.error(f"Failed to search papers: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/onboarding/<string:category>", methods=["GET"])
    def get_onboarding_papers(category: str):
        """获取用于 onboarding 的论文"""
        try:
            count = request.args.get("count", 10, type=int)
            
            from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
            km = get_knowledge_manager()
            papers = km.get_onboarding_papers(category, count)
            
            return jsonify({
                "papers": [p.to_dict() for p in papers],
                "category": category
            })
        except Exception as e:
            LOGGER.error(f"Failed to get onboarding papers: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/profile/<string:user_id>", methods=["GET"])
    def get_user_profile(user_id: str):
        """获取用户画像"""
        try:
            from ultrarag.paperagent.user_profile import get_profile_manager
            pm = get_profile_manager()
            profile = pm.get_profile(user_id)
            
            return jsonify({
                "profile": profile.to_dict() if profile else None
            })
        except Exception as e:
            LOGGER.error(f"Failed to get user profile: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/profile", methods=["POST"])
    def save_user_profile():
        """保存用户画像"""
        try:
            payload = request.get_json(force=True)
            user_id = payload.get("user_id", "")
            interests = payload.get("interests", [])
            selected_papers = payload.get("selected_papers", [])
            
            if not user_id:
                return jsonify({"error": "user_id required"}), 400
            
            from ultrarag.paperagent.user_profile import get_profile_manager
            pm = get_profile_manager()
            
            # 设置初始兴趣
            paper_ids = [p.get("arxiv_id", "") for p in selected_papers if p.get("arxiv_id")]
            profile = pm.set_initial_interests(user_id, interests, paper_ids)
            
            # 记录论文交互
            for paper in selected_papers:
                if paper.get("arxiv_id"):
                    pm.record_paper_interaction(
                        user_id=user_id,
                        arxiv_id=paper.get("arxiv_id"),
                        title=paper.get("title", ""),
                        categories=paper.get("categories", []),
                        interaction_type="selected"
                    )
            
            return jsonify({
                "status": "success",
                "profile": profile.to_dict() if profile else None
            })
        except Exception as e:
            LOGGER.error(f"Failed to save user profile: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/profile/interests", methods=["POST"])
    def set_user_interests():
        """设置用户研究兴趣"""
        try:
            payload = request.get_json(force=True)
            user_id = payload.get("user_id", "") or request.headers.get("X-User-ID", "")
            categories = payload.get("categories", [])
            paper_ids = payload.get("paper_ids", [])
            
            if not user_id:
                return jsonify({"error": "user_id required"}), 400
            if not categories:
                return jsonify({"error": "categories required"}), 400
            
            from ultrarag.paperagent.user_profile import get_profile_manager
            pm = get_profile_manager()
            profile = pm.set_initial_interests(user_id, categories, paper_ids)
            
            return jsonify({
                "status": "success",
                "profile": profile.to_dict()
            })
        except Exception as e:
            LOGGER.error(f"Failed to set user interests: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/profile/<string:user_id>", methods=["DELETE"])
    def delete_user_profile(user_id: str):
        """删除/清空用户画像"""
        try:
            from ultrarag.paperagent.user_profile import get_profile_manager
            pm = get_profile_manager()
            success = pm.delete_profile(user_id)
            
            if success:
                return jsonify({"status": "deleted", "user_id": user_id})
            return jsonify({"error": "Profile not found"}), 404
        except Exception as e:
            LOGGER.error(f"Failed to delete user profile: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/recommendations", methods=["GET"])
    def get_recommended_papers():
        """获取推荐论文"""
        try:
            user_id = request.args.get("user_id", "") or request.headers.get("X-User-ID", "")
            max_results = request.args.get("max_results", 20, type=int)
            
            if not user_id:
                return jsonify({"error": "user_id required"}), 400
            
            from ultrarag.paperagent.user_profile import get_profile_manager
            from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
            
            pm = get_profile_manager()
            km = get_knowledge_manager()
            
            profile = pm.get_profile(user_id)
            if not profile:
                return jsonify({"papers": [], "message": "Profile not found"})
            
            interests = profile.get_top_interests(5)
            keywords = list(profile.keyword_preferences.keys())[:10]
            exclude_ids = [pi.arxiv_id for pi in profile.paper_interactions]
            
            papers = km.get_recommended_papers(
                interests=interests,
                keywords=keywords,
                exclude_ids=exclude_ids,
                max_results=max_results
            )
            
            return jsonify({
                "papers": [p.to_dict() for p in papers],
                "based_on": {
                    "interests": interests,
                    "keywords": keywords[:5]
                }
            })
        except Exception as e:
            LOGGER.error(f"Failed to get recommendations: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/recommend", methods=["GET"])
    def get_recommend_papers():
        """获取推荐论文（用于 onboarding）"""
        try:
            categories = request.args.get("categories", "")
            max_results = request.args.get("max_results", 20, type=int)
            refresh = request.args.get("refresh", "false").lower() == "true"
            
            from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
            km = get_knowledge_manager()
            
            category_list = [c.strip() for c in categories.split(",") if c.strip()] if categories else []
            
            papers = km.get_onboarding_papers(
                categories=category_list,
                max_results=max_results,
                refresh=refresh
            )
            
            return jsonify({
                "papers": [p.to_dict() for p in papers],
                "total": len(papers),
                "refreshed": refresh
            })
        except Exception as e:
            LOGGER.error(f"Failed to get recommend papers: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/reports", methods=["GET"])
    def list_reports():
        """列出用户的报告"""
        try:
            user_id = request.args.get("user_id", "") or request.headers.get("X-User-ID", "")
            status = request.args.get("status", "")
            limit = request.args.get("limit", 50, type=int)
            
            from ultrarag.paperagent.report_manager import get_report_manager
            rm = get_report_manager()
            reports = rm.list_reports(
                user_id=user_id if user_id else None,
                status=status if status else None,
                limit=limit
            )
            
            return jsonify({"reports": reports})
        except Exception as e:
            LOGGER.error(f"Failed to list reports: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/reports/<string:user_id>", methods=["GET"])
    def list_user_reports(user_id: str):
        """列出指定用户的报告（同时包含 'default' 用户的报告）"""
        try:
            limit = request.args.get("limit", 50, type=int)
            
            from ultrarag.paperagent.report_manager import get_report_manager
            rm = get_report_manager()
            
            # 获取用户自己的报告
            user_reports = rm.list_reports(user_id=user_id, limit=limit)
            
            # 如果用户不是 default，也获取 default 用户的报告
            if user_id != "default":
                default_reports = rm.list_reports(user_id="default", limit=limit)
                # 合并并去重
                seen_ids = {r.get("report_id") for r in user_reports}
                for r in default_reports:
                    if r.get("report_id") not in seen_ids:
                        user_reports.append(r)
            
            # 按创建时间倒序排列
            user_reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            return jsonify({"reports": user_reports[:limit]})
        except Exception as e:
            LOGGER.error(f"Failed to list user reports: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/generate-report", methods=["POST"])
    def generate_report():
        """生成研究报告 - 使用 paper_research pipeline 后台执行"""
        try:
            payload = request.get_json(force=True)
            user_id = payload.get("user_id", "")
            topic = payload.get("topic", "")
            report_type = payload.get("report_type", "hotspot")
            
            if not topic:
                return jsonify({"error": "topic required"}), 400
            
            # 检查 paper_research pipeline 是否存在
            pipeline_name = "paper_research"
            pipelines = pm.list_pipelines()
            pipeline_exists = any(p["name"] == pipeline_name for p in pipelines)
            
            if not pipeline_exists:
                return jsonify({"error": f"Pipeline '{pipeline_name}' not found. Please create it first."}), 404
            
            # 为后台任务创建独立的 session ID
            import uuid as uuid_mod
            session_id = f"pa_{pipeline_name}_{int(time.time()*1000)}_{str(uuid_mod.uuid4())[:8]}"
            
            # 构建增强的问题（包含报告类型信息）
            type_labels = {
                "hotspot": "热点分析报告",
                "idea": "新 Idea 挖掘报告", 
                "survey": "综述概要报告"
            }
            enhanced_topic = f"请生成关于「{topic}」的{type_labels.get(report_type, '研究报告')}。"
            
            # 动态参数传递用户信息
            dynamic_params = {
                "user_id": user_id,
                "report_type": report_type,
                "original_topic": topic
            }
            
            # 使用后台任务执行 pipeline
            task_id = pm.run_background_chat(
                name=pipeline_name,
                question=enhanced_topic,
                session_id=session_id,
                dynamic_params=dynamic_params,
                user_id=user_id
            )
            
            LOGGER.info(f"Started paper_research pipeline for topic: {topic}, task_id: {task_id}")
            
            return jsonify({
                "status": "started",
                "task_id": task_id,
                "session_id": session_id,
                "message": "Report generation started in background"
            }), 202
            
        except Exception as e:
            LOGGER.error(f"Failed to start report generation: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/quick-report", methods=["POST"])
    def generate_quick_report():
        """快速生成简单报告（不使用 pipeline，直接搜索 + 格式化）"""
        try:
            payload = request.get_json(force=True)
            user_id = payload.get("user_id", "")
            topic = payload.get("topic", "")
            report_type = payload.get("report_type", "hotspot")
            
            if not topic:
                return jsonify({"error": "topic required"}), 400
            
            from ultrarag.paperagent.report_manager import get_report_manager
            from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
            
            rm = get_report_manager()
            km = get_knowledge_manager()
            
            # 搜索相关论文
            papers = km.search_papers(query=topic, max_results=20)
            
            # 根据报告类型生成不同内容
            type_labels = {
                "hotspot": "热点分析",
                "idea": "新 Idea 挖掘",
                "survey": "综述概要"
            }
            
            # 生成报告内容
            content = f"""# {topic} - {type_labels.get(report_type, '研究报告')}

## 概述

本报告基于对 arXiv 最新论文的分析，围绕「{topic}」主题进行研究热点和趋势分析。

## 相关论文

"""
            for i, paper in enumerate(papers[:10], 1):
                content += f"### [{i}] {paper.title}\n\n"
                content += f"- **arXiv ID**: {paper.arxiv_id}\n"
                content += f"- **作者**: {', '.join(paper.authors[:3])}\n"
                content += f"- **分类**: {', '.join(paper.categories[:3])}\n"
                content += f"- **发布日期**: {paper.published}\n\n"
                content += f"{paper.abstract[:500]}...\n\n"
            
            content += """
## 总结

基于以上论文分析，该领域目前的研究热点主要集中在...

---
*本报告由 PaperAgent 自动生成*
"""
            
            # 保存报告
            report = rm.create_report(
                user_id=user_id or "anonymous",
                title=f"{topic} - {type_labels.get(report_type, '研究报告')}",
                content=content,
                topics=[topic],
                categories=[p.primary_category for p in papers[:5] if p.primary_category],
                keywords=topic.split(),
                cited_papers=[{
                    "arxiv_id": p.arxiv_id,
                    "title": p.title
                } for p in papers[:10]],
                report_type=report_type
            )
            
            return jsonify({
                "status": "success",
                "report_id": report.report_id,
                "title": report.title
            })
        except Exception as e:
            LOGGER.error(f"Failed to generate quick report: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/reports/<string:report_id>", methods=["GET"])
    def get_report(report_id: str):
        """获取单个报告"""
        try:
            from ultrarag.paperagent.report_manager import get_report_manager
            rm = get_report_manager()
            report = rm.get_report(report_id)
            
            if not report:
                return jsonify({"error": "Report not found"}), 404
            
            return jsonify({"report": report.to_dict()})
        except Exception as e:
            LOGGER.error(f"Failed to get report: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/reports/<string:report_id>", methods=["DELETE"])
    def delete_report(report_id: str):
        """删除报告"""
        try:
            from ultrarag.paperagent.report_manager import get_report_manager
            rm = get_report_manager()
            success = rm.delete_report(report_id)
            
            if success:
                return jsonify({"status": "deleted"})
            return jsonify({"error": "Report not found"}), 404
        except Exception as e:
            LOGGER.error(f"Failed to delete report: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/reports/<string:report_id>/export", methods=["GET"])
    def export_report(report_id: str):
        """导出报告为 Markdown"""
        try:
            from ultrarag.paperagent.report_manager import get_report_manager
            rm = get_report_manager()
            markdown = rm.export_report_markdown(report_id)
            
            if not markdown:
                return jsonify({"error": "Report not found"}), 404
            
            return jsonify({"markdown": markdown, "report_id": report_id})
        except Exception as e:
            LOGGER.error(f"Failed to export report: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/report/<string:report_id>", methods=["GET"])
    def get_single_report(report_id: str):
        """获取单个报告（别名）"""
        try:
            from ultrarag.paperagent.report_manager import get_report_manager
            rm = get_report_manager()
            report = rm.get_report(report_id)
            
            if not report:
                return jsonify({"error": "Report not found"}), 404
            
            return jsonify({"report": report.to_dict()})
        except Exception as e:
            LOGGER.error(f"Failed to get report: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/report/<string:report_id>/download", methods=["GET"])
    def download_report(report_id: str):
        """下载报告为 Markdown 文件"""
        try:
            from flask import Response
            from ultrarag.paperagent.report_manager import get_report_manager
            rm = get_report_manager()
            report = rm.get_report(report_id)
            
            if not report:
                return jsonify({"error": "Report not found"}), 404
            
            markdown = rm.export_report_markdown(report_id)
            
            response = Response(
                markdown,
                mimetype="text/markdown",
                headers={
                    "Content-Disposition": f"attachment; filename={report.title}.md"
                }
            )
            return response
        except Exception as e:
            LOGGER.error(f"Failed to download report: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/corpus/build", methods=["POST"])
    def build_paper_corpus():
        """构建论文语料库"""
        try:
            payload = request.get_json(force=True)
            categories = payload.get("categories", [])
            query = payload.get("query", "")
            max_papers = payload.get("max_papers", 500)
            corpus_name = payload.get("corpus_name", "default")
            
            from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
            km = get_knowledge_manager()
            path = km.build_corpus(
                categories=categories,
                query=query,
                max_papers=max_papers,
                corpus_name=corpus_name
            )
            
            return jsonify({
                "status": "success",
                "corpus_path": str(path),
                "corpus_name": corpus_name
            })
        except Exception as e:
            LOGGER.error(f"Failed to build corpus: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/corpus", methods=["GET"])
    def list_paper_corpora():
        """列出论文语料库"""
        try:
            from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
            km = get_knowledge_manager()
            corpora = km.list_corpora()
            
            return jsonify({"corpora": corpora})
        except Exception as e:
            LOGGER.error(f"Failed to list corpora: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/paperagent/stats", methods=["GET"])
    def get_paperagent_stats():
        """获取 PaperAgent 统计信息"""
        try:
            from ultrarag.paperagent.knowledge_manager import get_knowledge_manager
            km = get_knowledge_manager()
            stats = km.get_stats()
            
            return jsonify({"stats": stats})
        except Exception as e:
            LOGGER.error(f"Failed to get stats: {e}")
            return jsonify({"error": str(e)}), 500

    return app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(host="0.0.0.0", port=5050, debug=True)