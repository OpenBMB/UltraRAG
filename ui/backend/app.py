from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context

from . import pipeline_manager as pm

import uuid
import threading
from datetime import datetime

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
EXAMPLES_DIR = BASE_DIR.parent.parent / "examples"
KB_TASKS = {}

def _run_kb_background(task_id, pipeline_name, target_file, output_dir, collection_name, index_mode, chunk_params=None, embedding_params=None):
    LOGGER.info(f"Task {task_id} started: {pipeline_name}")
    try:
        result = pm.run_kb_pipeline_tool(
            pipeline_name=pipeline_name,
            target_file_path=target_file,
            output_dir=output_dir,
            collection_name=collection_name,
            index_mode=index_mode,
            chunk_params=chunk_params,
            embedding_params=embedding_params,
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

    # 直接访问 /chat 或 /config 也返回前端入口，便于分离配置/对话页
    @app.route("/chat")
    def chat_page():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/config")
    def config_page():
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

    @app.route("/api/pipelines/<string:name>/yaml", methods=["PUT"])
    def save_pipeline_yaml(name: str):
        """直接保存 YAML 文本到文件"""
        yaml_content = request.get_data(as_text=True)
        return jsonify(pm.save_pipeline_yaml(name, yaml_content))

    @app.route("/api/pipelines/parse", methods=["POST"])
    def parse_pipeline_yaml():
        """解析任意 YAML 文本，返回结构化数据（供前端校验/同步画布）"""
        yaml_content = request.get_data(as_text=True)
        parsed = pm.parse_pipeline_yaml_content(yaml_content)

        # 同时返回原始文本，便于前端保持一致
        if isinstance(parsed, dict):
            parsed["_raw_yaml"] = yaml_content
        return jsonify(parsed)

    @app.route("/api/pipelines/<string:name>", methods=["GET"])
    def get_pipeline(name: str):
        return jsonify(pm.load_pipeline(name))

    @app.route("/api/pipelines/<string:name>", methods=["DELETE"])
    def delete_pipeline(name: str):
        pm.delete_pipeline(name)
        return jsonify({"status": "deleted"})

    @app.route("/api/pipelines/<string:name>/rename", methods=["POST"])
    def rename_pipeline(name: str):
        """Rename a pipeline"""
        payload = request.get_json(force=True)
        new_name = payload.get("new_name", "").strip()
        if not new_name:
            return jsonify({"error": "new_name is required"}), 400
        
        result = pm.rename_pipeline(name, new_name)
        return jsonify(result)

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
        
        # 新增：前端传入的对话历史（浏览器会话中之前的对话）
        # 兼容两种字段名：conversation_history 或 history
        raw_history = payload.get("conversation_history") or payload.get("history") or []
        
        # 转换格式：前端用 {role, text}，后端用 {role, content}
        conversation_history = []
        for msg in raw_history:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                # 兼容 text 和 content 两种字段名
                content = msg.get("content") or msg.get("text") or ""
                if role in ("user", "assistant") and content:
                    conversation_history.append({"role": role, "content": str(content)})
        
        # 新增：允许前端强制指定是否走完整 pipeline
        force_full_pipeline = payload.get("force_full_pipeline", False)

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
        
        # 根据前端传入的对话历史判断是否是第一次提问
        # 注意：前端在发送请求之前已经把当前问题添加到 history 中了
        # 所以：第一次提问时 history 有 1 条消息，第二次提问时 history 有 3 条消息
        # 只有当有 >= 2 条消息时（说明有之前的助手回复），才走多轮对话
        has_previous_conversation = len(conversation_history) >= 2
        
        LOGGER.info(f"Session {session_id}: conversation_history length = {len(conversation_history)}, has_previous = {has_previous_conversation}")
        
        if not has_previous_conversation or force_full_pipeline:
            # 第一次提问（或强制）：走完整 pipeline
            LOGGER.info(f"Session {session_id}: First turn (history empty), running full pipeline '{name}'")
            return Response(
                pm.chat_demo_stream(name, question, session_id, dynamic_params),
                mimetype='text/event-stream'
            )
        else:
            # 后续提问：走多轮对话，使用前端传入的对话历史
            LOGGER.info(f"Session {session_id}: Multi-turn chat mode with {len(conversation_history)} history messages")
            return Response(
                pm.chat_multiturn_stream(session_id, question, dynamic_params, conversation_history),
                mimetype='text/event-stream'
            )
    
    @app.route("/api/pipelines/chat/stop", methods=["POST"])
    def stop_chat_generation():
        payload = request.get_json(force=True) or {}
        session_id = payload.get("session_id")
        if not session_id:
            return jsonify({"error": "session_id required"}), 400
        return jsonify(pm.interrupt_chat(session_id))
    
    @app.route("/api/pipelines/chat/clear-history", methods=["POST"])
    def clear_chat_history():
        """清除会话的对话历史，使下一次提问重新走完整 pipeline"""
        payload = request.get_json(force=True) or {}
        session_id = payload.get("session_id")
        if not session_id:
            return jsonify({"error": "session_id required"}), 400
        
        session = pm.SESSION_MANAGER.get(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        session.clear_history()
        return jsonify({
            "status": "cleared",
            "session_id": session_id,
            "message": "Conversation history cleared. Next question will run full pipeline."
        })
    
    @app.route("/api/pipelines/chat/history", methods=["GET"])
    def get_chat_history():
        """获取当前会话的对话历史"""
        session_id = request.args.get("session_id")
        if not session_id:
            return jsonify({"error": "session_id required"}), 400
        
        session = pm.SESSION_MANAGER.get(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        history = session.get_conversation_history()
        return jsonify({
            "session_id": session_id,
            "history": history,
            "is_first_turn": session.is_first_turn(),
            "message_count": len(history)
        })

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

        # [新增] Embedding 参数 (用于 milvus_index)
        embedding_params = {
            "api_key": payload.get("emb_api_key", ""),
            "base_url": payload.get("emb_base_url", "https://api.openai.com/v1"),
            "model_name": payload.get("emb_model_name", "text-embedding-3-small")
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
            args=(task_id, pipeline_name, target_file, output_dir, collection_name, index_mode, chunk_params, embedding_params),
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

    # =========================================
    # Prompt Template API
    # =========================================
    
    PROMPTS_DIR = BASE_DIR.parent.parent / "prompt"
    
    @app.route("/api/prompts", methods=["GET"])
    def list_prompts():
        """List all prompt template files"""
        prompts = []
        if PROMPTS_DIR.exists():
            for f in sorted(PROMPTS_DIR.rglob("*.jinja*")):
                rel_path = f.relative_to(PROMPTS_DIR)
                prompts.append({
                    "name": f.name,
                    "path": str(rel_path),
                    "size": f.stat().st_size
                })
        return jsonify(prompts)
    
    @app.route("/api/prompts/<path:filepath>", methods=["GET"])
    def get_prompt(filepath: str):
        """Read a prompt template file"""
        file_path = PROMPTS_DIR / filepath
        if not file_path.exists():
            return jsonify({"error": "File not found"}), 404
        if not str(file_path.resolve()).startswith(str(PROMPTS_DIR.resolve())):
            return jsonify({"error": "Invalid path"}), 400
        
        content = file_path.read_text(encoding="utf-8")
        return jsonify({
            "path": filepath,
            "content": content
        })
    
    @app.route("/api/prompts", methods=["POST"])
    def create_prompt():
        """Create a new prompt template file"""
        payload = request.get_json(force=True)
        name = payload.get("name", "").strip()
        content = payload.get("content", "")
        
        if not name:
            return jsonify({"error": "Name is required"}), 400
        
        # 安全检查：确保文件名合法
        if ".." in name or name.startswith("/"):
            return jsonify({"error": "Invalid filename"}), 400
        
        file_path = PROMPTS_DIR / name
        if file_path.exists():
            return jsonify({"error": "File already exists"}), 409
        
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        
        return jsonify({
            "status": "created",
            "path": name
        })
    
    @app.route("/api/prompts/<path:filepath>", methods=["PUT"])
    def update_prompt(filepath: str):
        """Update a prompt template file"""
        payload = request.get_json(force=True)
        content = payload.get("content", "")
        
        file_path = PROMPTS_DIR / filepath
        if not str(file_path.resolve()).startswith(str(PROMPTS_DIR.resolve())):
            return jsonify({"error": "Invalid path"}), 400
        
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        
        return jsonify({
            "status": "saved",
            "path": filepath
        })
    
    @app.route("/api/prompts/<path:filepath>", methods=["DELETE"])
    def delete_prompt(filepath: str):
        """Delete a prompt template file"""
        file_path = PROMPTS_DIR / filepath
        if not file_path.exists():
            return jsonify({"error": "File not found"}), 404
        if not str(file_path.resolve()).startswith(str(PROMPTS_DIR.resolve())):
            return jsonify({"error": "Invalid path"}), 400
        
        file_path.unlink()
        return jsonify({"status": "deleted"})

    @app.route("/api/prompts/<path:filepath>/rename", methods=["POST"])
    def rename_prompt(filepath: str):
        """Rename a prompt template file"""
        payload = request.get_json(force=True)
        new_name = payload.get("new_name", "").strip()
        
        if not new_name:
            return jsonify({"error": "new_name is required"}), 400
        
        if ".." in new_name or new_name.startswith("/"):
            return jsonify({"error": "Invalid filename"}), 400
        
        old_path = PROMPTS_DIR / filepath
        if not old_path.exists():
            return jsonify({"error": "File not found"}), 404
        if not str(old_path.resolve()).startswith(str(PROMPTS_DIR.resolve())):
            return jsonify({"error": "Invalid path"}), 400
        
        new_path = PROMPTS_DIR / new_name
        if new_path.exists():
            return jsonify({"error": "A file with this name already exists"}), 409
        
        # 确保目标目录存在
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_path)
        
        return jsonify({
            "status": "renamed",
            "old_path": filepath,
            "new_path": new_name
        })

    # =========================================
    # AI Assistant API
    # =========================================
    
    @app.route("/api/ai/test", methods=["POST"])
    def test_ai_connection():
        """Test AI API connection"""
        import requests
        
        payload = request.get_json(force=True)
        provider = payload.get("provider", "openai")
        base_url = payload.get("baseUrl", "").rstrip("/")
        api_key = payload.get("apiKey", "")
        model = payload.get("model", "gpt-4")
        
        if not api_key:
            return jsonify({"success": False, "error": "API key is required"})
        
        try:
            if provider == "openai" or provider == "custom":
                # OpenAI-compatible API
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # Try to list models or send a simple request
                test_url = f"{base_url}/models"
                resp = requests.get(test_url, headers=headers, timeout=10)
                
                if resp.status_code == 200:
                    return jsonify({"success": True, "model": model})
                else:
                    # Try a simple completion as fallback
                    chat_url = f"{base_url}/chat/completions"
                    test_payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5
                    }
                    resp = requests.post(chat_url, headers=headers, json=test_payload, timeout=15)
                    
                    if resp.status_code == 200:
                        return jsonify({"success": True, "model": model})
                    else:
                        return jsonify({"success": False, "error": f"API returned {resp.status_code}: {resp.text[:200]}"})
                        
            elif provider == "anthropic":
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }
                test_url = f"{base_url}/messages"
                test_payload = {
                    "model": model,
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": "Hi"}]
                }
                resp = requests.post(test_url, headers=headers, json=test_payload, timeout=15)
                
                if resp.status_code == 200:
                    return jsonify({"success": True, "model": model})
                else:
                    return jsonify({"success": False, "error": f"API returned {resp.status_code}: {resp.text[:200]}"})
                    
            elif provider == "azure":
                headers = {
                    "api-key": api_key,
                    "Content-Type": "application/json"
                }
                # Azure uses deployment name in URL
                test_url = f"{base_url}/openai/deployments/{model}/chat/completions?api-version=2024-02-15-preview"
                test_payload = {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5
                }
                resp = requests.post(test_url, headers=headers, json=test_payload, timeout=15)
                
                if resp.status_code == 200:
                    return jsonify({"success": True, "model": model})
                else:
                    return jsonify({"success": False, "error": f"API returned {resp.status_code}: {resp.text[:200]}"})
            else:
                return jsonify({"success": False, "error": f"Unknown provider: {provider}"})
                
        except requests.Timeout:
            return jsonify({"success": False, "error": "Connection timeout"})
        except requests.RequestException as e:
            return jsonify({"success": False, "error": str(e)})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    
    @app.route("/api/ai/chat", methods=["POST"])
    def ai_chat():
        """Handle AI chat request"""
        import requests
        
        payload = request.get_json(force=True)
        settings = payload.get("settings", {})
        messages = payload.get("messages", [])
        context = payload.get("context", {})
        stream_response = payload.get("stream", False)
        
        provider = settings.get("provider", "openai")
        base_url = settings.get("baseUrl", "").rstrip("/")
        api_key = settings.get("apiKey", "")
        model = settings.get("model", "gpt-4")
        
        if not api_key:
            return jsonify({"error": "API key is required"})
        
        # Build system prompt with context
        system_prompt = build_ai_system_prompt(context)
        
        # Prepend system message
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        try:
            if stream_response and (provider == "openai" or provider == "custom"):
                def stream_openai_chat():
                    import requests
                    try:
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        }
                        chat_url = f"{base_url}/chat/completions"
                        chat_payload = {
                            "model": model,
                            "messages": full_messages,
                            "temperature": 0.7,
                            "stream": True
                        }
                        resp = requests.post(chat_url, headers=headers, json=chat_payload, stream=True, timeout=120)
                        
                        if resp.status_code != 200:
                            yield f"data: {json.dumps({'type': 'error', 'message': f'API error: {resp.status_code}'})}\n\n"
                            return
                        
                        full_text = ""
                        for raw_line in resp.iter_lines(decode_unicode=True):
                            if not raw_line:
                                continue
                            
                            line = raw_line.strip()
                            if not line:
                                continue
                            
                            if line.startswith("data:"):
                                line = line[len("data:"):].strip()
                            
                            if line == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(line)
                            except Exception:
                                continue
                            
                            delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                full_text += delta
                                yield f"data: {json.dumps({'type': 'token', 'content': delta, 'is_final': False})}\n\n"
                        
                        actions = parse_ai_actions(full_text, context)
                        yield f"data: {json.dumps({'type': 'final', 'content': full_text, 'actions': actions})}\n\n"
                    except requests.Timeout:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Request timeout'})}\n\n"
                    except Exception as e:
                        LOGGER.error(f"AI chat stream error: {e}")
                        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                
                return Response(stream_with_context(stream_openai_chat()), mimetype="text/event-stream")
            
            if provider == "openai" or provider == "custom":
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                chat_url = f"{base_url}/chat/completions"
                chat_payload = {
                    "model": model,
                    "messages": full_messages,
                    "temperature": 0.7
                }
                resp = requests.post(chat_url, headers=headers, json=chat_payload, timeout=120)
                
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    # Parse for actions
                    actions = parse_ai_actions(content, context)
                    
                    return jsonify({
                        "content": content,
                        "actions": actions
                    })
                else:
                    return jsonify({"error": f"API error: {resp.status_code}"})
                    
            elif provider == "anthropic":
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }
                
                # Convert messages format for Anthropic
                anthropic_messages = []
                for msg in messages:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                chat_url = f"{base_url}/messages"
                chat_payload = {
                    "model": model,
                    "max_tokens": 4096,
                    "system": system_prompt,
                    "messages": anthropic_messages
                }
                resp = requests.post(chat_url, headers=headers, json=chat_payload, timeout=120)
                
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("content", [{}])[0].get("text", "")
                    actions = parse_ai_actions(content, context)
                    return jsonify({"content": content, "actions": actions})
                else:
                    return jsonify({"error": f"API error: {resp.status_code}"})
                    
            elif provider == "azure":
                headers = {
                    "api-key": api_key,
                    "Content-Type": "application/json"
                }
                chat_url = f"{base_url}/openai/deployments/{model}/chat/completions?api-version=2024-02-15-preview"
                chat_payload = {
                    "messages": full_messages,
                    "temperature": 0.7
                }
                resp = requests.post(chat_url, headers=headers, json=chat_payload, timeout=120)
                
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    actions = parse_ai_actions(content, context)
                    return jsonify({"content": content, "actions": actions})
                else:
                    return jsonify({"error": f"API error: {resp.status_code}"})
            else:
                return jsonify({"error": f"Unknown provider: {provider}"})
                
        except requests.Timeout:
            return jsonify({"error": "Request timeout"})
        except Exception as e:
            LOGGER.error(f"AI chat error: {e}")
            return jsonify({"error": str(e)})

    return app


def build_ai_system_prompt(context: Dict) -> str:
    """Build system prompt with current context"""
    base_prompt = """You are an AI assistant for UltraRAG, a RAG (Retrieval-Augmented Generation) pipeline configuration system.

You help users:
1. Build and configure pipelines (YAML format)
2. Set parameters for servers (retriever, generator, reranker, etc.)
3. Edit Jinja2 prompt templates

When suggesting modifications, use the following format to indicate actionable changes:

For Pipeline modifications:
```yaml:pipeline
<your yaml content here>
```

For Prompt modifications:
```jinja:prompt:<filename>
<your jinja content here>
```

For Parameter changes, describe them clearly with the full path like:
"Set `generation.model_name` to `gpt-4`"

Be concise and helpful. Provide complete code when suggesting changes.
"""

    # Add context information
    context_info = []
    
    if context.get("currentMode"):
        context_info.append(f"Current mode: {context['currentMode']}")
    
    if context.get("selectedPipeline"):
        context_info.append(f"Selected pipeline: {context['selectedPipeline']}")
    
    if context.get("pipelineYaml"):
        context_info.append(f"Current pipeline YAML:\n```yaml\n{context['pipelineYaml']}\n```")
    
    if context.get("currentPromptFile"):
        context_info.append(f"Current prompt file: {context['currentPromptFile']}")
        if context.get("promptContent"):
            context_info.append(f"Current prompt content:\n```jinja\n{context['promptContent']}\n```")
    
    if context.get("parameters"):
        import json
        params_str = json.dumps(context["parameters"], indent=2, ensure_ascii=False)
        context_info.append(f"Current parameters:\n```json\n{params_str}\n```")
    
    if context_info:
        base_prompt += "\n\n## Current Context\n" + "\n".join(context_info)
    
    return base_prompt


def parse_ai_actions(content: str, context: Dict) -> list:
    """Parse AI response for actionable modifications"""
    import re
    
    actions = []
    
    # Parse pipeline YAML blocks
    yaml_pattern = r'```yaml:pipeline\s*\n(.*?)```'
    yaml_matches = re.findall(yaml_pattern, content, re.DOTALL)
    for match in yaml_matches:
        actions.append({
            "type": "modify_pipeline",
            "content": match.strip(),
            "preview": match.strip()[:500]
        })
    
    # Parse prompt blocks
    prompt_pattern = r'```jinja:prompt:([^\n]+)\s*\n(.*?)```'
    prompt_matches = re.findall(prompt_pattern, content, re.DOTALL)
    for filename, prompt_content in prompt_matches:
        actions.append({
            "type": "modify_prompt",
            "filename": filename.strip(),
            "content": prompt_content.strip(),
            "preview": prompt_content.strip()[:500]
        })
    
    # Parse parameter changes
    param_pattern = r'[Ss]et\s+`([^`]+)`\s+to\s+`([^`]+)`'
    param_matches = re.findall(param_pattern, content)
    for path, value in param_matches:
        # Try to parse value
        try:
            import json
            parsed_value = json.loads(value)
        except:
            parsed_value = value
        
        actions.append({
            "type": "modify_parameter",
            "path": path.strip(),
            "value": parsed_value,
            "preview": f"{path} = {value}"
        })
    
    return actions

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(host="0.0.0.0", port=5050, debug=True)