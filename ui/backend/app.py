from __future__ import annotations

import logging
import os
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

    return app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(host="0.0.0.0", port=5050, debug=True)