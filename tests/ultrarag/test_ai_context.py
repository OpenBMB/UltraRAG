"""Parameter assistance must not disclose configured service credentials."""

import copy
import importlib


def test_assistant_parameter_context_redacts_credentials_without_mutation(monkeypatch):
    module = importlib.import_module("ui.backend.app")
    monkeypatch.setattr(module, "load_llms_doc", lambda: "")
    context = {
        "parameters": {
            "generation": {
                "backend_configs": {
                    "openai": {
                        "api_key": "private-key-marker",
                        "model_name": "test-model",
                    }
                }
            },
            "password": "private-password-marker",
            "headers": {"Authorization": "private-header-marker"},
        }
    }
    original = copy.deepcopy(context)
    prompt = module.build_ai_system_prompt(context)
    assert "private-key-marker" not in prompt
    assert "private-password-marker" not in prompt
    assert "private-header-marker" not in prompt
    assert "test-model" in prompt
    assert context == original
