import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

NGROK_BASE: Final = "https://gateway.ngrok.ai/v1"


def test_ngrok_provider_resolution_keeps_catalog_model_id(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("NGROK_AI_GATEWAY_KEY", "ng-test-key")
    monkeypatch.delenv("NGROK_AI_GATEWAY_BASE_URL", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="ngrok/anthropic/claude-sonnet-5-5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "anthropic/claude-sonnet-5-5"
    assert provider == "ngrok"
    assert api_key == "ng-test-key"
    assert api_base == NGROK_BASE


def test_ngrok_provider_reads_base_url_env(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("NGROK_AI_GATEWAY_KEY", "ng-test-key")
    monkeypatch.setenv("NGROK_AI_GATEWAY_BASE_URL", "https://gateway.internal.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="ngrok/openai/gpt-5.5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "ngrok"
    assert api_base == "https://gateway.internal.example/v1"


def test_ngrok_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("NGROK_AI_GATEWAY_KEY", "ng-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="ngrok/openai/gpt-5.5",
        custom_llm_provider=None,
        api_base="https://gateway.internal.example/v1",
        api_key="ng-explicit-key",
    )

    assert provider == "ngrok"
    assert api_key == "ng-explicit-key"
    assert api_base == "https://gateway.internal.example/v1"


def test_ngrok_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    ngrok = next(provider for provider in providers if provider["litellm_provider"] == "ngrok")

    assert ngrok["provider"] == "NGROK"
    assert ngrok["provider_display_name"] == "ngrok AI Gateway"
    assert ngrok["default_model_placeholder"] == "ngrok/anthropic/claude-sonnet-5-5"
    assert {field["key"]: field["required"] for field in ngrok["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_ngrok_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    endpoints = json.loads(matrix_path.read_text())["providers"]["ngrok"]["endpoints"]

    assert {name for name, supported in endpoints.items() if supported} == {
        "chat_completions",
        "responses",
        "messages",
        "embeddings",
    }


def test_ngrok_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{NGROK_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_ngrok",
                "object": "chat.completion",
                "created": 1_794_500_000,
                "model": "anthropic/claude-sonnet-5-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from ngrok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="ngrok/anthropic/claude-sonnet-5-5",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="ng-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{NGROK_BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer ng-test-key"
    assert body["model"] == "anthropic/claude-sonnet-5-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from ngrok"


def test_ngrok_pinned_provider_model_id_passes_through():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{NGROK_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_ngrok_pinned",
                "object": "chat.completion",
                "created": 1_794_500_000,
                "model": "openai:gpt-5.5",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
        litellm.completion(
            model="ngrok/openai:gpt-5.5",
            messages=[{"role": "user", "content": "ping"}],
            api_key="ng-test-key",
        )

    assert json.loads(route.calls.last.request.content)["model"] == "openai:gpt-5.5"


def test_ngrok_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{NGROK_BASE}/responses").respond(
            200,
            json={
                "id": "resp_ngrok",
                "object": "response",
                "created_at": 1_794_500_000,
                "model": "openai/gpt-5.5",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_ngrok",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from ngrok", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="ngrok/openai/gpt-5.5",
            input="Say hello",
            api_key="ng-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{NGROK_BASE}/responses"
    assert request.headers["authorization"] == "Bearer ng-test-key"
    assert body["model"] == "openai/gpt-5.5"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from ngrok"


@pytest.mark.asyncio
async def test_ngrok_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{NGROK_BASE}/messages").respond(
            200,
            json={
                "id": "msg_ngrok",
                "type": "message",
                "role": "assistant",
                "model": "anthropic/claude-sonnet-5-5",
                "content": [{"type": "text", "text": "Hello from ngrok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="ngrok/anthropic/claude-sonnet-5-5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="ng-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{NGROK_BASE}/messages"
    assert request.headers["authorization"] == "Bearer ng-test-key"
    assert body["model"] == "anthropic/claude-sonnet-5-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from ngrok"


def test_ngrok_embedding_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{NGROK_BASE}/embeddings").respond(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "amazon/titan-embed-text-v2",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
        response: Final = litellm.embedding(
            model="ngrok/amazon/titan-embed-text-v2",
            input=["hello"],
            api_key="ng-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{NGROK_BASE}/embeddings"
    assert request.headers["authorization"] == "Bearer ng-test-key"
    assert body["model"] == "amazon/titan-embed-text-v2"
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
