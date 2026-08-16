import base64
import json

import httpx
import pytest

from paperless_clerk.clients.openai_compatible import ModelError, OpenAICompatibleClient
from paperless_clerk.config import Settings


@pytest.mark.asyncio
async def test_ocr_and_metadata_clients_share_one_base_url() -> None:
    settings = Settings(
        openai_base_url="http://models.local:1234/v1/", openai_api_key="shared-secret"
    )
    ocr = OpenAICompatibleClient(settings, "ocr")
    metadata = OpenAICompatibleClient(settings, "metadata")

    assert ocr.completions_url == "http://models.local:1234/v1/chat/completions"
    assert metadata.completions_url == ocr.completions_url
    assert ocr.client.headers["authorization"] == "Bearer shared-secret"
    assert metadata.client.headers["authorization"] == "Bearer shared-secret"

    await ocr.close()
    await metadata.close()


@pytest.mark.asyncio
async def test_structured_output_falls_back_without_consuming_retry_budget() -> None:
    formats: list[str] = []
    schema_prompts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        schema_prompts.append("\n".join(message["content"] for message in payload["messages"]))
        response_format = payload.get("response_format")
        formats.append(response_format.get("type") if response_format else "none")
        if len(formats) < 3:
            return httpx.Response(400, text="response_format is unsupported")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]},
        )

    client = OpenAICompatibleClient(Settings(model_max_retries=0), "metadata")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = await client.structured(
        name="answer",
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        system="Return JSON",
        user="Test",
    )
    await client.close()

    assert result == {"answer": "ok"}
    assert formats == ["json_schema", "json_object", "none"]
    assert all("Return exactly one JSON object matching" in prompt for prompt in schema_prompts)
    assert all('"answer"' in prompt for prompt in schema_prompts)


@pytest.mark.asyncio
async def test_ocr_response_scaffolding_is_removed() -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "<think>internal notes</think>\n```text\nExact page text\n```"
                        }
                    }
                ]
            },
        )

    client = OpenAICompatibleClient(Settings(), "ocr")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    text = await client.ocr_page(b"fake image", page_number=1, prompt="Transcribe")
    await client.close()

    assert text == "Exact page text"
    assert [message["role"] for message in requests[0]["messages"]] == ["system", "user"]
    assert requests[0]["messages"][1]["content"][0]["text"] == "Page 1. Transcribe"
    assert "top_k" not in requests[0]
    assert "chat_template" not in requests[0]
    assert "skip_special_tokens" not in requests[0]
    assert "vllm_xargs" not in requests[0]
    assert "repetition_detection" not in requests[0]


@pytest.mark.asyncio
async def test_deepseek_ocr_vllm_profile_sends_native_request_contract() -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Exact specialist OCR text"}}]},
        )

    client = OpenAICompatibleClient(
        Settings(ocr_profile="deepseek_ocr", ocr_model="user.DeepSeek-OCR-2"), "ocr"
    )
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    text = await client.ocr_page(
        b"\x89PNG\r\n\x1a\nfake image", page_number=9, prompt="generic prompt"
    )
    await client.close()

    assert text == "Exact specialist OCR text"
    payload = requests[0]
    assert payload["temperature"] == 0
    assert "top_k" not in payload
    # vLLM owns the DeepSeek-OCR fallback template. Sending one in the request
    # would require the unsafe-by-default --trust-request-chat-template flag.
    assert "chat_template" not in payload
    assert payload["skip_special_tokens"] is False
    assert payload["vllm_xargs"] == {
        "ngram_size": 20,
        "window_size": 90,
        "whitelist_token_ids": [128821, 128822],
    }
    assert "repetition_detection" not in payload
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {
        "type": "text",
        "text": "<|grounding|>Convert the document to markdown.",
    }
    assert "Page 9" not in json.dumps(payload)
    assert "generic prompt" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_deepseek_ocr_llamacpp_profile_reproduces_the_earlier_gguf_contract() -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Exact GGUF OCR text"}}]},
        )

    client = OpenAICompatibleClient(
        Settings(
            ocr_profile="deepseek_ocr_llamacpp",
            ocr_model="sabafallah/DeepSeek-OCR-2-GGUF",
        ),
        "ocr",
    )
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    text = await client.ocr_page(b"jpeg image", page_number=2, prompt="generic prompt")
    await client.close()

    assert text == "Exact GGUF OCR text"
    payload = requests[0]
    assert payload["temperature"] == 0
    assert payload["top_k"] == 1
    assert "skip_special_tokens" not in payload
    assert "vllm_xargs" not in payload
    assert "repetition_detection" not in payload
    content = payload["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "Free OCR."}


@pytest.mark.asyncio
async def test_deepseek_profile_removes_layout_annotations_but_keeps_ocr_text() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<|grounding|><|ref|>text<|/ref|>"
                                "<|det|>[[10, 20, 30, 40]]<|/det|>\nInvoice\n\n\n"
                                "<|ref|>sub_title<|/ref|>"
                                "<|det|>[[10, 50, 30, 70]]<|/det|>\n## Total\n$25.00"
                                "<｜end▁of▁sentence｜>"
                            )
                        }
                    }
                ]
            },
        )

    client = OpenAICompatibleClient(Settings(ocr_profile="deepseek_ocr"), "ocr")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    text = await client.ocr_page(b"fake image", page_number=1, prompt="unused")
    await client.close()

    assert text == "Invoice\n\n## Total\n$25.00"
    assert "sub_title" not in text


@pytest.mark.asyncio
async def test_deepseek_profile_rejects_annotation_only_output() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<|ref|>text<|/ref|>"
                                "<|det|>[[10, 20, 300, 80]]<|/det|>"
                                "<｜end▁of▁sentence｜>"
                            )
                        }
                    }
                ]
            },
        )

    client = OpenAICompatibleClient(Settings(ocr_profile="deepseek_ocr"), "ocr")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ModelError, match=r"1 layout region\(s\) but no transcription.*unspecified"):
        await client.ocr_page(b"fake image", page_number=1, prompt="unused")
    await client.close()


@pytest.mark.asyncio
async def test_deepseek_profile_rejects_partial_layout_annotations() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "<|ref|>sub_title<|/ref|>This must not be published"}}
                ]
            },
        )

    client = OpenAICompatibleClient(Settings(ocr_profile="deepseek_ocr"), "ocr")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ModelError, match="malformed layout annotations"):
        await client.ocr_page(b"fake image", page_number=1, prompt="unused")
    await client.close()


@pytest.mark.asyncio
async def test_ocr_connection_test_exercises_a_real_image_request() -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "PAPERLESS CLERK\nReference number: 4827\n"
                                "END OF CLERK OCR TEST"
                            )
                        }
                    }
                ]
            },
        )

    client = OpenAICompatibleClient(Settings(ocr_profile="deepseek_ocr"), "ocr")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    result = await client.test_connection()
    await client.close()

    image_url = requests[0]["messages"][0]["content"][0]["image_url"]["url"]
    image = base64.b64decode(image_url.split(",", 1)[1])
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(image) > 1_000
    assert result == {
        "ok": True,
        "model": "qwen2.5vl:7b",
        "profile": "deepseek_ocr",
        "response": "PAPERLESS CLERK\nReference number: 4827\nEND OF CLERK OCR TEST",
        "message": (
            "OCR responded using DeepSeek's required loop guard. This profile is "
            "review-only and will not replace Paperless OCR automatically."
        ),
    }


@pytest.mark.asyncio
async def test_ocr_connection_test_rejects_a_missing_footer_region() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "PAPERLESS CLERK\nReference number: 4827"
                        }
                    }
                ]
            },
        )

    client = OpenAICompatibleClient(Settings(ocr_profile="deepseek_ocr"), "ocr")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ModelError, match=r"missed test image region\(s\): footer"):
        await client.test_connection()
    await client.close()


@pytest.mark.asyncio
async def test_truncated_ocr_page_is_rejected() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "A partial page that must not be published"},
                    }
                ]
            },
        )

    client = OpenAICompatibleClient(Settings(), "ocr")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ModelError, match="token limit"):
        await client.ocr_page(b"fake image", page_number=7, prompt="Transcribe")
    await client.close()


@pytest.mark.asyncio
async def test_deepseek_token_limit_reports_failed_guard_without_recommending_more_tokens() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "Charles Schwab " * 100},
                    }
                ]
            },
        )

    client = OpenAICompatibleClient(Settings(ocr_profile="deepseek_ocr"), "ocr")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ModelError, match="despite its required vLLM loop guard") as raised:
        await client.ocr_page(b"fake image", page_number=1, prompt="unused")
    await client.close()

    assert "Increasing the token limit would only prolong this failure" in str(raised.value)


@pytest.mark.asyncio
async def test_server_repetition_finish_discards_partial_text() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "repetition",
                        "message": {"content": "Charles Schwab " * 8},
                    }
                ]
            },
        )

    client = OpenAICompatibleClient(Settings(ocr_profile="deepseek_ocr"), "ocr")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ModelError, match="repetition loop and was discarded"):
        await client.ocr_page(b"fake image", page_number=1, prompt="unused")
    await client.close()


@pytest.mark.asyncio
async def test_legitimate_repeated_ocr_text_is_never_rewritten_by_the_client() -> None:
    repeated = "Section 12 shall remain in effect. " * 10

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": repeated}}]},
        )

    client = OpenAICompatibleClient(Settings(ocr_profile="deepseek_ocr"), "ocr")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    text = await client.ocr_page(b"fake image", page_number=1, prompt="unused")
    await client.close()

    assert text == repeated.strip()


@pytest.mark.asyncio
async def test_empty_choices_are_reported_as_a_model_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = OpenAICompatibleClient(Settings(model_max_retries=0), "ocr")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ModelError, match=r"choices\[0\]"):
        await client.ocr_page(b"fake image", page_number=1, prompt="Transcribe")
    await client.close()


@pytest.mark.asyncio
async def test_non_object_response_is_rejected_cleanly() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = OpenAICompatibleClient(Settings(model_max_retries=0), "metadata")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ModelError, match="invalid response JSON"):
        await client.test_connection()
    await client.close()
