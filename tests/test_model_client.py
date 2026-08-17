import base64
import json

import httpx
import pytest

from paperless_clerk.clients.openai_compatible import ModelError, OpenAICompatibleClient
from paperless_clerk.config import Settings
from paperless_clerk.ocr_profiles import GENERIC_PAGE_PROMPT, GENERIC_SYSTEM_PROMPT


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


async def _ocr_client(settings: Settings, handler) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient(settings, "ocr")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _responds(content: str, **choice: object):
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{**choice, "message": {"content": content}}]})

    return handler


@pytest.mark.asyncio
async def test_generic_profile_sends_a_system_prompt_and_strips_scaffolding() -> None:
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

    client = await _ocr_client(Settings(), handler)
    text = await client.ocr_page(b"fake image", page_number=1)
    await client.close()

    assert text == "Exact page text"
    payload = requests[0]
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    assert "verbatim" in payload["messages"][0]["content"]
    content = payload["messages"][1]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["text"].startswith("Transcribe this page verbatim.")
    # Nothing server-specific: this profile has to suit any OpenAI-compatible endpoint.
    assert set(payload) == {"model", "temperature", "max_tokens", "messages"}


def test_the_generic_prompt_keeps_its_load_bearing_instructions() -> None:
    prompt = f"{GENERIC_SYSTEM_PROMPT}\n{GENERIC_PAGE_PROMPT}".casefold()

    # A chat model that announces its answer, narrates the page, or protects the
    # account numbers on it produces a filed document that is wrong rather than
    # merely worse, and an empty reply for a blank page fails the whole job.
    assert "no preamble" in prompt
    assert "never redact" in prompt
    assert "never describe, summarize" in prompt
    assert "[blank page]" in prompt
    assert "[illegible]" in prompt
    # It must not read as a request to explain, or a reasoning model will oblige.
    assert "?" not in prompt


@pytest.mark.asyncio
async def test_deepseek_vllm_profile_sends_the_recipe_repetition_guard(vllm_profiles: None) -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "Exact specialist OCR text"}}]}
        )

    client = await _ocr_client(
        Settings(ocr_profile="deepseek_ocr", ocr_model="user.DeepSeek-OCR-2"), handler
    )
    text = await client.ocr_page(b"jpeg image", page_number=9)
    await client.close()

    assert text == "Exact specialist OCR text"
    payload = requests[0]
    assert payload["model"] == "user.DeepSeek-OCR-2"
    assert payload["temperature"] == 0
    # The server-side NGramPerReqLogitsProcessor only engages when the request
    # supplies its window. Without this half, dense pages loop to the limit.
    assert payload["skip_special_tokens"] is False
    assert payload["vllm_xargs"] == {
        "ngram_size": 30,
        "window_size": 90,
        "whitelist_token_ids": [128821, 128822],
    }
    assert len(payload["messages"]) == 1
    content = payload["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "Free OCR."}
    assert "Page 9" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_deepseek_llamacpp_profile_keeps_the_known_good_gguf_request() -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Exact GGUF text"}}]})

    client = await _ocr_client(
        Settings(ocr_profile="deepseek_ocr_llamacpp", ocr_model="sabafallah/DeepSeek-OCR-2-GGUF"),
        handler,
    )
    text = await client.ocr_page(b"jpeg image", page_number=2)
    await client.close()

    assert text == "Exact GGUF text"
    payload = requests[0]
    assert payload["temperature"] == 0
    assert payload["top_k"] == 1
    # llama.cpp has no logits processor to configure; the vLLM-only fields must
    # never leak into the one serving path that already works.
    assert "skip_special_tokens" not in payload
    assert "vllm_xargs" not in payload
    assert len(payload["messages"]) == 1
    content = payload["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "Free OCR."}


@pytest.mark.asyncio
async def test_glm_ocr_profile_sends_its_native_task_command(vllm_profiles: None) -> None:
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Exact GLM text"}}]})

    client = await _ocr_client(
        Settings(ocr_profile="glm_ocr", ocr_model="user.GLM-OCR-vLLM", ocr_max_output_tokens=2345),
        handler,
    )
    text = await client.ocr_page(b"jpeg image", page_number=4)
    await client.close()

    assert text == "Exact GLM text"
    payload = requests[0]
    assert payload["model"] == "user.GLM-OCR-vLLM"
    assert payload["max_tokens"] == 2345
    assert payload["temperature"] == 0
    assert "top_k" not in payload
    assert "skip_special_tokens" not in payload
    assert "vllm_xargs" not in payload
    assert len(payload["messages"]) == 1
    content = payload["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "Text Recognition:"}


@pytest.mark.asyncio
async def test_grounding_annotations_and_control_tokens_are_removed(vllm_profiles: None) -> None:
    client = await _ocr_client(
        Settings(ocr_profile="deepseek_ocr"),
        _responds(
            "<|grounding|><|ref|>text<|/ref|>"
            "<|det|>[[10, 20, 30, 40]]<|/det|>\nInvoice\n\n\n"
            "<|ref|>sub_title<|/ref|>"
            "<|det|>[[10, 50, 30, 70]]<|/det|>\n## Total\n$25.00"
            "<｜end▁of▁sentence｜>"
        ),
    )
    text = await client.ocr_page(b"fake image", page_number=1)
    await client.close()

    assert text == "Invoice\n\n## Total\n$25.00"


@pytest.mark.asyncio
async def test_an_unpaired_layout_label_is_removed_rather_than_published(
    vllm_profiles: None,
) -> None:
    client = await _ocr_client(
        Settings(ocr_profile="deepseek_ocr"),
        _responds("<|ref|>sub_title<|/ref|>Amount due 25.00"),
    )
    text = await client.ocr_page(b"fake image", page_number=1)
    await client.close()

    assert text == "Amount due 25.00"


@pytest.mark.asyncio
async def test_annotation_only_output_reports_the_raw_response(vllm_profiles: None) -> None:
    client = await _ocr_client(
        Settings(ocr_profile="deepseek_ocr", model_max_retries=0),
        _responds("<|ref|>text<|/ref|><|det|>[[10, 20, 300, 80]]<|/det|>"),
    )

    with pytest.raises(ModelError, match=r"no transcription for page 1.*<\|ref\|>"):
        await client.ocr_page(b"fake image", page_number=1)
    await client.close()


@pytest.mark.asyncio
async def test_ocr_connection_test_exercises_a_real_image_request(vllm_profiles: None) -> None:
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
                                "PAPERLESS CLERK\nReference number: 4827\nEND OF CLERK OCR TEST"
                            )
                        }
                    }
                ]
            },
        )

    client = await _ocr_client(Settings(ocr_profile="deepseek_ocr"), handler)
    result = await client.test_connection()
    await client.close()

    image_url = requests[0]["messages"][0]["content"][0]["image_url"]["url"]
    image = base64.b64decode(image_url.split(",", 1)[1])
    assert image.startswith(b"\xff\xd8")
    assert len(image) > 1_000
    assert result == {
        "ok": True,
        "model": "qwen2.5vl:7b",
        "profile": "deepseek_ocr",
        "response": "PAPERLESS CLERK\nReference number: 4827\nEND OF CLERK OCR TEST",
    }


@pytest.mark.asyncio
async def test_ocr_connection_test_accepts_a_partial_but_recognizable_read() -> None:
    client = await _ocr_client(
        Settings(),
        _responds("PAPERLESS CLERK\nReference number: 4827"),
    )
    result = await client.test_connection()
    await client.close()

    assert result["ok"] is True
    assert result["response"] == "PAPERLESS CLERK\nReference number: 4827"


@pytest.mark.asyncio
async def test_ocr_connection_test_fails_when_nothing_recognizable_is_read() -> None:
    client = await _ocr_client(Settings(), _responds("Lorem ipsum dolor sit amet"))

    with pytest.raises(ModelError, match="Lorem ipsum"):
        await client.test_connection()
    await client.close()


def _responds_with_usage(content: str, *, prompt: int, completion: int, **choice: object):
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{**choice, "message": {"content": content}}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
            },
        )

    return handler


@pytest.mark.asyncio
async def test_truncated_page_keeps_the_text_transcribed_before_the_cut() -> None:
    partial = "Charles Schwab address change confirmation for account 4827."
    client = await _ocr_client(
        Settings(ocr_max_output_tokens=512),
        _responds(partial, finish_reason="length"),
    )

    assert await client.ocr_page(b"fake image", page_number=7) == partial
    await client.close()


@pytest.mark.asyncio
async def test_truncated_page_drops_a_runaway_decoder_loop() -> None:
    looped = "Real page text.\n" + "Page 1 of 1\n" * 40
    client = await _ocr_client(
        Settings(ocr_max_output_tokens=512),
        _responds(looped, finish_reason="length"),
    )

    text = await client.ocr_page(b"fake image", page_number=7)
    assert text == "Real page text.\nPage 1 of 1"
    await client.close()


@pytest.mark.asyncio
async def test_truncation_with_nothing_left_after_cleaning_still_fails() -> None:
    client = await _ocr_client(
        Settings(ocr_max_output_tokens=512),
        _responds_with_usage(
            "<|end▁of▁sentence|>", prompt=3053, completion=512, finish_reason="length"
        ),
    )

    with pytest.raises(ModelError, match="no usable text"):
        await client.ocr_page(b"fake image", page_number=1)
    await client.close()


@pytest.mark.asyncio
async def test_truncation_short_of_the_request_blames_the_context_not_the_output() -> None:
    """A server clamping output to fit the image needs a different fix than a loop."""

    client = await _ocr_client(Settings(ocr_max_output_tokens=4096), _responds("x"))
    detail = client._truncation_detail({"usage": {"prompt_tokens": 7800, "completion_tokens": 390}})

    assert "image filled the context" in detail
    assert "8190" in detail  # the context size the server actually needs to clear
    await client.close()


@pytest.mark.asyncio
async def test_truncation_at_the_full_request_blames_repetition() -> None:
    client = await _ocr_client(Settings(ocr_max_output_tokens=512), _responds("x"))
    detail = client._truncation_detail({"usage": {"prompt_tokens": 3053, "completion_tokens": 512}})

    assert "repeated itself" in detail
    await client.close()


@pytest.mark.asyncio
async def test_legitimate_repeated_ocr_text_is_never_rewritten_by_the_client() -> None:
    repeated = "Section 12 shall remain in effect. " * 10
    client = await _ocr_client(Settings(), _responds(repeated, finish_reason="stop"))

    text = await client.ocr_page(b"fake image", page_number=1)
    await client.close()

    assert text == repeated.strip()


@pytest.mark.asyncio
async def test_empty_choices_are_reported_as_a_model_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = await _ocr_client(Settings(model_max_retries=0), handler)

    with pytest.raises(ModelError, match=r"choices\[0\]"):
        await client.ocr_page(b"fake image", page_number=1)
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
