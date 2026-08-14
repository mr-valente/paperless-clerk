from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    page_start: int
    page_end: int
    text: str


def _split_large_text(text: str, maximum: int) -> list[str]:
    if len(text) <= maximum:
        return [text]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [text[index : index + maximum] for index in range(0, len(text), maximum)]

    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > maximum:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                paragraph[index : index + maximum] for index in range(0, len(paragraph), maximum)
            )
        elif not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= maximum:
            current += "\n\n" + paragraph
        else:
            pieces.append(current)
            current = paragraph
    if current:
        pieces.append(current)
    return pieces


def chunk_pages(pages: list[tuple[int, str]], maximum_characters: int) -> list[TextChunk]:
    """Create bounded chunks without discarding page provenance."""

    units: list[tuple[int, str]] = []
    # Reserve room for the provenance marker so the final prompt chunk, not
    # merely the raw page fragment, respects the configured ceiling.
    fragment_limit = max(256, maximum_characters - 32)
    for page_number, text in sorted(pages):
        for part in _split_large_text(text.strip(), fragment_limit):
            if part:
                units.append((page_number, part))

    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_pages: list[int] = []
    current_size = 0

    def flush() -> None:
        nonlocal current_parts, current_pages, current_size
        if not current_parts:
            return
        chunks.append(
            TextChunk(
                index=len(chunks),
                page_start=min(current_pages),
                page_end=max(current_pages),
                text="\n\n".join(current_parts),
            )
        )
        current_parts, current_pages, current_size = [], [], 0

    for page_number, part in units:
        decorated = f"[Page {page_number}]\n{part}"
        if current_parts and current_size + len(decorated) + 2 > maximum_characters:
            flush()
        current_parts.append(decorated)
        current_pages.append(page_number)
        current_size += len(decorated) + 2
    flush()
    return chunks


def pages_from_assembled_text(text: str) -> list[tuple[int, str]]:
    pattern = re.compile(r"(?m)^--- Page (\d+) ---\s*$")
    matches = list(pattern.finditer(text))
    if not matches:
        return [(1, text)]
    pages: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pages.append((int(match.group(1)), text[start:end].strip()))
    return pages
