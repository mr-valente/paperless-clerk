from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass

PAGE_MARKER = re.compile(r"(?im)^\s*(?:[-=#*]+\s*)?page\s+\d+\s*(?:of\s+\d+)?\s*[-=#*]*\s*$")
WORD = re.compile(r"[^\W_]+", re.UNICODE)


def normalized_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = PAGE_MARKER.sub(" ", text)
    return WORD.findall(text.casefold())


def meaningful_ocr(text: str, minimum_characters: int = 40) -> bool:
    tokens = normalized_tokens(text)
    characters = sum(len(token) for token in tokens)
    return characters >= minimum_characters and len(set(tokens)) >= 5


def _dice_counter(left: Counter[object], right: Counter[object]) -> float:
    total = sum(left.values()) + sum(right.values())
    if total == 0:
        return 1.0
    overlap = sum((left & right).values())
    return (2.0 * overlap) / total


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _shingles(tokens: list[str], width: int = 3) -> Counter[tuple[str, ...]]:
    if len(tokens) < width:
        return Counter((token,) for token in tokens)
    return Counter(tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1))


@dataclass(frozen=True)
class OCRComparison:
    score: float
    is_similar: bool
    token_overlap: float
    vocabulary_overlap: float
    ordered_shingle_overlap: float
    length_agreement: float
    numeric_overlap: float
    existing_tokens: int
    generated_tokens: int
    mismatch_snippets: list[dict[str, str]]

    def metrics(self) -> dict[str, float | int | bool]:
        result = asdict(self)
        result.pop("mismatch_snippets")
        return result


def compare_ocr(existing: str, generated: str, threshold: float = 0.82) -> OCRComparison:
    left = normalized_tokens(existing)
    right = normalized_tokens(generated)

    left_count = Counter(left)
    right_count = Counter(right)
    token_overlap = _dice_counter(left_count, right_count)
    vocabulary_overlap = _jaccard(set(left), set(right))
    ordered_overlap = _dice_counter(_shingles(left), _shingles(right))
    length_agreement = min(len(left), len(right)) / max(1, max(len(left), len(right)))

    def numeric_terms(tokens: list[str]) -> Counter[str]:
        # OCR punctuation normalization splits 1250.00 into "1250", "00".
        # A zero-only decimal tail carries no independent information and would
        # otherwise make a completely different amount appear 50% equal.
        return Counter(
            token
            for token in tokens
            if any(character.isdigit() for character in token)
            and set(character for character in token if character.isdigit()) != {"0"}
        )

    left_numbers = numeric_terms(left)
    right_numbers = numeric_terms(right)
    numeric_overlap = _dice_counter(left_numbers, right_numbers)

    score = (
        0.50 * token_overlap
        + 0.15 * vocabulary_overlap
        + 0.20 * ordered_overlap
        + 0.10 * length_agreement
        + 0.05 * numeric_overlap
    )
    if left_numbers and right_numbers and numeric_overlap < 0.5:
        # Numbers are often the most consequential OCR differences. This mild
        # penalty nudges otherwise similar invoices/statements into review.
        score *= 0.95
    score = round(max(0.0, min(1.0, score)), 4)

    numeric_agreement = not (left_numbers or right_numbers) or numeric_overlap >= 0.75
    enough_coverage = length_agreement >= 0.65 and token_overlap >= 0.72 and numeric_agreement
    return OCRComparison(
        score=score,
        is_similar=score >= threshold and enough_coverage,
        token_overlap=round(token_overlap, 4),
        vocabulary_overlap=round(vocabulary_overlap, 4),
        ordered_shingle_overlap=round(ordered_overlap, 4),
        length_agreement=round(length_agreement, 4),
        numeric_overlap=round(numeric_overlap, 4),
        existing_tokens=len(left),
        generated_tokens=len(right),
        mismatch_snippets=_mismatch_snippets(left, right),
    )


def _mismatch_snippets(left: list[str], right: list[str], limit: int = 8) -> list[dict[str, str]]:
    # The diagnostic diff is deliberately sampled. Running SequenceMatcher over
    # an unbounded 100-page document can have quadratic behavior.
    left_sample = left[:5000]
    right_sample = right[:5000]
    matcher = difflib.SequenceMatcher(a=left_sample, b=right_sample, autojunk=True)
    snippets: list[dict[str, str]] = []
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        if operation == "equal":
            continue
        before = " ".join(left_sample[max(0, i1 - 8) : min(len(left_sample), i2 + 8)])
        after = " ".join(right_sample[max(0, j1 - 8) : min(len(right_sample), j2 + 8)])
        snippets.append(
            {"operation": operation, "existing": before[:700], "generated": after[:700]}
        )
        if len(snippets) >= limit:
            break
    return snippets


def assemble_pages(pages: list[tuple[int, str]]) -> str:
    sections = []
    for page_number, text in sorted(pages):
        clean = text.strip()
        sections.append(f"--- Page {page_number} ---\n{clean}")
    return "\n\n".join(sections).strip() + "\n"
