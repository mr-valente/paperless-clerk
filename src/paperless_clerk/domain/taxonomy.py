from __future__ import annotations

import difflib
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

WORD = re.compile(r"[^\W_]+", re.UNICODE)
CORPORATE_SUFFIXES = {
    "ag",
    "co",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "inc",
    "incorporated",
    "llc",
    "limited",
    "ltd",
    "plc",
    "sa",
}
GENERIC_TAXONOMY_WORDS = {"document", "documents", "record", "records", "file", "files"}


@dataclass(frozen=True)
class Entity:
    id: int
    name: str
    document_count: int = 0
    raw: dict[str, Any] | None = None

    def prompt_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "usage": self.document_count}


@dataclass(frozen=True)
class DuplicateMatch:
    entity: Entity
    score: float
    reason: str


def _singularize(token: str) -> str:
    special = {"taxes": "tax", "policies": "policy", "receipts": "receipt", "invoices": "invoice"}
    if token in special:
        return special[token]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def name_tokens(name: str, kind: str = "tag") -> list[str]:
    normalized = unicodedata.normalize("NFKD", name.casefold())
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    tokens = [_singularize(token) for token in WORD.findall(normalized)]
    if kind == "correspondent":
        tokens = [token for token in tokens if token not in CORPORATE_SUFFIXES]
    else:
        tokens = [token for token in tokens if token not in GENERIC_TAXONOMY_WORDS]
    return tokens


def compact_name(name: str, kind: str = "tag") -> str:
    return "".join(name_tokens(name, kind))


def acronym(name: str, kind: str = "correspondent") -> str:
    tokens = name_tokens(name, kind)
    return "".join(token[0] for token in tokens if token)


def duplicate_similarity(left: str, right: str, kind: str) -> tuple[float, str]:
    left_tokens = name_tokens(left, kind)
    right_tokens = name_tokens(right, kind)
    left_compact = "".join(left_tokens)
    right_compact = "".join(right_tokens)
    if left_compact and left_compact == right_compact:
        return 1.0, "same normalized name"

    left_acronym = acronym(left, kind)
    right_acronym = acronym(right, kind)
    if (
        left_compact
        and right_compact
        and (
            (len(left_compact) <= 6 and left_compact == right_acronym)
            or (len(right_compact) <= 6 and right_compact == left_acronym)
        )
    ):
        return 0.99, "acronym of existing name"

    left_set, right_set = set(left_tokens), set(right_tokens)
    token_score = len(left_set & right_set) / max(1, len(left_set | right_set))
    sequence_score = difflib.SequenceMatcher(
        a=left_compact, b=right_compact, autojunk=False
    ).ratio()
    containment = 0.0
    if (
        left_compact
        and right_compact
        and (left_compact in right_compact or right_compact in left_compact)
    ):
        containment = min(len(left_compact), len(right_compact)) / max(
            len(left_compact), len(right_compact)
        )
    score = max(sequence_score, 0.65 * token_score + 0.35 * sequence_score, containment * 0.96)
    if (
        kind in {"tag", "document_type"}
        and left_set
        and right_set
        and left_set != right_set
        and (left_set < right_set or right_set < left_set)
    ):
        # "Income Taxes" beside "Taxes" and "Medical Invoice" beside
        # "Invoice" are fragmentation risks even when character similarity is
        # modest. The planner can still accept an explicitly justified
        # distinction rather than silently creating the modifier-only label.
        return max(score, 0.88), "existing concept with only a modifier"
    return score, "near-duplicate normalized name"


def find_duplicate(
    name: str, entities: list[Entity], kind: str, threshold: float = 0.86
) -> DuplicateMatch | None:
    best: DuplicateMatch | None = None
    for entity in entities:
        score, reason = duplicate_similarity(name, entity.name, kind)
        if score >= threshold and (best is None or score > best.score):
            best = DuplicateMatch(entity=entity, score=round(score, 4), reason=reason)
    return best


def entities_from_payload(items: list[dict[str, Any]]) -> list[Entity]:
    return [
        Entity(
            id=int(item["id"]),
            name=str(item["name"]),
            document_count=int(item.get("document_count") or 0),
            raw=item,
        )
        for item in items
        if item.get("id") is not None and item.get("name")
    ]


def select_candidates(
    text: str,
    entities: list[Entity],
    *,
    kind: str,
    current_ids: set[int] | None = None,
    limit: int = 80,
) -> list[Entity]:
    if len(entities) <= limit:
        return sorted(entities, key=lambda entity: (-entity.document_count, entity.name.casefold()))

    current_ids = current_ids or set()
    document_tokens = set(name_tokens(text, kind))
    ranked: list[tuple[float, Entity]] = []
    for entity in entities:
        tokens = set(name_tokens(entity.name, kind))
        overlap = len(tokens & document_tokens) / max(1, len(tokens))
        exact = 1.0 if tokens and tokens <= document_tokens else 0.0
        entity_acronym = acronym(entity.name, kind)
        acronym_hit = 1.0 if len(entity_acronym) >= 2 and entity_acronym in document_tokens else 0.0
        current = 4.0 if entity.id in current_ids else 0.0
        usage = min(0.35, math.log1p(entity.document_count) / 30)
        ranked.append((current + 2.0 * exact + 1.6 * acronym_hit + 1.2 * overlap + usage, entity))
    ranked.sort(key=lambda pair: (-pair[0], -pair[1].document_count, pair[1].name.casefold()))
    return [entity for _, entity in ranked[:limit]]
