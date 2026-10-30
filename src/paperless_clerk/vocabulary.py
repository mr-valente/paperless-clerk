"""On-demand vocabulary consolidation.

The creation gate keeps new duplicates out; this module finds the ones that
already exist, whether inherited from earlier tooling or slipped past, and merges
them on request. Detection combines the lexical duplicate rules the planner uses
with one model pass over the whole library. Merging moves every document from one
entry to another and deletes the emptied entry, so it only ever runs on an
explicit user action.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from paperless_clerk.clients.openai_compatible import ModelError, OpenAICompatibleClient
from paperless_clerk.clients.paperless import PaperlessClient
from paperless_clerk.config import Settings
from paperless_clerk.domain.taxonomy import Entity, duplicate_similarity, entities_from_payload
from paperless_clerk.metadata import MetadataAnalyzer

log = logging.getLogger(__name__)

RESOURCES = {"tag": "tags", "correspondent": "correspondents", "document_type": "document_types"}
LEXICAL_THRESHOLD = 0.86


def pair_key(kind: str, left_id: int, right_id: int) -> str:
    return f"{kind}:{min(left_id, right_id)}:{max(left_id, right_id)}"


def visible_catalog(
    catalog: dict[str, list[dict[str, Any]]], settings: Settings
) -> dict[str, list[dict[str, Any]]]:
    """The vocabulary as the model sees it: no inbox or workflow tags."""
    hidden = settings.hidden_tag_names
    return {
        **catalog,
        "tags": [
            tag
            for tag in catalog.get("tags", [])
            if not tag.get("is_inbox_tag")
            and " ".join(str(tag.get("name", "")).split()).casefold() not in hidden
        ],
    }


def _canonical_order(left: Entity, right: Entity) -> tuple[Entity, Entity]:
    """Keep the busier entry; when usage ties, keep the longer, spelled-out name."""
    ordered = sorted(
        (left, right), key=lambda item: (-item.document_count, -len(item.name), item.name)
    )
    return ordered[0], ordered[1]


def _suggestion(kind: str, keep: Entity, merge: Entity, reason: str, source: str) -> dict[str, Any]:
    return {
        "key": pair_key(kind, keep.id, merge.id),
        "kind": kind,
        "keep_id": keep.id,
        "keep_name": keep.name,
        "keep_usage": keep.document_count,
        "merge_id": merge.id,
        "merge_name": merge.name,
        "merge_usage": merge.document_count,
        "reason": reason,
        "source": source,
    }


def lexical_suggestions(catalog: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Same-kind pairs the planner would already fold into one another."""
    suggestions: list[dict[str, Any]] = []
    for kind, resource in RESOURCES.items():
        entities = entities_from_payload(catalog.get(resource, []))
        for index, left in enumerate(entities):
            for right in entities[index + 1 :]:
                score, reason = duplicate_similarity(left.name, right.name, kind)
                if score < LEXICAL_THRESHOLD:
                    continue
                keep, merge = _canonical_order(left, right)
                suggestions.append(
                    _suggestion(kind, keep, merge, f"{reason} ({score:.0%} similar)", "lexical")
                )
    return suggestions


async def review_vocabulary(
    settings: Settings, paperless: PaperlessClient, model: OpenAICompatibleClient
) -> dict[str, Any]:
    """Collect merge suggestions from the lexical rules and one model audit."""
    catalog = visible_catalog(await paperless.catalog(), settings)
    suggestions = lexical_suggestions(catalog)
    seen = {item["key"] for item in suggestions}
    by_kind = {
        kind: {entity.id: entity for entity in entities_from_payload(catalog.get(resource, []))}
        for kind, resource in RESOURCES.items()
    }
    model_status = "completed"
    try:
        audit = await MetadataAnalyzer(settings, model).audit_vocabulary(catalog)
    except ModelError as exc:
        log.warning("Vocabulary audit model pass failed; lexical suggestions only: %s", exc)
        model_status = f"failed: {str(exc)[:300]}"
    else:
        for item in audit.suggestions:
            keep = by_kind[item.kind].get(item.keep_id)
            merge = by_kind[item.kind].get(item.merge_id)
            if keep is None or merge is None or keep.id == merge.id:
                continue
            key = pair_key(item.kind, keep.id, merge.id)
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(_suggestion(item.kind, keep, merge, item.reason, "model"))
    order = {"tag": 0, "document_type": 1, "correspondent": 2}
    suggestions.sort(key=lambda item: (order[item["kind"]], item["keep_name"].casefold()))
    return {
        "reviewed_at": time.time(),
        "counts": {resource: len(catalog.get(resource, [])) for resource in RESOURCES.values()},
        "model_status": model_status,
        "suggestions": suggestions,
    }


async def merge_entities(
    paperless: PaperlessClient, kind: str, keep_id: int, merge_id: int
) -> dict[str, Any]:
    """Move every document from ``merge_id`` onto ``keep_id`` and delete the emptied entry."""
    resource = RESOURCES[kind]
    entities = {int(item["id"]): item for item in await paperless.list_resource(resource)}
    keep = entities.get(keep_id)
    merge = entities.get(merge_id)
    if keep is None or merge is None:
        raise LookupError(f"Both {resource} must still exist in Paperless")
    document_ids = await paperless.document_ids_with(kind, merge_id)
    if document_ids:
        if kind == "tag":
            await paperless.bulk_edit(
                document_ids, "modify_tags", {"add_tags": [keep_id], "remove_tags": [merge_id]}
            )
        elif kind == "correspondent":
            await paperless.bulk_edit(document_ids, "set_correspondent", {"correspondent": keep_id})
        else:
            await paperless.bulk_edit(document_ids, "set_document_type", {"document_type": keep_id})
    await paperless.delete_entity(resource, merge_id)
    log.info(
        "Merged %s %s (%s) into %s (%s) across %s document(s)",
        kind,
        merge.get("name"),
        merge_id,
        keep.get("name"),
        keep_id,
        len(document_ids),
    )
    return {
        "kind": kind,
        "keep": {"id": keep_id, "name": keep.get("name")},
        "merged": {"id": merge_id, "name": merge.get("name")},
        "documents": len(document_ids),
    }
