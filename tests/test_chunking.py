from paperless_clerk.domain.chunking import chunk_pages, pages_from_assembled_text
from paperless_clerk.domain.ocr_compare import assemble_pages


def test_hundred_page_document_is_bounded_and_complete() -> None:
    pages = [
        (page, f"Page-specific token-{page} " + ("account statement text " * 180))
        for page in range(1, 121)
    ]

    chunks = chunk_pages(pages, maximum_characters=2200)

    assert len(chunks) > 120
    assert all(len(chunk.text) <= 2200 for chunk in chunks)
    combined = "\n".join(chunk.text for chunk in chunks)
    for page in range(1, 121):
        assert f"token-{page}" in combined
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 120


def test_assembled_page_text_round_trips() -> None:
    pages = [(1, "First page"), (2, "Second page\nwith a second line")]

    assert pages_from_assembled_text(assemble_pages(pages)) == pages
