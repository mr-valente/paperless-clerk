from pathlib import Path

import pymupdf

from paperless_clerk.rendering import DocumentRenderer, render_ocr_test_image


def _write_test_pdf(path: Path) -> None:
    document = pymupdf.open()
    try:
        page = document.new_page(width=420, height=300)
        page.insert_text((30, 80), "Small but lossless text", fontsize=8)
        document.save(path)
    finally:
        document.close()


def test_document_renderer_produces_a_jpeg_page(tmp_path: Path) -> None:
    path = tmp_path / "page.pdf"
    _write_test_pdf(path)

    with DocumentRenderer(path, dpi=160, max_pixels=16_000_000, jpeg_quality=86) as renderer:
        image = renderer.render(0)

    assert image.startswith(b"\xff\xd8")
    assert pymupdf.Pixmap(image).width > 0


def test_ocr_connection_fixture_is_a_realistic_page() -> None:
    image = render_ocr_test_image()
    pixmap = pymupdf.Pixmap(image)

    assert image.startswith(b"\xff\xd8")
    assert pixmap.height > pixmap.width
    assert pixmap.width == 1224
    assert pixmap.height == 1584
