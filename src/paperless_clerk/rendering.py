from __future__ import annotations

import math
from pathlib import Path

import pymupdf


class RenderError(RuntimeError):
    pass


def render_ocr_test_image(image_format: str = "jpeg") -> bytes:
    """Render a page-like document so an OCR health check exercises the projector."""

    if image_format not in {"jpeg", "png"}:
        raise ValueError("image_format must be jpeg or png")

    document = pymupdf.open()
    try:
        # DeepSeek's document prompt is trained on page layouts. A previous
        # wide, single-line banner could be classified as a layout region but
        # yield no Markdown transcription after the coordinates were removed.
        # Keep this fixture close to the pages production actually sends.
        page = document.new_page(width=612, height=792)
        page.insert_text((54, 78), "PAPERLESS CLERK", fontsize=24, fontname="hebo")
        page.insert_text((54, 112), "OCR CONNECTION TEST", fontsize=18, fontname="helv")
        page.draw_line((54, 130), (558, 130), width=1)
        page.insert_text((54, 185), "Connection check document", fontsize=15, fontname="hebo")
        page.insert_text(
            (54, 225),
            "Paperless Clerk is reading this rendered page through the vision model.",
            fontsize=12,
            fontname="helv",
        )
        page.insert_text(
            (54, 250),
            "Reference number: 4827. The expected result contains the word Clerk.",
            fontsize=12,
            fontname="helv",
        )
        page.insert_text((54, 735), "END OF CLERK OCR TEST", fontsize=11, fontname="helv")
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False, colorspace=pymupdf.csRGB)
        if image_format == "png":
            return pixmap.tobytes("png")
        return pixmap.tobytes("jpeg", jpg_quality=92)
    finally:
        document.close()


class DocumentRenderer:
    """Sequential, memory-conscious document page renderer."""

    def __init__(
        self,
        path: Path,
        *,
        dpi: int,
        max_pixels: int,
        jpeg_quality: int,
        image_format: str = "jpeg",
    ):
        self.path = path
        self.dpi = dpi
        self.max_pixels = max_pixels
        self.jpeg_quality = jpeg_quality
        if image_format not in {"jpeg", "png"}:
            raise ValueError("image_format must be jpeg or png")
        self.image_format = image_format
        try:
            self.document = pymupdf.open(path)
        except Exception as exc:  # pragma: no cover - library-specific exception tree
            raise RenderError(f"Unable to open document: {exc}") from exc
        if self.document.needs_pass:
            self.document.close()
            raise RenderError("Password-protected documents cannot be rendered")

    @property
    def page_count(self) -> int:
        return self.document.page_count

    def render(self, page_index: int) -> bytes:
        try:
            page = self.document.load_page(page_index)
            scale = self.dpi / 72.0
            target_pixels = page.rect.width * scale * page.rect.height * scale
            if target_pixels > self.max_pixels:
                scale *= math.sqrt(self.max_pixels / target_pixels)
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale), alpha=False, colorspace=pymupdf.csRGB
            )
            if self.image_format == "png":
                return pixmap.tobytes("png")
            return pixmap.tobytes("jpeg", jpg_quality=self.jpeg_quality)
        except Exception as exc:  # pragma: no cover - library-specific exception tree
            raise RenderError(f"Unable to render page {page_index + 1}: {exc}") from exc

    def close(self) -> None:
        self.document.close()

    def __enter__(self) -> DocumentRenderer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
