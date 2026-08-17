from paperless_clerk.domain.ocr_compare import compare_ocr, meaningful_ocr


def test_formatting_noise_is_not_a_conflict() -> None:
    existing = """INTERNAL REVENUE SERVICE

    Form 1099-INT — Tax Year 2025
    Payer's federal identification number: 12-3456789
    Interest income: $1,245.22"""
    generated = """Internal Revenue Service Form 1099 INT
    Tax Year 2025. Payers federal identification number 12 3456789.
    Interest income $1 245 22"""

    comparison = compare_ocr(existing, generated, threshold=0.78)

    assert comparison.is_similar
    assert comparison.token_overlap > 0.9


def test_line_order_artifacts_are_tolerated() -> None:
    existing = "Account holder Jane Doe\nStatement period January 2026\nOpening balance 1200\nClosing balance 1400"
    generated = "Statement period January 2026\nAccount holder Jane Doe\nClosing balance 1400\nOpening balance 1200"

    comparison = compare_ocr(existing, generated, threshold=0.75)

    assert comparison.is_similar
    assert comparison.token_overlap == 1.0
    assert comparison.ordered_shingle_overlap < comparison.token_overlap


def test_meaningful_disagreement_creates_conflict() -> None:
    existing = "Invoice 1048 from Acme Medical. Total due 1250.00 by February 10 2026."
    generated = "Meeting notes for a garden committee. Bring seeds and tools next Saturday."

    comparison = compare_ocr(existing, generated)

    assert not comparison.is_similar
    assert comparison.score < 0.5
    assert comparison.mismatch_snippets


def test_single_critical_number_disagreement_is_not_silently_accepted() -> None:
    existing = "Amount due for the annual insurance policy is USD 1250.00."
    generated = "Amount due for the annual insurance policy is USD 7250.00."

    comparison = compare_ocr(existing, generated)

    assert comparison.numeric_overlap == 0
    assert not comparison.is_similar


def test_meaningful_ocr_rejects_tiny_scanner_noise() -> None:
    assert not meaningful_ocr("scan 1 of 1")
    assert meaningful_ocr(
        "This is a complete statement containing several distinct and useful words."
    )


def test_token_counts_expose_a_missing_footer_even_when_documents_are_similar() -> None:
    body = (
        "Charles Schwab updated your contact information and asks you to review the "
        "account details online. Thank you for investing with Schwab. "
    ) * 8
    footer = (
        "Brokerage products are not FDIC insured and may lose value. "
        "Deposit products are offered by Charles Schwab Bank Member FDIC."
    )

    comparison = compare_ocr(body + footer, body)

    assert comparison.is_similar
    assert comparison.generated_tokens < 0.98 * comparison.existing_tokens
