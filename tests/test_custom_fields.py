import pytest

from paperless_clerk.metadata import normalize_custom_value


def test_select_value_is_normalized_to_existing_option_id() -> None:
    definition = {
        "data_type": "select",
        "extra_data": {"select_options": [{"id": "health", "label": "Health Insurance"}]},
    }
    assert normalize_custom_value("health insurance", definition) == "health"


def test_monetary_value_uses_field_currency() -> None:
    definition = {"data_type": "monetary", "extra_data": {"default_currency": "USD"}}
    assert normalize_custom_value("1,245.2", definition) == "USD1245.20"


def test_invalid_typed_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid integer"):
        normalize_custom_value("twelve", {"data_type": "integer"})

    with pytest.raises(ValueError, match="supported range"):
        normalize_custom_value(2**40, {"data_type": "integer"})

    with pytest.raises(ValueError, match="must be finite"):
        normalize_custom_value("NaN", {"data_type": "float"})


def test_document_link_ids_are_not_trusted_without_a_candidate_set() -> None:
    with pytest.raises(ValueError, match="manual assignment"):
        normalize_custom_value([123], {"data_type": "documentlink"})
