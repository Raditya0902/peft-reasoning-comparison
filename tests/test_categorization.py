"""Tests for example categorization logic."""

import pytest
from src.categorize import categorize


@pytest.mark.unit
def test_arithmetic_buys_more():
    result = categorize("If she has 3 apples and buys 4 more")
    assert "arithmetic" in result


@pytest.mark.unit
def test_fractions_percentages_percent_sign():
    result = categorize("What is 30% of 240?")
    assert "fractions_percentages" in result


@pytest.mark.unit
def test_unit_conversion_hours_to_minutes():
    result = categorize("Convert 3 hours to minutes")
    assert "unit_conversion" in result


@pytest.mark.unit
def test_implicit_unit_conversion_overlaps_arithmetic():
    tags = categorize("She buys a 2-yard ribbon. How many feet of ribbon does she have?")
    assert tags == ["arithmetic", "unit_conversion"]


@pytest.mark.unit
def test_conversion_does_not_match_a_single_unit_or_an_object():
    assert "unit_conversion" not in categorize("How many hours are left after 3 hours?")
    assert "unit_conversion" not in categorize("Convert 3 cups of flour to dough.")


@pytest.mark.unit
def test_measurement_abbreviation_does_not_count_as_an_equation():
    assert "algebraic" not in categorize("A 2 m + 4 m ribbon is cut into pieces.")


@pytest.mark.unit
def test_multi_hop_first_then_then():
    result = categorize("First she earns X, then spends Y, then saves Z")
    assert "multi_hop" in result


@pytest.mark.unit
def test_algebraic_variable_with_equation():
    result = categorize("What is x if 2x + 5 = 13?")
    assert "algebraic" in result


@pytest.mark.unit
def test_comparison_who_has_more():
    result = categorize("Who has more, Alice or Bob?")
    assert "comparison" in result


@pytest.mark.unit
def test_distractor_heavy_color_with_multiple_numbers():
    result = categorize("She has a red hat, 3 cats, and 12 dollars.")
    assert "distractor_heavy" in result


@pytest.mark.unit
def test_pure_arithmetic_only_one_tag():
    # No color words, no percent, no "to <unit>", no "first...then", no variable, no comparison marker
    result = categorize("She has 5 apples and buys 3 more.")
    assert result == ["arithmetic"]


@pytest.mark.unit
def test_no_match_returns_empty_list():
    result = categorize("Describe the concept of gravity.")
    assert result == []
