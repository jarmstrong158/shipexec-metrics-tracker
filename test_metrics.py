"""Tests for the pure data logic in metrics.py.

This tool scrapes ShipExec and turns the result into a daily report whose
headline is a labour-savings figure. The Selenium half needs a live browser
and a live ShipExec and is not tested here. The half that decides *what the
numbers say* — coercing scraped cells to integers, mapping people to regions,
matching initials — is pure, and is what a wrong report would come from.

Scraped cells are the messiest possible input: blanks, thousands separators,
floats-as-text, and NaN from pandas. safe_to_int swallows all of it by design,
so these pin what it swallows it *into*: a silent 0 where a real count was
meant is a report that is wrong rather than obviously broken.
"""
import numpy as np
import pandas as pd
import pytest

from metrics import (
    consignee_tokens_in_order,
    find_similar_initials,
    region_from_email,
    safe_to_int,
    sort_by_last_initial,
)


CONFIG = {
    "regions": {
        "North": {"email_pattern": "@north.example.com"},
        "South": {"email_pattern": "@south.example.com"},
    }
}


class TestSafeToInt:
    @pytest.mark.parametrize("value,expected", [
        (5, 5),
        (5.0, 5),
        (5.9, 5),            # truncates, does not round
        ("5", 5),
        ("  5  ", 5),
        ("1,234", 1234),     # thousands separator from a rendered table
        ("12.7", 12),        # float-as-text truncates too
        ("-3", -3),
    ])
    def test_parses_the_shapes_a_scraped_cell_arrives_in(self, value, expected):
        assert safe_to_int(value) == expected

    @pytest.mark.parametrize("value", ["", "   ", None, float("nan"), np.nan, pd.NA])
    def test_empty_and_missing_become_zero(self, value):
        """An absent cell means "nobody did any", not "unknown"."""
        assert safe_to_int(value) == 0

    def test_digits_are_rescued_from_surrounding_junk(self):
        """The bare `except` falls back to a regex, which is the only reason a
        cell like '42 units' does not become 0. That is load-bearing, so it is
        pinned rather than left as an implementation detail."""
        assert safe_to_int("42 units") == 42
        assert safe_to_int("qty: 17") == 17

    def test_text_with_no_digits_is_zero(self):
        assert safe_to_int("n/a") == 0
        assert safe_to_int("--") == 0

    def test_negative_is_preserved_not_absolute(self):
        assert safe_to_int("adjustment -8") == -8


class TestSortByLastInitial:
    def test_sorts_on_the_last_character_first(self):
        """The report is read by surname, so 'JA' (A-something) must come
        before 'AB' (B-something) — plain alphabetical would invert it."""
        assert sort_by_last_initial(["AB", "JA"]) == ["JA", "AB"]

    def test_ties_on_last_char_fall_back_to_the_rest(self):
        assert sort_by_last_initial(["ZA", "BA", "MA"]) == ["BA", "MA", "ZA"]

    def test_is_case_insensitive(self):
        assert sort_by_last_initial(["ab", "JA"]) == ["JA", "ab"]

    def test_empty_string_does_not_raise(self):
        assert sort_by_last_initial(["", "AB"]) == ["", "AB"]

    def test_empty_list(self):
        assert sort_by_last_initial([]) == []


class TestFindSimilarInitials:
    def test_prefix_match_counts_as_similar(self):
        assert "JA" in find_similar_initials("JAM", ["JA", "ZZ"])

    def test_single_character_difference_at_equal_length(self):
        assert "JAM" in find_similar_initials("JAN", ["JAM"])

    def test_two_differences_is_not_similar(self):
        assert find_similar_initials("JAN", ["ZQN"]) == []

    def test_case_insensitive(self):
        assert "ja" in find_similar_initials("JAM", ["ja"])

    def test_empty_known_entries_are_skipped_not_matched(self):
        """Without the guard, '' is a prefix of everything and every unknown
        initial would look similar to a blank config entry."""
        assert find_similar_initials("JAM", ["", "ZZ"]) == []


class TestRegionFromEmail:
    def test_matches_the_configured_pattern(self):
        assert region_from_email("bob@north.example.com", CONFIG) == "North"

    def test_is_case_and_whitespace_insensitive(self):
        assert region_from_email("  BOB@NORTH.EXAMPLE.COM ", CONFIG) == "North"

    def test_unknown_domain_is_none_not_a_default_region(self):
        """Guessing a region here would silently attribute one site's work to
        another, which is exactly the number the report exists to state."""
        assert region_from_email("bob@elsewhere.com", CONFIG) is None

    @pytest.mark.parametrize("value", [None, 42, float("nan")])
    def test_non_string_input_is_none(self, value):
        assert region_from_email(value, CONFIG) is None


class TestConsigneeTokens:
    def test_extracts_alpha_runs_uppercased(self):
        assert consignee_tokens_in_order("acme corp") == ["ACME", "CORP"]

    def test_digits_and_punctuation_are_separators(self):
        assert consignee_tokens_in_order("ACME-123 LLC") == ["ACME", "LLC"]

    def test_non_string_is_empty(self):
        assert consignee_tokens_in_order(None) == []
        assert consignee_tokens_in_order(7) == []

    def test_empty_string_is_empty(self):
        assert consignee_tokens_in_order("") == []
