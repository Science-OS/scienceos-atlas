from __future__ import annotations

import pytest

from atlas.media import keys as K


def test_orcid_accepts_urls_prefixes_and_lowercase_check_digit():
    assert K.orcid_id("https://orcid.org/0000-0002-1992-2684") == "0000-0002-1992-2684"
    assert K.orcid_id("orcid:0000-0002-1992-2684") == "0000-0002-1992-2684"
    assert K.orcid_id("0000-0002-1825-009x") == "0000-0002-1825-009X"
    assert K.orcid_id("not an orcid") is None and K.orcid_id(None) is None


def test_other_id_forms():
    assert K.wikidata_id("https://www.wikidata.org/entity/Q3571662") == "Q3571662"
    assert K.wikidata_id("wd:q42") == "Q42" and K.wikidata_id("P18") is None
    assert K.openalex_id("https://openalex.org/A5071252540") == "A5071252540"
    assert K.openalex_id("oa:a5071252540") == "A5071252540" and K.openalex_id("W123") is None
    assert K.ror_id("https://ror.org/0190AK572") == "0190ak572" and K.ror_id("ror:0190ak572") == "0190ak572"


def test_author_keys_best_first_and_unusable_ids_dropped():
    ids = {"openalex": "https://openalex.org/A5071252540", "orcid": "0000-0002-1992-2684", "wikidata": "Q3571662"}
    assert K.author_keys(ids) == ["orcid-0000-0002-1992-2684", "wikidata-Q3571662", "openalex-A5071252540"]
    assert K.author_keys({"orcid": "nope", "openalex": "A1"}) == ["openalex-A1"]
    assert K.author_keys({}) == []


def test_institution_keys():
    assert K.institution_keys({"wikidata": "Q49210", "ror": "https://ror.org/0190ak572"}) == ["ror-0190ak572", "wikidata-Q49210"]
    assert K.institution_keys({"ror": None}) == []


@pytest.mark.parametrize("raw,key", [
    ("2610.30108", "arxiv-2610.30108"),
    ("2610.30108v3", "arxiv-2610.30108"),
    ("arXiv:2610.30108v1", "arxiv-2610.30108"),
    ("hep-th/9901001", "arxiv-hep-th_9901001"),
    ("math.GT/0309136v2", "arxiv-math.GT_0309136"),
])
def test_paper_key_strips_version_and_flattens_old_ids(raw, key):
    assert K.paper_key(raw) == key
    assert K.valid_key("paper", key)


@pytest.mark.parametrize("kind,key,ok", [
    ("author", "orcid-0000-0002-1992-2684", True), ("author", "wikidata-Q3571662", True), ("author", "openalex-A5071252540", True),
    ("author", "ror-0190ak572", False), ("institution", "ror-0190ak572", True), ("institution", "wikidata-Q49210", True),
    ("institution", "orcid-0000-0002-1992-2684", False),
    ("author", "../../etc/passwd", False), ("author", "wikidata-Q1/../x", False), ("author", "wikidata-Q1.json", False),
    ("author", "orcid-0000-0002-1992-268", False), ("institution", "ror-ABC", False), ("author", "", False),
    ("author", "openalex-a5071252540", False),                 # keys are canonical: normalise first
    ("thing", "wikidata-Q1", False),
])
def test_valid_key_is_strict_so_queue_rows_cannot_reach_the_filesystem(kind, key, ok):
    assert K.valid_key(kind, key) is ok


def test_ids_from_keys_round_trip():
    keys = K.author_keys({"orcid": "0000-0002-1992-2684", "openalex": "A5071252540"})
    assert K.ids_from_keys("author", keys) == {"orcid": "0000-0002-1992-2684", "openalex": "A5071252540"}
    assert K.ids_from_keys("author", ["bogus", "wikidata-Q1"]) == {"wikidata": "Q1"}
