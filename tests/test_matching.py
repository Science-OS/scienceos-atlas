from __future__ import annotations

from atlas.matching import match_authors, name_similarity


def au(i, name):
    return {"id": i, "name": name, "orcid": None, "institutions": []}


def test_similarity_cases():
    assert name_similarity("Steve Lawford", "Lawford, Steve") == 1.0
    assert name_similarity("S. Lawford", "Steve Lawford") >= 0.9
    assert name_similarity("José García", "Jose Garcia") == 1.0
    assert name_similarity("Wei Li", "Li Wei") == 1.0
    assert name_similarity("Anna Smith", "Bob Jones") < 0.5
    assert name_similarity("", "x") == 0.0


def test_positional_match_and_unmatched():
    oai = ["Steve Lawford", "A. Nonexistent", "Zed Zed"]
    oa = [au("A1", "Steve Lawford"), au("A2", "Totally Different"), au("A3", "Zed Zed")]
    m = match_authors(oai, oa)
    assert {k: v["id"] for k, v in m.items()} == {0: "A1", 2: "A3"}


def test_reordered_authors_match_by_name_elsewhere():
    oai = ["Ada Lovelace", "Alan Turing"]
    oa = [au("A2", "Alan Turing"), au("A1", "Ada Lovelace")]
    m = match_authors(oai, oa)
    assert {k: v["id"] for k, v in m.items()} == {0: "A1", 1: "A2"}


def test_null_openalex_ids_are_never_attached():
    m = match_authors(["Mido Assran"], [au(None, "Mido Assran")])
    assert m == {}


def test_one_authorship_is_not_reused():
    m = match_authors(["John Smith", "John Smith"], [au("A1", "John Smith")])
    assert len(m) == 1
