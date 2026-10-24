from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from atlas.media.queue import CURSOR_META, QueueRow, QueueUnavailable, advance_cursor, fetch_rows
from atlas.media.cascade import Cascade
from atlas.media.stage import Target, institution_target, process_targets, run_media, state_targets
from atlas.media.store import MediaStore
from atlas.models import ts
from atlas.store import Store
from media_fakes import BASE, ETH, PAGE, FakeNet, make_cfg, make_http

NOW = datetime(2026, 10, 24, 6, 0, tzinfo=timezone.utc)
ROR_ETH = {"ror": ETH["ror"], "openalex": "I1", "name": "ETH Zurich", "country": "CH", "type": "education"}


def card(oid, name, orcid=None, insts=()):
    return {"id": oid, "name": name, "alternatives": [], "orcid": orcid, "institutions": list(insts)}


def seed_state() -> Store:
    st = Store(":memory:")
    work = {"authorships": [
        {"id": PAGE["openalex"], "name": PAGE["name"], "orcid": PAGE["orcid"], "institutions": [ROR_ETH]},
        {"id": "A2", "name": "Nobody Known", "orcid": None, "institutions": []},
        {"id": "A3", "name": "Gone Missing", "orcid": None, "institutions": []},
    ]}
    st.set_oa_work("2610.00001", "found", work, ts(NOW))
    st.upsert_authors({PAGE["openalex"]: card(PAGE["openalex"], PAGE["name"], PAGE["orcid"], [ROR_ETH]), "A2": card("A2", "Nobody Known"),
                       "A3": {"id": "A3", "missing": True}}, ts(NOW))
    return st


@pytest.fixture
def env(tmp_path):
    cfg, net = make_cfg(), FakeNet()
    return cfg, net, make_http(cfg, net), MediaStore(tmp_path / "ms", BASE), seed_state()


def run(env, feed, now=NOW):
    cfg, net, http, media, st = env
    return run_media(cfg, st, media, http, feed, now)


# ---------------------------------------------------------------- targets from the state


def test_state_targets_authors_then_institutions_deduped_and_missing_cards_skipped():
    st = seed_state()
    ts_ = state_targets(st, 25)
    assert [(t.kind, t.keys) for t in ts_] == [
        ("author", [f"orcid-{PAGE['orcid']}", f"openalex-{PAGE['openalex']}"]), ("author", ["openalex-A2"]), ("institution", ["ror-05a28rw58"])]
    assert ts_[0].name == PAGE["name"]


def test_state_targets_follow_the_given_paper_order_and_limit_authors_per_paper():
    st = seed_state()
    st.set_oa_work("2610.00002", "found", {"authorships": [{"id": "A9", "institutions": []}]}, ts(NOW))
    st.upsert_authors({"A9": card("A9", "Nine")}, ts(NOW))
    assert state_targets(st, 25, ["2610.00002", "2610.00001"])[0].keys == ["openalex-A9"]
    assert [t.keys for t in state_targets(st, 1, ["2610.00001"]) if t.kind == "author"] == [[f"orcid-{PAGE['orcid']}", f"openalex-{PAGE['openalex']}"]]
    assert state_targets(st, 25, ["2610.00404"]) == []                                 # unknown papers contribute nothing


def test_orphan_cards_are_included_only_for_a_backfill():
    st = seed_state()
    st.upsert_authors({"A7": card("A7", "Orphan", insts=[{"ror": "02mhbdp94", "name": "Other U"}])}, ts(NOW))
    assert "openalex-A7" not in [k for t in state_targets(st, 25) for k in t.keys]
    all_ = state_targets(st, 25, include_orphans=True)
    assert "openalex-A7" in [k for t in all_ for k in t.keys] and "ror-02mhbdp94" in [k for t in all_ for k in t.keys]


def test_targets_without_a_usable_id_are_dropped():
    assert institution_target({"name": "No ROR"}).keys == []


# ---------------------------------------------------------------- the queue


def test_queue_request_shape_and_publishable_key_only():
    cfg, net = make_cfg(), FakeNet()
    http = make_http(cfg, net)
    net.queue_rows = [{"id": 5, "entity_kind": "author", "entity_key": "wikidata-Q1", "created_at": "2026-10-24T00:00:00Z"}]
    rows = fetch_rows(cfg.media.queue, 4, 500, http.queue)
    req = net.requests[0]
    assert req.url.host == "rwkduugxjvzrniomfjri.supabase.co" and req.url.path == "/rest/v1/media_requests" and req.method == "GET"
    assert dict(req.url.params) == {"select": "id,entity_kind,entity_key,created_at", "id": "gt.4", "order": "id.asc", "limit": "500"}
    assert req.headers["apikey"].startswith("sb_publishable_") and req.headers["accept-profile"] == "social" and "authorization" not in req.headers
    assert rows == [QueueRow(5, "author", "wikidata-Q1")] and rows[0].valid


def test_a_legacy_anon_jwt_key_is_also_sent_as_bearer():
    cfg, net = make_cfg(), FakeNet()
    cfg.media.queue.publishable_key = "eyJhbGciOi.fake.jwt"
    fetch_rows(cfg.media.queue, 0, 10, make_http(cfg, net).queue)
    assert net.requests[0].headers["authorization"] == "Bearer eyJhbGciOi.fake.jwt"


def test_queue_unavailable_when_unconfigured_or_not_created_yet():
    cfg, net = make_cfg(), FakeNet()
    http = make_http(cfg, net)
    net.queue_status = 404                                      # the table does not exist yet
    with pytest.raises(QueueUnavailable) as ei:
        fetch_rows(cfg.media.queue, 0, 10, http.queue)
    assert str(ei.value) == "HTTP 404"                          # the response body is not echoed
    cfg.media.queue.url = ""
    with pytest.raises(QueueUnavailable):
        fetch_rows(cfg.media.queue, 0, 10, http.queue)


def test_malformed_rows_are_flagged_invalid_or_ignored():
    cfg, net = make_cfg(), FakeNet()
    net.queue_raw = [{"id": 1, "entity_kind": "author", "entity_key": "../../etc/passwd"}, {"id": 2, "entity_kind": "paper", "entity_key": "arxiv-2610.1"},
                      {"id": 3, "entity_kind": "institution", "entity_key": "orcid-0000-0002-1992-2684"}, {"id": 4, "entity_kind": "author", "entity_key": "wikidata-Q5"},
                      {"id": "x", "entity_kind": "author", "entity_key": "wikidata-Q5"}, {"entity_kind": "author"}]
    rows = fetch_rows(cfg.media.queue, 0, 50, make_http(cfg, net).queue)
    assert [(r.id, r.valid) for r in rows] == [(1, False), (2, False), (3, False), (4, True)]    # rows without an integer id are ignored


def test_cursor_stops_at_the_first_row_not_handled():
    rows = [QueueRow(i, "author", "wikidata-Q1") for i in (3, 4, 5, 6)]
    assert advance_cursor(rows, {3, 4, 6}, 2) == 4              # 5 was not reached, so 6 must be seen again
    assert advance_cursor(rows, {3, 4, 5, 6}, 2) == 6
    assert advance_cursor(rows, set(), 2) == 2 and advance_cursor([], set(), 7) == 7


def test_queue_budget_and_cursor_across_two_runs(env):
    cfg, net, http, media, st = env
    cfg.media.max_new_per_run = 2
    net.queue_rows = [{"id": 1, "entity_kind": "author", "entity_key": "wikidata-Q7356570"},
                      {"id": 2, "entity_kind": "author", "entity_key": "../../x"},
                      {"id": 3, "entity_kind": "institution", "entity_key": "ror-05a28rw58"},
                      {"id": 4, "entity_kind": "author", "entity_key": "orcid-0000-0000-0000-0001"}]
    s1 = run(env, [])
    assert s1["processed_author"] == 1 and s1["processed_institution"] == 1 and s1["budget_exhausted"]
    assert s1["queue"] == {"rows": 4, "invalid": 1, "cursor_before": 0, "cursor_after": 3, "pending": 1}
    assert st.get_meta(CURSOR_META) == "3"
    assert media.read_entry("author", "wikidata-Q7356570") and media.read_entry("institution", "ror-05a28rw58")
    s2 = run(env, [])
    assert s2["queue"]["rows"] == 1 and s2["queue"]["cursor_after"] == 4 and s2["processed_author"] == 1 and not s2["budget_exhausted"]
    assert st.get_meta(CURSOR_META) == "4"
    s3 = run(env, [])
    assert "queue" not in s3 or s3["queue"].get("rows", 0) == 0                          # nothing left after the cursor


def test_queue_failure_does_not_stop_the_feed(env):
    cfg, net, http, media, st = env
    net.queue_status = 404
    s = run(env, state_targets(st, 25))
    assert s["queue"] == {"unavailable": "HTTP 404"} and s["processed_author"] >= 1 and st.get_meta(CURSOR_META) is None


def test_client_keys_are_only_ids_everything_is_resolved_by_the_pipeline(env):
    cfg, net, http, media, st = env
    net.queue_rows = [{"id": 1, "entity_kind": "author", "entity_key": "wikidata-Q7356570",
                       "photo": "https://evil.example/me.png", "image_url": "https://evil.example/me.png"}]
    run(env, [])
    a = media.read_entry("author", "wikidata-Q7356570")["assets"]
    assert a[0]["hosted"] and "evil.example" not in str(a) and not [r for r in net.requests if r.url.host == "evil.example"]


def test_queue_and_feed_are_served_alternately(env):
    cfg, net, http, media, st = env
    cfg.media.max_new_per_run = 2
    net.queue_rows = [{"id": 1, "entity_kind": "institution", "entity_key": "ror-05a28rw58"},
                      {"id": 2, "entity_kind": "author", "entity_key": "orcid-0000-0000-0000-0001"}]
    feed = [Target("author", {"openalex": "A2"}, name="Nobody Known"), Target("author", {"openalex": "A5"}, name="Five")]
    run(env, feed)
    done = {k for k in ("institution:ror-05a28rw58", "author:orcid-0000-0000-0000-0001", "author:openalex-A2", "author:openalex-A5")
            if st.media_checks(k.split(":")[0], [k.split(":", 1)[1]])}
    assert done == {"institution:ror-05a28rw58", "author:openalex-A2"}                 # first of each source, neither starves


# ---------------------------------------------------------------- TTLs, budgets, retries


def feed_for(st):
    return state_targets(st, 25)


def test_ttls_photo_30_days_logo_180_days_nothing_found_7_days(env):
    cfg, net, http, media, st = env
    run(env, feed_for(st))
    a = st.media_checks("author", [f"orcid-{PAGE['orcid']}"])[f"orcid-{PAGE['orcid']}"]
    i = st.media_checks("institution", ["ror-05a28rw58"])["ror-05a28rw58"]
    n = st.media_checks("author", ["openalex-A2"])["openalex-A2"]
    assert (a["status"], a["expires_at"]) == ("ok", ts(NOW + timedelta(days=30)))
    assert (i["status"], i["expires_at"]) == ("ok", ts(NOW + timedelta(days=180)))
    assert (n["status"], n["expires_at"]) == ("negative", ts(NOW + timedelta(days=7)))


def sources(net):
    """Requests to the media sources (not the one cheap queue read every run makes)."""
    return [r for r in net.requests if not r.url.path.startswith("/rest/v1/")]


def test_everything_fresh_means_zero_source_requests(env):
    cfg, net, http, media, st = env
    run(env, feed_for(st))
    before = len(sources(net))
    s = run(env, feed_for(st), NOW + timedelta(hours=20))
    assert len(sources(net)) == before and s["fresh"] == 3 and s.get("processed_author", 0) == 0


def test_negative_results_are_rechecked_after_7_days_found_ones_after_30(env):
    cfg, net, http, media, st = env
    run(env, feed_for(st))
    s6 = run(env, feed_for(st), NOW + timedelta(days=6))
    assert s6.get("processed_author", 0) == 0
    s8 = run(env, feed_for(st), NOW + timedelta(days=8))
    assert s8["processed_author"] == 1 and s8["fresh"] == 2                              # only the negative author A2
    s31 = run(env, feed_for(st), NOW + timedelta(days=31))
    assert s31["processed_author"] == 2 and s31["fresh"] == 1                            # photo author and A2 again; the logo lives 180 d


def test_per_run_budget_never_resolves_more_than_max_new_per_run_and_never_loses_the_rest(env):
    cfg, net, http, media, st = env
    cfg.media.max_new_per_run = 1
    for hour in range(3):                                            # 3 entities, 1 per run: the backlog drains one by one
        s = run(env, feed_for(st), NOW + timedelta(hours=hour))
        assert s.get("processed_author", 0) + s.get("processed_institution", 0) == 1
        assert s["budget_exhausted"] is (hour < 2)
        assert st.conn.execute("SELECT COUNT(DISTINCT kind || key) FROM media_checks").fetchone()[0] >= hour + 1
    s4 = run(env, feed_for(st), NOW + timedelta(hours=3))
    assert s4["fresh"] == 3 and not s4["budget_exhausted"] and s4.get("processed_author", 0) == 0


def test_the_wall_clock_cap_stops_the_stage_and_loses_nothing(env):
    cfg, net, http, media, st = env
    cascade = Cascade(cfg, http, media, ts(NOW), {})
    ticks = iter([0.0, 10.0, 200.0, 300.0, 400.0])                       # start, first check, then past the 100 s cap
    done = process_targets(cfg, st, media, cascade, feed_for(st), NOW, None, 100, clock=lambda: next(ticks))
    assert done.budget_exhausted and done.stats["processed_author"] + done.stats.get("processed_institution", 0) == 1
    s2 = run(env, feed_for(st), NOW + timedelta(hours=1))
    assert s2["fresh"] == 1 and s2["processed_author"] + s2.get("processed_institution", 0) == 2


def test_never_resolved_entities_go_before_expired_ones(env):
    cfg, net, http, media, st = env
    run(env, feed_for(st))
    cfg.media.max_new_per_run = 1
    fresh_one = Target("author", {"openalex": "A55"}, name="Brand New")
    s = run(env, [*feed_for(st), fresh_one], NOW + timedelta(days=8))                 # A2 is stale, A55 was never seen
    assert s["processed_author"] == 1 and st.media_checks("author", ["openalex-A55"])


def test_a_failed_entity_is_retried_the_next_day_without_anyone_asking_again(env):
    cfg, net, http, media, st = env
    net.down["www.wikidata.org"] = 503
    s1 = run(env, [Target("author", {"orcid": PAGE["orcid"]}, name=PAGE["name"])])
    assert s1["errors"] == 1 and not media.read_entry("author", f"orcid-{PAGE['orcid']}")
    chk = st.media_checks("author", [f"orcid-{PAGE['orcid']}"])[f"orcid-{PAGE['orcid']}"]
    assert chk["status"] == "error" and chk["expires_at"] == ts(NOW + timedelta(hours=24))
    del net.down["www.wikidata.org"]
    assert run(env, [], NOW + timedelta(hours=2)).get("retries") == 0                  # not before 24 h
    s2 = run(env, [], NOW + timedelta(hours=25))
    assert s2["retries"] == 1 and s2["processed_author"] == 1 and s2["found"] == 1
    assert media.read_entry("author", f"orcid-{PAGE['orcid']}")


def test_a_partial_result_is_kept_and_looked_at_again_in_a_day(env):
    cfg, net, http, media, st = env
    net.down["public.api.bsky.app"] = 503
    run(env, [Target("author", {"orcid": PAGE["orcid"]}, name=PAGE["name"])])
    e = media.read_entry("author", f"orcid-{PAGE['orcid']}")
    assert [a["source"] for a in e["assets"]] == ["wikidata_commons", "github"]
    chk = st.media_checks("author", [f"orcid-{PAGE['orcid']}"])[f"orcid-{PAGE['orcid']}"]
    assert chk["status"] == "partial" and chk["expires_at"] == ts(NOW + timedelta(hours=24))
    del net.down["public.api.bsky.app"]
    run(env, [], NOW + timedelta(hours=25))
    assert [a["source"] for a in media.read_entry("author", f"orcid-{PAGE['orcid']}")["assets"]] == ["wikidata_commons", "github", "bluesky"]


def test_every_alias_key_is_marked_checked_so_other_keys_do_not_cost_requests(env):
    cfg, net, http, media, st = env
    run(env, [Target("author", {"openalex": PAGE["openalex"]}, name=PAGE["name"])])
    assert set(st.media_checks("author", ["openalex-A5068055706", "orcid-" + PAGE["orcid"], "wikidata-Q7356570"])) == {
        "openalex-A5068055706", "orcid-" + PAGE["orcid"], "wikidata-Q7356570"}
    before = len(net.requests)
    net.queue_rows = [{"id": 1, "entity_kind": "author", "entity_key": "wikidata-Q7356570"}]
    s = run(env, [], NOW + timedelta(hours=1))
    assert len(net.requests) == before + 1 and s["fresh"] == 1 and s["queue"]["cursor_after"] == 1            # only the queue read itself
    for key in ("orcid-" + PAGE["orcid"], "wikidata-Q7356570", "openalex-A5068055706"):
        assert media.read_entry("author", key)["assets"][0]["hosted"]


def test_a_queue_row_for_an_entity_already_done_this_run_is_handled_not_repeated(env):
    cfg, net, http, media, st = env
    net.queue_rows = [{"id": 1, "entity_kind": "author", "entity_key": "orcid-" + PAGE["orcid"]}]
    s = run(env, [Target("author", {"orcid": PAGE["orcid"]}, name=PAGE["name"])])
    assert s["processed_author"] == 1 and s["queue"]["cursor_after"] == 1 and s.get("deduped", 0) + s.get("fresh", 0) == 1


def test_the_state_card_name_guards_reverse_lookups(env):
    cfg, net, http, media, st = env
    s = run(env, [Target("author", {"orcid": PAGE["orcid"]}, name="Somebody Else Entirely")])
    assert s["negative"] == 1 and s["notes"] == {"name_mismatch": 1} and not media.read_entry("author", f"orcid-{PAGE['orcid']}")


def test_nothing_is_written_for_entities_without_assets(env):
    cfg, net, http, media, st = env
    run(env, [Target("author", {"openalex": "A2"}, name="Nobody Known")])
    assert not (media.root / "c").exists()
    assert st.media_checks("author", ["openalex-A2"])["openalex-A2"]["status"] == "negative"


def test_etags_are_persisted_in_the_state(env):
    cfg, net, http, media, st = env
    run(env, [Target("author", {"orcid": PAGE["orcid"]}, name=PAGE["name"])])
    assert list(st.media_etags().values()) == ['W/"v1"']
    run(env, [Target("author", {"orcid": PAGE["orcid"]}, name=PAGE["name"])], NOW + timedelta(days=31))
    assert [r.headers.get("if-none-match") for r in net.hits("upload.wikimedia.org")] == [None, 'W/"v1"']


def test_config_pins_the_budgets_ttls_and_queue_of_the_brief():
    from atlas.config import MediaCfg, load_config
    for m in (load_config().media, MediaCfg()):                                  # config.yaml and the code defaults agree
        assert (m.max_new_per_run, m.requests_per_run, m.max_site_mb) == (300, 500, 250)
        assert (m.photo_ttl_days, m.logo_ttl_days, m.negative_ttl_days) == (30, 180, 7)
        assert (m.max_px, m.webp_quality, m.homepage_min_icon_px, m.favicon_min_px) == (256, 80, 120, 64)
    q = load_config().media.queue
    assert q.url == "https://rwkduugxjvzrniomfjri.supabase.co" and q.publishable_key.startswith("sb_publishable_")
    assert (q.schema_name, q.table) == ("social", "media_requests")
    assert load_config().output.max_site_mb == 450
