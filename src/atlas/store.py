"""SQLite state (the thing actions/cache carries between runs). Every write commits immediately, so a
crashed run keeps everything it already fetched and the next run resumes from there."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .arxiv import ApiEntry, OaiRecord
from .hf import HfSignal

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS papers(
  arxiv_id TEXT PRIMARY KEY,
  feed_date TEXT NOT NULL,
  datestamp TEXT NOT NULL,
  record TEXT NOT NULL,
  api TEXT NOT NULL,
  first_seen TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS papers_feed_date ON papers(feed_date);
CREATE TABLE IF NOT EXISTS hf(arxiv_id TEXT PRIMARY KEY, signal TEXT NOT NULL, fetched_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS oa_works(
  arxiv_id TEXT PRIMARY KEY, status TEXT NOT NULL, work TEXT, attempts INTEGER NOT NULL,
  last_attempt TEXT NOT NULL, fetched_at TEXT
);
CREATE TABLE IF NOT EXISTS authors(oa_id TEXT PRIMARY KEY, card TEXT NOT NULL, fetched_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS briefs(
  arxiv_id TEXT PRIMARY KEY, text TEXT NOT NULL, source TEXT NOT NULL, model TEXT,
  generated_at TEXT NOT NULL, cost_usd REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS rejected(arxiv_id TEXT PRIMARY KEY, datestamp TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS thumbs(arxiv_id TEXT PRIMARY KEY, status TEXT NOT NULL, bytes INTEGER, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS media_checks(
  kind TEXT NOT NULL, key TEXT NOT NULL, status TEXT NOT NULL, checked_at TEXT NOT NULL, expires_at TEXT NOT NULL,
  PRIMARY KEY(kind, key)
);
CREATE TABLE IF NOT EXISTS media_etags(url TEXT PRIMARY KEY, etag TEXT NOT NULL, updated_at TEXT NOT NULL);
"""


@dataclass
class StoredPaper:
    record: OaiRecord
    api: ApiEntry
    feed_date: str


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # -- meta ------------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def is_empty(self) -> bool:
        return self.conn.execute("SELECT 1 FROM papers LIMIT 1").fetchone() is None

    # -- papers ----------------------------------------------------------------
    def has_paper(self, arxiv_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM papers WHERE arxiv_id=?", (arxiv_id,)).fetchone() is not None

    def known_ids(self, ids: Iterable[str]) -> set[str]:
        ids = list(ids)
        known: set[str] = set()
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            q = f"SELECT arxiv_id FROM papers WHERE arxiv_id IN ({','.join('?' * len(chunk))})"
            known.update(r["arxiv_id"] for r in self.conn.execute(q, chunk))
        return known

    def add_paper(self, rec: OaiRecord, api: ApiEntry, feed_date: str, now: str) -> bool:
        with self.conn:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO papers(arxiv_id,feed_date,datestamp,record,api,first_seen) VALUES(?,?,?,?,?,?)",
                (rec.arxiv_id, feed_date, rec.datestamp, json.dumps(rec.to_json()), json.dumps(api.__dict__), now),
            )
        return cur.rowcount > 0

    def papers_between(self, date_from: str, date_to: str) -> list[StoredPaper]:
        rows = self.conn.execute(
            "SELECT record, api, feed_date FROM papers WHERE feed_date BETWEEN ? AND ? ORDER BY feed_date, arxiv_id",
            (date_from, date_to),
        ).fetchall()
        return [
            StoredPaper(OaiRecord.from_json(json.loads(r["record"])), ApiEntry(**json.loads(r["api"])), r["feed_date"])
            for r in rows
        ]

    def get_paper(self, arxiv_id: str) -> StoredPaper | None:
        r = self.conn.execute("SELECT record, api, feed_date FROM papers WHERE arxiv_id=?", (arxiv_id,)).fetchone()
        if r is None:
            return None
        return StoredPaper(OaiRecord.from_json(json.loads(r["record"])), ApiEntry(**json.loads(r["api"])), r["feed_date"])

    def count_papers(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]

    def feed_dates(self) -> list[str]:
        return [r["feed_date"] for r in self.conn.execute("SELECT DISTINCT feed_date FROM papers ORDER BY feed_date")]

    # -- rejected (OAI records that were revisions of old papers, not new ones) -----
    def rejected(self, ids: Iterable[str]) -> dict[str, str]:
        ids = list(ids)
        out: dict[str, str] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            q = f"SELECT arxiv_id, datestamp FROM rejected WHERE arxiv_id IN ({','.join('?' * len(chunk))})"
            out.update({r["arxiv_id"]: r["datestamp"] for r in self.conn.execute(q, chunk)})
        return out

    def is_rejected(self, arxiv_id: str, datestamp: str) -> bool:
        row = self.conn.execute("SELECT datestamp FROM rejected WHERE arxiv_id=?", (arxiv_id,)).fetchone()
        return row is not None and row["datestamp"] == datestamp

    def add_rejected(self, rows: Iterable[tuple[str, str]]) -> None:
        with self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO rejected(arxiv_id,datestamp) VALUES(?,?)", list(rows))

    # -- hf --------------------------------------------------------------------
    def upsert_hf(self, signals: Iterable[HfSignal], now: str) -> int:
        n = 0
        with self.conn:
            for s in signals:
                self.conn.execute(
                    "INSERT INTO hf(arxiv_id,signal,fetched_at) VALUES(?,?,?) "
                    "ON CONFLICT(arxiv_id) DO UPDATE SET signal=excluded.signal, fetched_at=excluded.fetched_at",
                    (s.arxiv_id, json.dumps(s.to_json()), now),
                )
                n += 1
        return n

    def hf_signals(self) -> dict[str, tuple[HfSignal, str]]:
        return {
            r["arxiv_id"]: (HfSignal.from_json(json.loads(r["signal"])), r["fetched_at"])
            for r in self.conn.execute("SELECT arxiv_id, signal, fetched_at FROM hf")
        }

    # -- openalex --------------------------------------------------------------
    def oa_work_rows(self) -> dict[str, dict[str, Any]]:
        out = {}
        for r in self.conn.execute("SELECT arxiv_id,status,work,attempts,last_attempt,fetched_at FROM oa_works"):
            out[r["arxiv_id"]] = {
                "status": r["status"],
                "work": json.loads(r["work"]) if r["work"] else None,
                "attempts": r["attempts"],
                "last_attempt": r["last_attempt"],
                "fetched_at": r["fetched_at"],
            }
        return out

    def set_oa_work(self, arxiv_id: str, status: str, work: dict[str, Any] | None, now: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO oa_works(arxiv_id,status,work,attempts,last_attempt,fetched_at) VALUES(?,?,?,1,?,?) "
                "ON CONFLICT(arxiv_id) DO UPDATE SET status=excluded.status, work=excluded.work, "
                "attempts=oa_works.attempts+1, last_attempt=excluded.last_attempt, fetched_at=excluded.fetched_at",
                (arxiv_id, status, json.dumps(work) if work else None, now, now if status == "found" else None),
            )

    def author_rows(self) -> dict[str, tuple[dict[str, Any], str]]:
        return {
            r["oa_id"]: (json.loads(r["card"]), r["fetched_at"])
            for r in self.conn.execute("SELECT oa_id, card, fetched_at FROM authors")
        }

    def upsert_authors(self, cards: dict[str, dict[str, Any]], now: str) -> None:
        with self.conn:
            for oa_id, card in cards.items():
                self.conn.execute(
                    "INSERT INTO authors(oa_id,card,fetched_at) VALUES(?,?,?) "
                    "ON CONFLICT(oa_id) DO UPDATE SET card=excluded.card, fetched_at=excluded.fetched_at",
                    (oa_id, json.dumps(card), now),
                )

    # -- briefs ----------------------------------------------------------------
    def brief_rows(self) -> dict[str, dict[str, Any]]:
        return {r["arxiv_id"]: dict(r) for r in self.conn.execute("SELECT * FROM briefs")}

    def set_brief(self, arxiv_id: str, text: str, source: str, model: str | None, now: str, cost_usd: float) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO briefs(arxiv_id,text,source,model,generated_at,cost_usd) VALUES(?,?,?,?,?,?)",
                (arxiv_id, text, source, model, now, cost_usd),
            )

    # -- thumbs ----------------------------------------------------------------
    def thumb_rows(self) -> dict[str, dict[str, Any]]:
        return {r["arxiv_id"]: dict(r) for r in self.conn.execute("SELECT * FROM thumbs")}

    def set_thumb(self, arxiv_id: str, status: str, nbytes: int | None, now: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO thumbs(arxiv_id,status,bytes,updated_at) VALUES(?,?,?,?)",
                (arxiv_id, status, nbytes, now),
            )

    # -- media (when each entity of the central media store was last resolved; never pruned) --------------
    def media_checks(self, kind: str, keys: Iterable[str]) -> dict[str, dict[str, Any]]:
        keys = list(keys)
        if not keys:
            return {}
        q = f"SELECT * FROM media_checks WHERE kind=? AND key IN ({','.join('?' * len(keys))})"
        return {r["key"]: dict(r) for r in self.conn.execute(q, [kind, *keys])}

    def set_media_check(self, kind: str, keys: Iterable[str], status: str, checked_at: str, expires_at: str) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO media_checks(kind,key,status,checked_at,expires_at) VALUES(?,?,?,?,?)",
                [(kind, k, status, checked_at, expires_at) for k in keys],
            )

    def media_retries(self, now: str) -> list[tuple[str, str]]:
        """(kind, key) of entities whose last attempt failed or was partial and whose retry time has come."""
        return [(r["kind"], r["key"]) for r in self.conn.execute(
            "SELECT kind, key FROM media_checks WHERE status IN ('error','partial') AND expires_at <= ? ORDER BY expires_at, kind, key", (now,))]

    def media_etags(self) -> dict[str, str]:
        return {r["url"]: r["etag"] for r in self.conn.execute("SELECT url, etag FROM media_etags")}

    def set_media_etags(self, etags: dict[str, str], now: str) -> None:
        with self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO media_etags(url,etag,updated_at) VALUES(?,?,?)", [(u, e, now) for u, e in etags.items()])

    # -- retention -------------------------------------------------------------
    def prune(self, keep_from_date: str) -> list[str]:
        """Drop everything older than keep_from_date. Returns pruned arxiv ids (so thumbs can be deleted)."""
        ids = [r["arxiv_id"] for r in self.conn.execute("SELECT arxiv_id FROM papers WHERE feed_date < ?", (keep_from_date,))]
        with self.conn:
            self.conn.execute("DELETE FROM papers WHERE feed_date < ?", (keep_from_date,))
            for table in ("hf", "oa_works", "briefs", "thumbs"):
                self.conn.execute(f"DELETE FROM {table} WHERE arxiv_id NOT IN (SELECT arxiv_id FROM papers)")
        with self.conn:
            self.conn.execute("DELETE FROM rejected WHERE datestamp < ?", (keep_from_date,))
        # authors no longer referenced by any remaining work
        referenced: set[str] = set()
        for r in self.conn.execute("SELECT work FROM oa_works WHERE work IS NOT NULL"):
            for au in json.loads(r["work"]).get("authorships", []):
                if au.get("id"):
                    referenced.add(au["id"])
        with self.conn:
            for oa_id in [r["oa_id"] for r in self.conn.execute("SELECT oa_id FROM authors")]:
                if oa_id not in referenced:
                    self.conn.execute("DELETE FROM authors WHERE oa_id=?", (oa_id,))
        return ids
