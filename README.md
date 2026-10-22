# ScienceOS Atlas

A $0, static, daily-updated paper feed for the ScienceOS iPhone and Mac apps.
A GitHub Actions job runs after arXiv's announcement, builds JSON files and publishes them to GitHub Pages.
The apps read the files directly. There is no server, no database and no PDF re-serving.

- Pages: <https://science-os.github.io/scienceos-atlas/>
- Manifest: <https://science-os.github.io/scienceos-atlas/index.json>
- Latest day: <https://science-os.github.io/scienceos-atlas/feeds/latest.json>
- Contract: `schemas/atlas.schema.json` + `schemas/common.schema.json`, copied from ScienceOS Link v1
  (`protocol/link/v1/schemas`). Every published file is validated against them in every run, and a run that
  produces an invalid file does not deploy.

## What gets published

```
site/
  index.json                         FeedManifest: latestDate + every shard (feed, date, path, count, sha256, bytes)
  status.json                        extras the strict manifest cannot carry: days, categories, counts, last run stats
  index.html                         human landing page
  feeds/
    YYYY-MM-DD.json                  FeedShard, feed "today": the day's top papers, ranked          (30 days)
    latest.json                      copy of the newest day (not listed in the manifest)
    trending/YYYY-MM-DD.json         FeedShard, feed "trending": top of the trailing 7 days          (30 days)
    trending/latest.json
    categories/<cat>/YYYY-MM-DD.json FeedShard, feed "cat:<cat>" (e.g. cat:cs.LG)                    (30 days)
    categories/<cat>/latest.json     newest day that has that category
  papers/<arxivId>.json              AtlasPaperFile, top papers of each day
  authors/<openalexId>.json          AtlasAuthorFile, light author card (id without the "oa:" prefix)
  thumbs/<arxivId>.jpg               600 px first page, ONLY for CC BY / CC BY-SA / CC0 papers
```

Clients should use `index.json` and the `path` values in it; `sha256` lets them skip unchanged shards.
Pages serves `ETag` / `Last-Modified`. Shards whose content did not change keep byte-identical content
(their `generatedAt` only moves when the content moves), so conditional requests work across days.

### Conventions worth knowing

- **Paper ids** are `arxiv:<id>`; **author ids** in this feed are `oa:<OpenAlex id>` and the file is
  `authors/<OpenAlex id>.json`. A `PaperAuthor.authorId` is set only when an author file exists for it.
- **Author cards are light** (`enrichment.stage = "A"`, `complete = false`, `pending = [photo, experience, coauthors]`):
  name, aliases, ORCID/OpenAlex ids, last known institutions with ROR, h-index / i10 / citations / works
  **labelled `source: "openalex"`**, top topics. `citationsByYear` is empty on purpose (OpenAlex
  `counts_by_year` is publication-year bucketed, not citing-year). Full profiles are the Mac resolver's job.
- **Thumbnails** are linked from the paper as `{"kind": "other", "label": "thumbnail", "url": ...}`.
  The link exists only when the file does. Papers under any other licence (arXiv's non-exclusive licence,
  any NC/ND variant) never get one; apps render the first page on-device from arxiv.org.
- **Briefs**: `brief.source` is `llm` (optional, see below), else `hf_ai_summary` (Hugging Face) when present,
  else the field is omitted.
- **New papers only.** Revised old papers (a v3 announced today) are not "new" and are excluded. `publishedAt`
  is the v1 submission timestamp, `version` the latest version.
- Optional fields are omitted when unset (never `null`), per the Link v1 convention.

## Pipeline

1. **arXiv OAI-PMH** `ListRecords` (`metadataPrefix=arXiv`), sets `cs, stat, eess, q-bio, econ`, `from` = last
   completed day (first run: yesterday minus 2 days), `resumptionToken` followed, 503 `Retry-After` honoured.
   Records are kept when a category matches `arxiv.categories` (fnmatch patterns in `config.yaml`).
   **One request per 3 s, one connection, across OAI + API + PDF** (a single shared throttle). UA
   `ScienceOS-Atlas/1.0 (+https://github.com/Science-OS/scienceos-atlas)`.
2. **New vs revised.** In the OAI `arXiv` format `created` is the date of the latest version, so a 2021 paper
   revised yesterday looks new. Candidates (id's YYMM is the datestamp's month or the month before) are
   therefore checked against the arXiv export API (`id_list`, 100 per request), whose `published` is the v1
   timestamp. v1 within `new_paper_max_age_days` of the OAI datestamp = new. Rejected ids are remembered.
3. **Hugging Face Daily Papers** (`api/daily_papers?date=`, last 7 days re-fetched because upvotes grow):
   upvotes, GitHub repo + stars, project page, `ai_summary`.
4. **OpenAlex** (CC0): free singleton `works/doi:10.48550/arXiv.<id>` for the top `enrich_top_n` papers of each
   day: author ids, ORCIDs, institutions (ROR), `cited_by_count`. OpenAlex lags arXiv by 1 to 3 days, so
   "missing" papers are retried every 18 h for 7 days. Authors are then looked up in batches
   (`authors?filter=ids.openalex:A1|A2...`, 1 credit per 100). After every response `X-RateLimit-Remaining` is
   read; below `min_remaining_credits`, or on HTTP 429, enrichment stops (the run continues).
   No key is needed. A free key (`OPENALEX_API_KEY`, optional secret) raises the daily budget from $0.10 shared
   per IP to $1.
5. **Thumbnails**: top `thumbs.top_n` papers per day whose licence is CC BY, CC BY-SA or CC0; PDF fetched
   politely, first page rendered with pypdfium2 (Apache-2.0 / BSD; chosen over PyMuPDF, which is AGPL),
   600 px JPEG. At most `max_per_run` new ones per run.
6. **Briefs hook** (`src/atlas/briefs.py`): provider interface with OpenAI (default) and Anthropic
   implementations, cost guard (`max_cost_usd_per_run`, projected worst case before each call), cache by paper
   id in the state DB. **Off unless the provider's key env var exists** (`briefs.enabled: auto`).
7. **Build + validate**: shards, papers, authors, thumbs, manifest, status; every JSON file is validated against
   the copied schema, manifest hashes and counts are cross-checked, author/thumbnail references must resolve,
   and the site must stay under `output.max_site_mb` (450).

Enrichment lag is expected and handled: OpenAlex indexes a paper 1 to 3 days after arXiv, and brand-new author
ids exist in `works` before `authors` can resolve them. Papers are retried for 7 days, unresolved author ids are
cached as "missing" and retried daily, and a `PaperAuthor.authorId` appears only once its author file exists.

Every stage is wrapped: an outage in one source degrades the output instead of failing the run. A harvest
with any failed set (or stopped by `--limit`) does not advance the cursor, so the same window is retried.

## Trending score

Documented here, implemented in `src/atlas/scoring.py`, unit-tested in `tests/test_scoring.py`. Weights and
saturation points live in `config.yaml` (`scoring:`).

```
score = 0.50 * hf + 0.15 * gh + 0.10 * cross + 0.15 * recency + 0.10 * code        (weights sum to 1)

hf       = min(1, ln(1 + upvotes) / ln(1 + 100))            Hugging Face Daily Papers upvotes
gh       = min(1, ln(1 + stars)   / ln(1 + 5000))           GitHub stars of the linked repo (via HF)
cross    = min(1, (n_categories - 1) / 3)                   cross-listing count
recency  = 0.5 ** (age_days / 3)                            age = shard date - v1 submission date
code     = 1 if a GitHub repo is linked on Hugging Face else 0
```

Score is in [0, 1], rounded to 4 decimals. Ties break on newer `publishedAt`, then id. A paper's `reason` is
`trending` when it has at least one HF upvote, otherwise `new`; category shards use `category`.

## Running it

```bash
uv sync --extra thumbs          # Python 3.12 (see .python-version)
uv run pytest                   # no network needed
uv run atlas run --limit 50     # smoke test: real APIs, polite, ~2 minutes
uv run atlas run                # a normal daily run
uv run atlas run --offline      # rebuild site/ from state/ only
uv run atlas validate           # validate an existing site/
```

`state/` (SQLite + thumbnails + run logs) and `site/` are git-ignored. Each run writes
`state/runs/atlas_ingest_<key>_<sha7>_<date>_<time>.log` and a `.meta.json` sidecar with parameters and
per-stage statistics; the workflow uploads them as the `run-logs` artifact and a summary table to the job page.

## GitHub Actions

- `ci.yml`: tests on pushes and PRs.
- `daily.yml`: cron `30 5 * * *` UTC (arXiv announces at 00:00 UTC) and `workflow_dispatch` with `limit` and
  `backfill_days` inputs. It restores `state/` with `actions/cache` (so the harvest cursor and enrichment are
  incremental), runs the pipeline, saves the cache even if validation failed, and deploys `site/` with
  `actions/deploy-pages`. Superseded state caches are deleted. A `keepalive` job makes an empty commit when the
  last commit is older than 45 days, because GitHub pauses scheduled workflows in public repos after 60 idle days.

If the cache is ever evicted, the next run cold-starts (yesterday plus 2 days) and the 30-day history rebuilds
over the following days; shards already published are replaced by the rebuilt site.

### Secrets (all optional)

| Secret | Effect |
|---|---|
| `OPENALEX_API_KEY` | free key from openalex.org/settings/api: $1/day budget instead of $0.10 shared per IP |
| `OPENAI_API_KEY` | turns LLM briefs on (provider `openai`) |
| `ANTHROPIC_API_KEY` | turns LLM briefs on if `briefs.provider: anthropic` |

Before enabling briefs, verify the model ids and prices under `briefs.providers` in `config.yaml`, then do a
small `workflow_dispatch` run. The cost guard stops at `max_cost_usd_per_run` (default $0.25).

## Politeness and licensing

- arXiv: 1 request / 3 s, single connection, honours `Retry-After`; metadata is CC0; PDFs are never re-served,
  and thumbnails exist only for CC BY / CC BY-SA / CC0 papers.
- OpenAlex: CC0; singleton GETs are free; stops on a low budget or 429.
- Hugging Face Daily Papers: unauthenticated, 7 small requests per run; signals, links and `ai_summary` are
  attributed to HF through `source: "hf"` / `hf_ai_summary`. HF's redistribution terms for `ai_summary` were not
  verified; if they turn out to be restrictive, remove the `ai_summary` fallback in `src/atlas/build.py`.
- Not redistributed from here: Semantic Scholar, Google Scholar numbers, avatars (clients fetch or hotlink those).

## Repo layout

```
config.yaml                 every tunable
schemas/                    copied Link v1 schemas (+ repo-local status.schema.json)
src/atlas/                  arxiv, hf, openalex, scoring, matching, licenses, thumbs, briefs, store (SQLite),
                            build (site assembly), validate, pipeline, cli
site_static/                files copied verbatim into site/
tests/                      pytest; tests/data holds small real responses captured once (2026-10-22)
.github/workflows/          ci.yml, daily.yml
```
