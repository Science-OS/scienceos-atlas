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
  thumbs/<arxivId>.jpg               600 px first page, ONLY for CC BY / CC BY-SA / CC0 papers (kept for older app builds)
  media/                             the central media store: photos, logos, thumbnails (see "Media store")
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
  When the media catalogue knows a photo, `author.photo` is set (a Link `Photo`) and `photo` leaves `pending`;
  `affiliations[].logoUrl` and `institutions[].logo` are filled the same way.
- **Thumbnails** are linked from the paper as `{"kind": "other", "label": "thumbnail", "url": ...}`.
  The link exists only when the file does, and points at the hosted file under `media/b/` once the store has it
  (`thumbs/<id>.jpg` is still published for the current window: older app builds build that path themselves).
  Papers under any other licence (arXiv's non-exclusive licence,
  any NC/ND variant) never get one; apps render the first page on-device from arxiv.org.
- **Briefs**: `brief.source` is `llm` (optional, see below), else `hf_ai_summary` (Hugging Face) when present and `briefs.hf_summary_fallback` is on (off by default),
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
   Default model `gpt-6-luna` with `reasoning_effort: none` (it reasons by default and would spend the output
   budget on hidden tokens). A brief is 1 to 2 plain-language sentences (about 35 words) written from the title
   and abstract only; the abstract is treated as data. Measured: about $0.00007 per paper, so roughly $0.007 for a
   day's top 100; the first run also covers the two previous days (about $0.02).
7. **Media store** (`src/atlas/media/`): thumbnails move into the store, then photos and logos are resolved for
   the feed's authors and institutions plus the client work queue. See "Media store" below.
8. **Build + validate**: shards, papers, authors, thumbs, media, manifest, status; every JSON file is validated against
   the copied schema, manifest hashes and counts are cross-checked, author/thumbnail/media references must resolve,
   and the site must stay under `output.max_site_mb` (450), of which `media/` may use `media.max_site_mb` (250).

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

## Media store

Every researcher photo, institution logo and paper thumbnail Discover shows is resolved **once** here, stored, and reused
by every client instead of being scraped again. Contract: `ScienceOS-Electron/.claude/logs/2026-10-21-mobile-companion/prompts/media-contract.md`
(v1, frozen). Code: `src/atlas/media/`.

```
<site_base_url>/media/
  b/<sha256[0:2]>/<sha256>.<webp|jpg>   immutable bytes. Photos and logos: WebP, longest side <= 256 px, alpha kept. Thumbnails: the 600 px JPEG
  c/<kind>/<key>.json                   catalogue entry; a 404 means unknown        kind = author | institution | paper
  manifest.json                         {version, generatedAt, counts:{author,institution,paper,hostedFiles,linkOnly}, hostedBytes}
```

**Keys** are filename-safe: `orcid-0000-0002-1992-2684`, `wikidata-Q3571662`, `openalex-A5071252540` (authors),
`ror-0190ak572`, `wikidata-Q49210` (institutions), `arxiv-2610.30108` (papers; version stripped, old ids `hep-th_9901001`).
An entity known under several keys gets one file per key; the files are identical except for `key`. `manifest.counts`
counts entities, not alias files. Keys read from the work queue are validated against these patterns before they get near a path.

**Catalogue entry** (`schemas/media.schema.json`): `{version, kind, key, ids, updatedAt, assets[]}`. An asset is
`{role: photo|logo|thumbnail, url, hosted, sha256, mime, width, height, source, sourceUrl, license, licenseUrl, attribution, attributionUrl, confidence, fetchedAt}`;
`url, source, license, attribution, attributionUrl, confidence, fetchedAt` are exactly Link v1 `$defs.Photo`. Assets are ordered best first and
hosted ones come before link-only ones of the same role; clients use the first asset of the role they need.
`source` is one of `wikidata_commons`, `github`, `bluesky`, `homepage_icon`, `favicon`, `arxiv_pdf`. An entity with nothing found has no file.

**What may be re-hosted** (bytes on the `media` branch): Wikimedia Commons files whose `LicenseShortName` is CC0, public domain (`PD-*`, including
PD-textlogo), CC BY or CC BY-SA, always with artist, licence and file-page link (a CC BY / BY-SA file without an artist is not hosted); resized, never cropped.
NC/ND, GFDL-only, "copyrighted free use", NonFree-flagged and licence-less files are never hosted. Paper thumbnails are only those the thumbs
stage already gated to CC BY / CC BY-SA / CC0, with the paper's licence. **Link-only** (catalogued with provenance, bytes never stored): GitHub and Bluesky
avatars, homepage icons, Google faviconV2. Each client keeps its own private byte cache of those.

**Cascades** (identity chains only, never a name search; a name is used only to reject a Wikidata label that clearly disagrees with the one we hold):

- *Authors*: ORCID / Wikidata QID / OpenAlex id, then Wikidata (`haswbstatement:P496` reverse lookup when only an ORCID is known, `P10283` for an OpenAlex id; no result or
  more than one item attaches nothing), then P18, Commons licence gate, hosted photo. Plus P2037 GitHub (a `HEAD` on `github.com/<login>.png`) and P12361 Bluesky
  (`app.bsky.actor.getProfile`) as link-only avatars.
- *Institutions*: ROR record, Wikidata (`P154`, preferred rank first, retired logos last), licence gate, hosted logo. SVG logos come from
  `Special:FilePath/<file>?width=256`, which serves a PNG render. Otherwise a homepage icon of at least 120 px (checked by fetching it; SSRF-guarded), otherwise
  faviconV2 at 128 (accepted when the image is at least 64 px), both link-only.

**What `atlas run` resolves** (after enrichment, `media.*` in `config.yaml`): the authors (any of orcid / openalex) and ROR institutions of the feed, and the rows of
`social.media_requests` newer than the cursor kept in the SQLite state. Retries that failed or came back partial are due again after `error_retry_hours`.
At most `max_new_per_run` (300) entities are resolved per run (and the stage stops after `max_seconds`, 25 minutes), never-resolved ones first, the queue and the feed alternately; at most `requests_per_run` (500) queue rows are read.
The cursor stops at the first row the budget did not reach, so nothing is skipped. TTLs live in the state (`media_checks`): authors 30 days, institutions 180,
nothing found 7. Image re-checks send `If-None-Match` (ETags kept in `media_etags`); a 304 reuses the catalogue asset. A fresh resolution never drops a hosted
file found earlier, and files and bytes are never deleted or rewritten: same bytes, same path.

**Work queue** `social.media_requests` (prod Supabase, schema `social`): clients insert *which entity* they want (`entity_kind`, `entity_key`); the pipeline reads
`id, entity_kind, entity_key, created_at` with the publishable key in `media.queue` and re-resolves everything itself. Nothing a client says is trusted as an image.
Until the table exists the stage logs "queue unavailable" and carries on with the feed.

**Publishing.** The store lives on the append-only branch `media` (git history is the archive; the workflow commits `chore(media): +N authors, +M institutions, ...`
and pushes `HEAD:media`, never a force). The daily build copies it into `site/media/`, minus the JPEGs (and catalogue entries) of papers outside
`feeds.retention_days`, so the published catalogue never points at bytes that are not published. `validate.py` checks every entry against
`schemas/media.schema.json`, that each hosted file exists and matches its SHA-256, that published JSON only points at files that exist, and the size limits.

**Durability of everything scraped.** After each run `state/atlas.sqlite` is gzipped, encrypted (`openssl enc -aes-256-cbc -pbkdf2`, key = Actions secret
`ATLAS_STATE_KEY`) and force-pushed as the single commit of branch `state` (file `atlas.sqlite.gz.enc`). When the Actions cache misses, the next run restores it
first. Without the secret both steps warn and skip; they never fail the run. To open a snapshot by hand:
`git fetch origin state && git show origin/state:atlas.sqlite.gz.enc | openssl enc -d -aes-256-cbc -pbkdf2 -pass env:ATLAS_STATE_KEY | gunzip > atlas.sqlite`.

**Locally**

```bash
uv run atlas run --media-dir /tmp/media-store                   # a normal run with the store in a scratch directory (default: media-store/, git-ignored)
uv run atlas media backfill --from-db state/atlas.sqlite --media-dir /tmp/media-store --limit 30
                                                                # every author and institution already in the state; --limit caps the entities, rerun to continue
uv run atlas media checkout --media-dir media-store             # what CI does first: the media branch as a worktree (orphan branch the first time)
uv run atlas media publish --media-dir media-store --no-push    # commit as chore(media): <counts>; drop --no-push to push HEAD:media
uv run atlas state snapshot --no-push                           # needs ATLAS_STATE_KEY in the environment
uv run atlas state restore --force
```

A backfill writes TTL bookkeeping into the database you point it at; work on a copy of `state/atlas.sqlite` for experiments.

## Running it

```bash
uv sync --extra thumbs          # Python 3.12 (see .python-version)
uv run pytest                   # no network needed
uv run atlas run --limit 50     # smoke test: real APIs, polite, ~2 minutes
uv run atlas run                # a normal daily run
uv run atlas run --offline      # rebuild site/ from state/ only
uv run atlas validate           # validate an existing site/ (including site/media)
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

Around the run, `daily.yml` has four more steps, each a thin call into the CLI: `atlas state restore` (a no-op unless the cache missed),
`atlas media checkout` (branch `media` as the `media-store/` worktree, orphan branch on the first run), `atlas media publish` (commit and push `HEAD:media`) and
`atlas state snapshot`. The build job has `contents: write` for those two branches; the deploy job is unchanged.

If the cache is ever evicted, the next run restores `state/atlas.sqlite` from the encrypted `state` branch. Without that branch (or the secret) it
cold-starts (yesterday plus 2 days) and the 30-day history rebuilds over the following days; shards already published are replaced by the rebuilt site.

### Secrets (all optional)

| Secret | Effect |
|---|---|
| `OPENALEX_API_KEY` | free key from openalex.org/settings/api: $1/day budget instead of $0.10 shared per IP |
| `OPENAI_API_KEY` | turns LLM briefs on (provider `openai`) |
| `ANTHROPIC_API_KEY` | turns LLM briefs on if `briefs.provider: anthropic` |
| `ATLAS_STATE_KEY` | passphrase for the encrypted state snapshot on branch `state`; without it the snapshot and restore steps skip with a warning |

Before changing the briefs model, verify its id, prices and supported `reasoning_effort` values under
`briefs.providers` in `config.yaml` (prices there were checked on 2026-10-22), then do a small `workflow_dispatch`
run. The cost guard stops at `max_cost_usd_per_run` (default $0.25).

## Politeness and licensing

- arXiv: 1 request / 3 s, single connection, honours `Retry-After`; metadata is CC0; PDFs are never re-served,
  and thumbnails exist only for CC BY / CC BY-SA / CC0 papers.
- OpenAlex: CC0; singleton GETs are free; stops on a low budget or 429.
- Hugging Face Daily Papers: unauthenticated, 7 small requests per run; signals, links and `ai_summary` are
  attributed to HF through `source: "hf"` / `hf_ai_summary`. HF's redistribution terms for `ai_summary` were not
  verified, so the fallback is off (`briefs.hf_summary_fallback: false`); `ai_summary` is never published unless it is turned on.
- Wikimedia Commons, Wikidata, ROR: polite (one request at a time per source group, at least `media.min_interval_s` = 0.25 s apart, `Retry-After` honoured, descriptive User-Agent). Commons photos and logos are
  re-hosted only when freely licensed, with credit (see "Media store").
- Not redistributed from here: Semantic Scholar, Google Scholar numbers, GitHub and Bluesky avatars, homepage icons and favicons (catalogued as links; clients keep their own copy).

## Repo layout

```
config.yaml                 every tunable
schemas/                    copied Link v1 schemas (+ repo-local status.schema.json, media.schema.json)
src/atlas/                  arxiv, hf, openalex, scoring, matching, licenses, thumbs, briefs, store (SQLite),
                            build (site assembly), validate, pipeline, snapshot (encrypted state), cli
src/atlas/media/            keys, images (WebP), catalogue, store (content-addressed files), sources + web (Wikidata, Commons,
                            ROR, GitHub, Bluesky, homepages), cascade, queue, stage (budgets, TTLs), papers, publish (site copy, git)
site_static/                files copied verbatim into site/
tests/                      pytest; tests/data holds small real responses captured once (2026-10-22), tests/data/media the Wikidata,
                            Commons, ROR and Bluesky responses of the media cascades (2026-10-24)
.github/workflows/          ci.yml, daily.yml
```
