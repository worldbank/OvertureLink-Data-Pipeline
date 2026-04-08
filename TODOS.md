# TODOS

Deferred work from the 2026-04-08 CEO review (Hardening Pass).
CEO plan lives at `~/.gstack/projects/worldbank-OvertureLink-Data-Pipeline/ceo-plans/2026-04-08-pipeline-hardening-pass.md`.

## P1 — Major features (after hardening pass ships)

### Interactive CLI (`o2agol interactive`)
**What:** Wizard-driven CLI built on `questionary`. Pick mode → country (fuzzy search 176) → theme → filters → confirm → run. Under the hood calls the same Typer commands, no logic duplication.
**Why:** The tool is usable today only if you memorize flags; an interactive mode unblocks non-CLI-native operators.
**Pros:** Zero impact on existing commands; attracts casual users.
**Cons:** New surface to test; `questionary` is a new dep.
**Context:** Explicitly deferred in 2026-04-08 CEO review. User said "focus on pipeline perfect before features." Implement AFTER hardening pass Phase 7 (QA Playbook) lands.
**Effort:** S (human ~2d / CC ~30min)
**Priority:** P1
**Depends on:** Hardening pass complete (Phases 1-7)

### Resumable runs database
**What:** SQLite file at `~/.o2agol/runs.db` tracking `{country, theme, release, mode, status, started_at, finished_at, error_hash}` per pipeline run. Interactive mode surfaces "3 runs failed last session, retry?"
**Why:** Batch operators currently have no memory of what succeeded. A retry loop saves hours on 50+ country runs.
**Pros:** Enables real batch workflows; foundation for idempotent reruns.
**Cons:** New schema to version; failure modes include DB lock.
**Context:** Tied to interactive CLI. Deferred in CEO review.
**Effort:** S (human ~1d / CC ~30min)
**Priority:** P1
**Depends on:** Interactive CLI

### Databricks integration
**What:** `SourceBackend` protocol with `DuckDBBackend` (current) and `SparkBackend` (new). Spark backend reads Overture parquet via `spark.read.parquet("s3://...")` and does spatial filtering with Sedona or Mosaic. Plus Databricks Asset Bundle (`databricks.yml`) for declarative job deployment, plus secret-scope adapter in `config/settings.py`.
**Why:** Running on a laptop caps what one operator can do in a week. Databricks unlocks parallel country processing.
**Pros:** 10-100x throughput for multi-country jobs; shared pattern with sibling megaproject `minimum-data-package-hubs`.
**Cons:** Two runtimes to maintain; debugging distributed transforms is painful; AGOL publishing from Spark workers is unreliable (recommended pattern: Databricks job writes partitioned GeoParquet to S3/ADLS, a small driver-side step reads back with pandas and publishes via existing FeatureLayerManager).
**Context:** CEO review flagged the right architectural prep: the `GISClient` Protocol (Phase 2 of hardening) opens the door; lazy CLI wiring (Phase 1) makes config injection from `dbutils.secrets` cleaner.
**Effort:** XL (human ~6-8w / CC ~8-12 sessions)
**Priority:** P1
**Depends on:** Hardening pass + Interactive CLI

## P2 — Hardening pass follow-ups (after the 7-phase plan ships)

### Reusable QA harness extraction
**What:** Extract `tests/conftest.py` fixture patterns, `tests/contracts/*.json` schema loaders, `validation.py`, structured-log conventions, and error taxonomy into a standalone installable package (e.g., `o2agol-qa` or `wbgeoqa`). Sibling megaproject `minimum-data-package-hubs` imports it directly.
**Why:** The 7 data categories in the sibling project each need the same QA floor; extracting avoids copy-paste drift.
**Pros:** Single source of truth for the pattern; both projects evolve together.
**Cons:** Adds a coordination dependency between repos; early extraction risks wrong API shape.
**Context:** Deferred expansion #1 in the 2026-04-08 CEO review. Rationale for deferring: "build clean inside o2agol first, extract once the shape is known."
**Effort:** M (human ~1w / CC ~1 session)
**Priority:** P2
**Depends on:** Hardening pass complete; sibling megaproject ready to consume

### `source.py` decomposition
**What:** `pipeline/source.py` (1,840 LOC) split into `source/duckdb_remote.py`, `source/local_dump.py`, `source/cache.py`, `source/__init__.py` (facade). Tests drive the split.
**Why:** God-module hides bugs; one file for two fundamentally different data paths (S3 vs local) is an accident of history.
**Pros:** Easier to swap in a Spark backend later; each path testable in isolation.
**Cons:** Large refactor; must be done with test coverage as a net.
**Context:** Deferred in CEO review — tests-before-refactor policy.
**Effort:** M
**Priority:** P2
**Depends on:** Hardening pass Phase 3 (tests exist)

### Full AGOL output pydantic contracts
**What:** Plan's Phase 2 only contracts Overture *inputs*. Adding *output* contracts (the feature-layer schema AGOL receives) would assert every publish emits the expected fields.
**Why:** Closes the loop on schema drift — catches bugs where transform silently changes output shape.
**Pros:** Symmetric contract story; easy Playbook addition.
**Cons:** Adds pydantic work on top of an already-tested path.
**Effort:** S
**Priority:** P2
**Depends on:** Hardening pass Phase 2

### Property-based tests on transform invariants
**What:** `hypothesis` tests beyond the `apply_sql_filter` fuzz test: assert transform invariants (CRS always EPSG:4326, feature count monotonic or logged, no empty geoms out, no duplicate IDs introduced).
**Why:** Catches bugs golden-file tests miss.
**Pros:** High confidence per LOC of test code.
**Cons:** `hypothesis` can be slow; invariants must be carefully stated.
**Effort:** S
**Priority:** P2

### HTML diff report for golden-file regressions
**What:** When a golden-file test fails on CI, render a side-by-side HTML diff of expected vs actual GeoDataFrame (field-by-field + a small map showing geometry deltas). Upload as CI artifact.
**Why:** Pytest output for GeoDataFrames is unreadable; a visual diff turns 10-minute debugging into 30 seconds.
**Pros:** Quality-of-life for every future regression.
**Cons:** New tooling; requires a small reporting script.
**Effort:** S
**Priority:** P2

### Chaos test for SIGINT-mid-publish cleanup
**What:** Subprocess-based test that starts a publish, sends SIGINT after GPKG staging, asserts no `.gpkg` files linger in `/tmp`.
**Why:** `cleanup.py` exists but its interrupt path is untested; orphan staging files silently accumulate.
**Pros:** Catches a real operational hazard.
**Cons:** Subprocess tests are flaky on Windows CI.
**Effort:** S
**Priority:** P3

### Eliminate private arcgis API reaches in publish.py
**What:** Remove the two private-attribute reaches in `src/o2agol/pipeline/publish.py`:
- Line 427: `self.gis._con.post(url, {"f": "json", "groups": gid})` — direct portal HTTP call bypassing the library
- Line 929: `self.gis._portal = None` — private state manipulation, likely to work around an auth/profile bug

**Why:** Both use underscore-prefixed attributes that are not part of the arcgis library's public API. Any library upgrade can break them silently. A `pip install arcgis==2.5.x` future bump could ship without either error.
**Pros:** Library upgrades become safer; code becomes typed by real API.
**Cons:** Requires finding the public API equivalent. Line 929 in particular is likely a workaround for an auth-path bug and may need coordinated arcgis-library understanding.
**Context:** Found during 2026-04-08 /plan-eng-review (Issue 3b). publish.py:427 is in a group-sharing code path; publish.py:929 is inside an auth cleanup block. Phase 3 error taxonomy migration will naturally touch these sites — good opportunity to fix them then.
**Effort:** S (human ~4h / CC ~30min)
**Priority:** P2
**Depends on:** Phase 3 of hardening pass (error taxonomy migration touches publish.py)

### External webhook alerting (Slack/email)
**What:** On drift-canary failure, also post to a webhook (Slack, email via GitHub Actions).
**Why:** GitHub issue dedup is sufficient for teams watching the repo; external alerts close the loop for teams that don't.
**Pros:** Faster human reaction time.
**Cons:** Webhook secrets to manage.
**Effort:** XS
**Priority:** P3

### Manual live AGOL acceptance procedure
**What:** Before each release, an operator publishes 3 known countries (small/medium/large) to a dedicated test AGOL folder, runs `o2agol doctor --live`, verifies item metadata + feature counts + geometry types in the AGOL UI.
**Why:** Highest-risk production surface is real AGOL behavior; mocked tests verify structure but not real append/overwrite semantics. Outside voice (Codex) flagged this as priority-inverted in the original plan.
**Pros:** Catches real-world bugs no mock will.
**Cons:** Manual; requires a dedicated test AGOL folder with rotate-friendly creds.
**Effort:** XS (document only; running it is operator work)
**Priority:** P2
**Depends on:** Phase 4 doctor subcommand; Phase 7 documents the procedure
