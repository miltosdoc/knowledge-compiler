# Knowledge Compiler v2 — Architecture & Build Plan

**Status:** in progress · **Started:** 2026-08-31 · **Supersedes:** the v1 pipeline described in `PROGRESS.md`

---

## 0. The one-sentence thesis

> **Store observations. Resolve norms at read time — by tool lookup, never from memory.**

Everything below follows from that sentence.

---

## 1. What went wrong in v1

v1 was a good idea executed against three defects that made its central claim untestable.

### 1.1 Retrieval was lexical substring matching

`compiler.py` seeds traversal with `seed.lower() in title.lower()`. Recovered from the April
session logs, live in production:

```
intuition_compile("refractory angina")  ->  0 concepts, out of 717
```

A dozen angina notes existed. None matched. The gateway made this worse by using
`msg.content[:100]` — the first 100 characters of a user's question — as the seed. A
100-character sentence is never a substring of a note title, so **auto-injection was a
permanent no-op.**

### 1.2 The graph was mostly dangling edges

`graph_indexer.py` counts every `[[link]]` target as a node whether or not a note backs it.

- Measured on the 112 recovered notes: **199 link targets, 21 resolve — 10%.**
- Confirmed independently by v1's own stats: **717 "concepts" vs 437 notes** — at minimum
  280 phantom nodes.

So "graph traversal beats RAG" was never actually tested. The graph it traversed was
40–90% empty.

### 1.3 Knowledge decay — the defect that matters most

The distiller welds two incompatible kinds of knowledge into one flat prose blob:

| | ages in | example from the vault |
|---|---|---|
| **Reasoning patterns** | decades | "the palpitations are a nuisance; the stroke is the threat" |
| **Normative parameters** | months | "a CHA2DS2-VASc score mandates it" |

ESC 2024 replaced CHA₂DS₂-VASc with **CHA₂DS₂-VA** (sex removed). For a 62-year-old woman
with AF and no other risk factors this is not cosmetic — it is the difference between a
score of 1 and a score of 0, and it changes what you say about anticoagulation.

**Worse: you cannot grep your way out of this.** A scan of the 112 notes found 16 (14%)
with a detectable time-sensitive claim. It *missed* this, in `Ruling Out Secondary Drivers
in New Arrhythmias`:

> "aggressive lipid management (LDL 4.5 to 2.6)"

ESC dyslipidaemia guidance puts very-high-risk LDL at **<1.4 mmol/L**. The note calls 2.6
"aggressive" when today it is a miss. No acronym, no unit — just a number inside a value
judgement. **The staleness is in the judgement, not the token.**

Compounding it: distillation ran on Gemma4-31b, which was asked to synthesise *general
rules* from specific encounters — so it promoted its own training-cutoff parameters into
timeless-sounding law.

### 1.4 Secondary defects

- Hardcoded `/Users/meditalks/knowledge-compiler` in all four modules — unrunnable elsewhere.
- **A live API key committed to a public repo** (`distiller.py:15`, `gateway.py:21`).
- `README` claims env-var key management; no `os.getenv` exists anywhere.
- `README` claims "no patient data leaves the local machine" while `distiller.py` POSTs
  full raw transcripts to a remote staging API.
- `sieve.py` emits `{id}.txt`; `distiller.py` reads `{id}.json` needing `transcript` +
  `notes`. The stages do not connect; whatever built the real vault is not in the repo.
- `distiller.py:85` caps each run at `remaining[:100]` — the real reason progress froze
  at 437/1311.
- Tag parsing keeps brackets: `tags: [#protocol, #intuition]` → `[#protocol`, `#intuition]`.
- Gateway accepts `stream: true` and ignores it; binds `0.0.0.0:8199` with no auth;
  recompiles on every request with no cache.
- Zero tests or evals for any of it — which is why 1.1 and 1.3 sat unnoticed for months.

---

## 2. The v2 invariant

Never store a sentence that can become false.

```
BAD   (rots):    "CHA2DS2-VASc mandates anticoagulation."
GOOD  (forever): "On 2024-03-12, in an AF case, the clinician applied the
                  AF stroke-risk score to decide anticoagulation."
```

The second is a historical fact about the clinician's practice. History does not change.
At answer time the system resolves *what the AF stroke-risk score currently is* —
CHA₂DS₂-VA, ESC 2024 — and applies that.

**The pattern survives. The parameter is re-resolved on every read.**

### 2.1 Corollary: numbers come from tools, not from weights

A language model asked for a threshold will answer from its training cutoff, confidently
and invisibly. So v2 makes that structurally impossible:

1. Compiled prompts contain **norm references**, never norm values.
2. A hard directive forbids emitting any threshold, score definition, target or dose
   from memory.
3. `norm_lookup` / `norm_search` MCP tools are the *only* sanctioned source of a number.
4. The freshness gate refuses to serve a norm past its TTL without flagging it.

---

## 3. Architecture

```
  transcripts ──► DISTILL ──► typed notes ──► CANONICALISE ──► INDEX ──► RETRIEVE ──► COMPILE ──► SERVE
                  (schema)    pattern|norm|    alias table      bm25+     hybrid+      refs, not   MCP +
                              vocabulary       real edges       dense+    rerank       values      gateway
                                                                graph
                                        ▲                                                 │
                                        │                                                 ▼
                                  NORMS LAYER  ◄────────── norm_lookup at read time ──────┘
                                  ~200 records, versioned, TTL'd, human-approved
```

### 3.1 Typed extraction

Constrained JSON output (grammar/`json_schema`), so notes are schema-valid at birth and
the bracket bug becomes unrepresentable. `kind` is required:

- `pattern` — durable reasoning. **May not contain a number or a score name.** Must
  reference norms by id.
- `norm` — a versioned parameter. The only place values are allowed to live.
- `vocabulary` — patient language → clinical meaning. Never expires.
- `protocol` — a procedural sequence; may reference norms.

Every note carries provenance: source transcript id, extracting model, date, confidence.

The distiller is constrained to what is *in the transcript*. It records
`the clinician applied X` — an observation — never `X is correct`.

### 3.2 Norm record schema

```yaml
id: norm:af-stroke-risk-score
kind: norm
current: CHA2DS2-VA
authority: ESC 2024 AF Guidelines
valid_from: 2024-08-30
volatility: 12mo          # -> TTL / re-verification cadence
last_verified: 2026-08-31
thresholds:
  oac_recommended: ">=2"
  oac_consider: "1"
  omit: "0"
supersedes:
  - name: CHA2DS2-VASc
    until: 2024-08-30
    migration: "drop sex category; a female-only score of 1 maps to 0"
```

**This is the whole answer to "forever."** One record changes and every note referencing
it is current instantly. You maintain ~200 norm records, not 1,311 prose notes.

### 3.3 Supersede, never delete

The retired `CHA₂DS₂-VASc` entry stays, with its migration rule. Two reasons:

- The notes are **evidence of your reasoning at a point in time**; rewriting them destroys
  that record.
- When a 2024 transcript surfaces, the compiler renders it forward:
  *"you used VASc here; under ESC 2024 that is VA, and for this patient the answer changes."*

Historical fidelity preserved, current guidance emitted.

### 3.4 Canonicalisation

Alias resolution over concept names — exact, fuzzy, then token-overlap — collapsing
`[[Orthopnea]]`, `[[orthopnoea]]`, `[[Orthopnea in HF]]` onto one canonical node. This is
what converts dangling links into real edges and makes graph expansion meaningful.
Embedding-based clustering replaces the string heuristics once vectors land.

### 3.5 Hybrid retrieval

BM25 (lexical) + dense vectors (semantic) fused by reciprocal rank, then 1–2 hop graph
expansion with score decay, then rerank. Query expansion (HyDE-style) matters specifically
here: you paste a *case*, not a term, and a paragraph of patient narrative must be expanded
into concepts before it can match anything.

Already demonstrated on the recovered corpus — `"refractory angina"` returns the full
angina cluster where v1 returned nothing.

### 3.6 Global layer (planned)

Community detection over the canonicalised graph plus an LLM summary per community. Local
retrieval answers *"what do I know about X."* Community summaries answer *"what are the
recurring patterns in how I reason about ischaemia"* — questions with no single seed. This
is the difference between a lookup tool and something that responds as an expert on the
domain.

### 3.7 Serving

- **MCP** — the five recovered v1 tools (`intuition_search`, `intuition_get_note`,
  `intuition_neighbors`, `intuition_compile`, `intuition_graph_stats`) plus `norm_lookup`,
  `norm_search`, and `ingest`.
- **Gateway** — real streaming passthrough, compile cache, auth, env-var config.

---

## 4. Guardrails

| Risk | Guardrail |
|---|---|
| Stale threshold emitted silently | Freshness gate; superseded norms inject a correction block at the top of the prompt |
| Model answers a number from memory | Values never appear in compiled prompts; hard directive + `norm_lookup` as the only source |
| Silent regression | Temporal regression suite in CI |
| PHI leaving the box | Local inference on `localhost:8000`; `vault/` fully gitignored |
| Unverifiable output | Every claim carries its note id; guideline content labelled distinctly from vault content |

---

## 5. Temporal regression suite

The thing v1 had no version of. Cases whose correct answer **changed over time**:

1. **62F, AF, no other risk factors** → CHA₂DS₂-VA = 0 → no OAC. Fails if the system says
   VASc, or scores her at 1, or recommends a NOAC.
2. **Very-high-risk LDL target** → <1.4 mmol/L. Fails on 2.6, 1.8, or "aggressive at 2.6".
3. Further cases added as norms are seeded.

Runs on every change. A superseded norm in an answer is a build failure — not a discovery
made months later by chance.

---

## 6. Build order

- [x] Recover clinical corpus from Hermes session logs — **112 notes** (`vault/atomic/`)
- [x] Diagnose v1 (measurements in §1)
- [x] Hybrid retriever prototype — BM25 + canonicalisation + graph expansion
- [x] `PLAN_v2.md` (this document)
- [x] **`kc2/` package** — config, schema, norms, index, retrieve, compile, cli
- [x] **Norms layer** — 4 seeded records (AF stroke risk, LDL targets, AF bleeding risk, HFrEF therapy)
- [x] **Temporal regression suite** — 26 tests; VASc→VA and LDL both covered
- [x] **MCP server** — 5 recovered tools + `norm_lookup` / `norm_search`
- [x] **Vault auditor** (`kc2/audit.py`) — per-note decay report
- [x] **Analyte-anchored detection** — catches `LDL 2.6`, the unit-less case a
      unit-based scan misses
- [x] **Pluggable dense backend** (`kc2/embed.py`) — sentence-transformers or any
      OpenAI-compatible `/v1/embeddings`; activates automatically, no hard dependency
- [ ] Install an embedding provider to switch dense retrieval on — pending approval
- [ ] Typed distiller with constrained decoding, running locally on `:8000`
- [ ] Community detection + global summaries
- [ ] Gateway v2 — streaming, cache, auth
- [ ] Backfill: ingest the Mac mini vault (437 notes) + distil the remaining 874

---

## 7. Open items requiring the operator

1. **Rotate the API key.** The v1 modules hardcoded a live xsilico key, which was
   public from the initial commit and is also present in `~/.hermes/config.yaml`.
   The v2 root commit discarded that history and no key appears anywhere in this
   repository, but the credential itself must still be rotated at the provider —
   discarding history reduces exposure, it does not end it.
2. **Approve `pip install sentence-transformers`** (+ `bge-small-en-v1.5`, ~130 MB, CPU) to
   enable the dense half of hybrid retrieval.
3. **Ship the Mac mini `vault/`** — `atomic/` is the irreplaceable part; `raw/` only if
   distillation is to finish on the Linux box (recommended: keeps PHI off the staging API).
4. **Norm review cadence.** Refresh proposes diffs; a clinician approves. Nothing about a
   clinical parameter is auto-merged.

---

## 7a. First audit of the recovered corpus (112 notes)

```
notes scanned                     112
clean (no time-sensitive content) 105
citing a RETIRED parameter          1   Hematological History and Thromboembolic Risk
values with no norm record          9   incl. Ruling Out Secondary Drivers (LDL 4.5 -> 2.6)
```

Nine notes hold a number a future reader could mistake for current guidance.
Each is a candidate to be lifted out of prose into a norm record.

---

## 8. What "forever" honestly means

Decay never stops. What this architecture buys is that decay becomes **localised, visible
and cheap**:

- confined to ~200 typed records instead of smeared through 1,311 prose notes,
- surfaced by an expiry clock instead of by luck,
- fixed by editing one field instead of re-distilling a corpus.

The 1,311 transcripts are a permanent, appreciating asset — they record how a specific
clinician actually thinks, and that does not expire. Only the parameters do. The
architecture's single job is to make sure those two things are never welded together again.
