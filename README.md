# Knowledge Compiler

A system for turning a clinician's recorded reasoning into a reusable reasoning
module — and for keeping that module correct as the guidelines underneath it change.

## The problem

Retrieval systems return passages. What a specialist actually carries is different:
a set of priors about what matters, what to rule out, and which detail in a patient's
account reframes the case. That reasoning is implicit in consultation transcripts —
present in what was asked and what was ordered, absent from the final note.

Extracting it is one problem. Keeping it usable is a second, harder one.

Clinical reasoning has two components with very different lifespans:

- **Reasoning patterns** — *the palpitations are a nuisance, the stroke is the threat.*
  These age over decades.
- **Normative parameters** — score definitions, thresholds, targets, doses.
  These age in months.

Prose welds them together. Once fused, a change in guidance silently invalidates
notes that still read as authoritative, and no amount of re-reading reliably finds
them: the staleness sits in the judgement, not in any greppable token. A note
describing a lipid result as well controlled carries no marker that the target moved.

## The principle

> **Store observations. Resolve norms at read time — by tool lookup, never from memory.**

A note records *that a clinician applied a given scoring system on a given date*. That
statement never becomes false. What the scoring system currently **is** gets resolved on
every read, from a versioned record, through an explicit lookup.

The pattern survives. The parameter is re-resolved.

A language model asked for a threshold will answer from its training cutoff — confidently,
and invisibly. So compiled output carries *references*, never values, and instructs its
reader to look every parameter up rather than recall it.

## Architecture

```
transcripts → distil → typed notes → canonicalise → index → retrieve → compile → serve
                         │                                                │
                         pattern | norm | vocabulary | protocol           │
                                                                          ▼
                                    norms layer ──── looked up at read time
                                    versioned · TTL'd · human-approved
```

**Typed extraction.** Schema-constrained output. A `pattern` note that inlines a score
name or a threshold fails validation, so the defect becomes unrepresentable.

**Norms layer.** Every parameter that can change lives in exactly one versioned record
with an authority, a re-verification interval, and migration rules. Retired parameters
are superseded, never deleted — the notes are evidence of reasoning at a point in time,
and rewriting them would destroy that record. One record changes and the whole corpus
is current.

**Canonicalisation.** Link targets resolve onto real notes before the graph is built,
so traversal moves along edges that exist.

**Hybrid retrieval.** Lexical and dense scoring fused by reciprocal rank, then graph
expansion with distance decay. Queries are usually a whole case, not a keyword.

**Compilation.** Retrieved reasoning is assembled under a token budget. Any retired
parameter found in the notes raises a correction block above them, where it cannot be
missed. Values appearing in note prose are relabelled as historical observations.

**Serving.** An MCP server exposes search, note retrieval, neighbour traversal,
compilation, and the norm lookups.

## Usage

```bash
pip install -e .

kc2 stats
kc2 search "exertional dyspnoea in a hypertensive patient"
kc2 compile "62F atrial fibrillation, no other risk factors"
kc2 norm CHA2DS2-VASc          # resolves a retired name to its current replacement
kc2 norms --stale              # parameters past re-verification
python -m kc2.audit            # per-note decay report over the vault
```

Configuration is entirely environmental — `KC_VAULT_DIR`, `KC_NORMS_DIR`, `KC_BASE_URL`,
`KC_MODEL`, `KC_API_KEY`. Inference defaults to a local OpenAI-compatible endpoint so
clinical text need not leave the machine. No credential is ever read from source.

## Layout

```
kc2/          the system
  schema.py     typed records; validation that forbids inlined parameters
  norms.py      versioned parameters, supersession, freshness
  index.py      note loading, canonicalisation, lexical index
  retrieve.py   hybrid retrieval with graph expansion
  compile.py    reasoning modules carrying references, not values
  audit.py      vault decay report
  embed.py      pluggable dense backend
  mcp_server.py MCP tools
norms/        the versioned parameter records
tests/        including a temporal regression suite
legacy/       the frozen v1 pipeline, kept for reference only
PLAN_v2.md    design rationale, measured diagnosis of v1, build order
```

## Verification

Correctness here is temporal: an answer that was right last year may be wrong now. The
test suite therefore includes cases whose correct answer **changed**, and a superseded
parameter reaching an answer fails the build rather than surfacing months later by chance.

```bash
pytest tests/
```

## Getting data in

Place source databases in `data/` — SQLite (`.db`, `.sqlite`) or pg_dump
(`.dump`, `.sql`). Both `data/` and `vault/` are excluded from version control.

```bash
kc2 sources data/ --preview 3
```

Discovery is introspective: it reports each table it found and which columns it
believes hold the transcript and the clinical note, so the mapping can be
confirmed before anything is ingested. An agent driving the MCP server reads
encounters through `source_read`, distils each one itself, and calls
`concept_ingest` per reasoning pattern — the model already in the loop does the
extraction, and no clinical text is sent anywhere else.

## Clinical data

No clinical content belongs in this repository. Distilled notes retain comorbidity
chains and verbatim patient phrasing, which are indirect identifiers; `vault/` is
excluded in full, and the tests use synthetic notes. Distillation is intended to run
against a local model.

## Status

The v2 package is functional: typed records, the norms layer with supersession and
freshness, canonicalised retrieval, compilation with correction blocks, the auditor,
the MCP surface, and the regression suite. Dense retrieval is implemented behind a
backend probe and activates when an embedding provider is present.

`legacy/` holds the original pipeline. It is frozen, its stages do not connect, and it
should not be built on — `PLAN_v2.md` documents why in detail.

## License

MIT. See `LICENSE`.
