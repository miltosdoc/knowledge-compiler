# Instructions for the agent working in this folder

You have been pointed at a Knowledge Compiler project directory. This file tells
you how to set it up, load clinical data, and build the knowledge base. Follow it
in order. Nothing here needs environment variables, API keys, or configuration —
every path resolves from this folder.

---

## 1. Set up

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[mcp,dense]"
```

Use the virtual environment. Installing into a system or conda environment
upgrades `starlette` and has broken unrelated packages.

`[dense]` pulls PyTorch (~500 MB) for semantic search. If it fails or the machine
is small, install `pip install -e ".[mcp]"` instead — retrieval degrades to
lexical plus graph, which works but finds fewer related concepts.

Verify:

```bash
./.venv/bin/kc2 stats
```

An empty vault reporting `notes 0` is correct on a fresh install.

## 2. Load the clinical data

The operator places source files in `data/`. Supported: SQLite (`.db`,
`.sqlite`) and PostgreSQL dumps (`.dump`, `.sql`).

```bash
./.venv/bin/kc2 sources --preview 3
```

This reports each table found and which columns it believes hold the transcript
and the clinical note. **Check that mapping before ingesting anything.** If
`transcript ->` or `note ->` reads `None`, the column names did not match the
built-in hints — show the operator the column list and ask which to use rather
than guessing.

If `data/` is empty, stop and tell the operator where to put their files.

## 3. Build the knowledge base

Work through the encounters yourself. **You** are the distiller — do not call
another model. These are raw clinical encounters and they must not leave this
machine.

For each batch:

1. `source_read(limit=5, offset=N)` — read five encounters.
2. For each encounter, identify the **reasoning patterns**: what the clinician
   attended to, what they ruled out, what a patient's phrasing told them. Not the
   facts of the case — the reasoning behind it.
3. For each pattern, call `concept_ingest(title, content, source, observed_on)`.
4. If it returns `"action": "ambiguous"`, it has found near-matches and is
   deliberately refusing to guess. Read the candidates and decide: same concept →
   `concept_merge`; genuinely different → `concept_ingest` again with a more
   distinct title.
5. Every 50 encounters, call `concept_compounding`.

Watch `evidence_per_concept` in that last call. It should **rise** while concept
count flattens. That means new encounters are reinforcing what is already known.
If concept count grows in step with encounters, knowledge is being appended
rather than integrated — say so, and stop to investigate rather than continuing.

### What to extract

Good — a durable reasoning pattern:

> **Title:** Exertional Relief as the Anchor for Angina
> **Content:** Symptoms reproduced by exertion and relieved within minutes of rest
> anchor an ischaemic mechanism more strongly than the character of the pain does.

Bad — a case report:

> **Title:** 62-year-old woman with chest pain
> **Content:** She presented in March with...

Bad — a frozen parameter:

> **Content:** Anticoagulate if the CHA2DS2-VASc score is 2 or more.

That last one will fail validation, by design. See below.

## 4. The rule about numbers

**Never write a threshold, score name, target value or drug dose into a concept.**
Those change. A note asserting one is correct today and wrong later, with nothing
marking it as stale.

Record instead what the clinician *did*: "the clinician applied the AF stroke-risk
score to decide anticoagulation." That stays true permanently.

Current values live in `norms/` as versioned records. When you need one, call
`norm_lookup` — never answer from memory. Your training data has a cutoff and
guidelines move; `CHA2DS2-VASc` was replaced by `CHA2DS2-VA` in 2024, and that
single change alters the anticoagulation answer for any woman under 65.

If a norm you need is missing, say so. Do not fill the gap from memory.

## 5. Using the knowledge base

Once built:

- `intuition_search` — find reasoning relevant to a case
- `intuition_compile` — assemble a reasoning module for a case or topic
- `concept_candidates` — check what is already known before adding
- `norm_lookup` / `norm_search` — current clinical parameters
- `kc2 sources`, `kc2 stats`, `kc2 norm <name>` from the shell

Compiled modules carry references rather than values, and raise a correction
block above the notes when they contain a superseded parameter. Respect it: the
correction is current, the note prose is historical.

## 6. Ground rules

- **Clinical text stays on this machine.** Distil with your own reasoning. Do not
  send transcripts or notes to any external service.
- **`data/` and `vault/` are gitignored.** Never commit them, never paste their
  contents into a message that leaves the machine.
- **Do not invent clinical content.** Every concept must trace to an encounter in
  the source data.
- **Ambiguity is surfaced, not resolved silently.** When the system asks you to
  decide, decide — that is the design, not a failure.

## 7. Layout

```
kc2/        the system
norms/      versioned clinical parameters
data/       source databases and dumps (operator-supplied, never committed)
vault/      generated knowledge — concepts/ holds the graph (never committed)
tests/      including a temporal regression suite
PLAN_v2.md  design rationale and the diagnosis of the previous version
```

The vault is Obsidian-native: YAML frontmatter, `[[wikilinks]]`, plain headings.
The operator can open `vault/concepts/` in Obsidian, browse the graph, and edit
by hand; those edits read back in.
