"""Command line entry point for Knowledge Compiler v2.

    python -m kc2.cli stats
    python -m kc2.cli search "exertional dyspnoea in a hypertensive patient"
    python -m kc2.cli compile "62F atrial fibrillation, no other risk factors"
    python -m kc2.cli norm CHA2DS2-VASc
    python -m kc2.cli norms --stale
"""
from __future__ import annotations

import argparse
import sys

from .compile import Compiler
from .norms import NormStore


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kc2", description="Knowledge Compiler v2")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="graph and norms-layer statistics")

    s = sub.add_parser("search", help="search the reasoning graph")
    s.add_argument("query", nargs="+")
    s.add_argument("-k", type=int, default=10)

    c = sub.add_parser("compile", help="compile a reasoning module")
    c.add_argument("query", nargs="+")
    c.add_argument("--max-tokens", type=int, default=8000)

    n = sub.add_parser("norm", help="look up one clinical parameter")
    n.add_argument("norm_id")

    ns = sub.add_parser("norms", help="list the norms layer")
    ns.add_argument("--stale", action="store_true", help="only those past re-verification")

    im = sub.add_parser("sources", help="inspect raw clinical databases and dumps")
    im.add_argument("directory", nargs="?", default="data")
    im.add_argument("--preview", type=int, default=0, help="show N extracted records")

    a = p.parse_args(argv)

    if a.cmd == "stats":
        comp = Compiler()
        st = comp.retriever.stats()
        st["norms"] = len(comp.norms.norms)
        st["stale_norms"] = [x.id for x in comp.norms.stale()]
        for k, v in st.items():
            print(f"{k:18} {v}")
        return 0

    if a.cmd == "search":
        comp = Compiler()
        for title, score in comp.retriever.retrieve(" ".join(a.query), k=a.k):
            print(f"  {score:.3f}  {title}")
        return 0

    if a.cmd == "compile":
        out = Compiler().compile(" ".join(a.query), max_tokens=a.max_tokens)
        print(out.prompt)
        print(
            f"\n[{len(out.concepts)} concepts | ~{out.estimated_tokens} tokens | "
            f"{len(out.corrections)} correction(s) | norms: {', '.join(out.norm_refs) or '-'}]",
            file=sys.stderr,
        )
        return 0

    if a.cmd == "norm":
        store = NormStore()
        norm = store.get(a.norm_id)
        retired = False
        if norm is None:
            norm, retired = store.resolve_name(a.norm_id)
        if norm is None:
            print(f"no such norm: {a.norm_id}", file=sys.stderr)
            return 1
        if retired:
            print(f"!! {a.norm_id} is RETIRED -> current is {norm.current}\n")
        print(f"id            {norm.id}")
        print(f"current       {norm.current}")
        print(f"authority     {norm.authority}")
        print(f"valid_from    {norm.valid_from}")
        print(f"last_verified {norm.last_verified}  (expires {norm.expires_on})")
        print(f"stale         {norm.is_stale()}")
        for k, v in norm.thresholds.items():
            print(f"  {k:26} {v}")
        for sup in norm.supersedes:
            print(f"  supersedes {sup.get('name')}: {str(sup.get('migration','')).strip()[:120]}")
        return 0

    if a.cmd == "sources":
        from .sources import discover, extract

        found = discover(a.directory)
        if not found:
            print(f"no clinical sources found in {a.directory!r}", file=sys.stderr)
            print("place .db / .sqlite / .dump / .sql files there", file=sys.stderr)
            return 1
        for src in found:
            print(src.describe())
            print()
        if a.preview:
            usable = [s for s in found if s.usable]
            if usable:
                print(f"--- preview of {usable[0].path.name} ---")
                for rec in extract(usable[0], limit=a.preview):
                    print(f"  [{rec['id']}] {rec['transcript'][:90]!r}")
                    print(f"          note: {rec['notes'][:70]!r}")
        return 0

    if a.cmd == "norms":
        store = NormStore()
        items = store.stale() if a.stale else store.norms.values()
        for norm in sorted(items, key=lambda x: x.id):
            mark = "STALE" if norm.is_stale() else "ok"
            print(f"  [{mark:5}] {norm.id:34} {norm.current}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
