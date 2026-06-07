"""End-to-end QA smoke test (Task 6.2).

Runs a small curated question set against the running backend and prints, per
case, the grounded answer + citations + visuals, with simple PASS/FAIL
heuristics. Not a unit-test suite - a quick demo/QA harness.

Prereq: backend running (`uvicorn backend.main:app`).
Run:    python qa_smoke.py [--base http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import sys

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REFUSAL = "i could not find this in the documents"


def post_chat(client: httpx.Client, message: str, session_id: str | None):
    r = client.post("/api/chat", json={"session_id": session_id, "message": message})
    r.raise_for_status()
    return r.json()


def show(case: str, q: str, data: dict, checks: list[tuple[str, bool]]):
    print("\n" + "=" * 78)
    print(f"{case}: {q!r}")
    print("-" * 78)
    ans = data.get("answer_markdown", "")
    print("answer:", " ".join(ans.split())[:320])
    cites = data.get("citations", [])
    vis = data.get("visuals", [])
    print(f"citations ({len(cites)}):", [c.get("label") for c in cites[:5]])
    print(f"visuals ({len(vis)}):", [(v.get("type"), v.get("url")) for v in vis[:4]])
    for name, ok in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    c = httpx.Client(base_url=args.base, timeout=120)

    passed = 0
    total = 0

    # (a) financial figure needing a TABLE
    d = post_chat(c, "What were the total borrowings from the twenty largest lenders?", None)
    total += 1
    passed += show("A · financial/table", "twenty largest lenders", d, [
        ("answer is grounded (has a [year, p.X] citation)", len(d["citations"]) > 0),
        ("a table/visual is attached", any(v["type"] == "table" for v in d["visuals"]) or len(d["visuals"]) > 0),
    ])

    # (b) chart / figure question
    d = post_chat(c, "Show the performance of the Rural Infrastructure Development Fund over the years.", None)
    total += 1
    passed += show("B · chart/figure", "RIDF performance", d, [
        ("has citations", len(d["citations"]) > 0),
        ("a visual is attached", len(d["visuals"]) > 0),
    ])

    # (c) narrative question
    d = post_chat(c, "What is NABARD's role in rural development?", None)
    total += 1
    passed += show("C · narrative", "NABARD role", d, [
        ("grounded answer with citations", len(d["citations"]) > 0),
        ("not a refusal", REFUSAL not in d["answer_markdown"].lower()),
    ])

    # (d) out-of-corpus -> refusal
    d = post_chat(c, "What is the population of Brazil?", None)
    total += 1
    passed += show("D · out-of-corpus", "population of Brazil", d, [
        ("honest refusal", REFUSAL in d["answer_markdown"].lower()),
        ("no fabricated citations/visuals", len(d["citations"]) == 0 and len(d["visuals"]) == 0),
    ])

    # (e+f) 2-turn follow-up in one session
    d1 = post_chat(c, "What was the Rural Infrastructure Development Fund allocation in 2023-24?", None)
    sid = d1["session_id"]
    total += 1
    passed += show("E · follow-up turn 1", "RIDF 2023-24", d1, [
        ("grounded", len(d1["citations"]) > 0),
    ])
    d2 = post_chat(c, "what about the previous year?", sid)
    total += 1
    passed += show("F · follow-up turn 2 (history)", "previous year", d2, [
        ("resolved via history (mentions 2022-23 or FY2022/2023)",
         any(k in d2["answer_markdown"] for k in ("2022-23", "2022", "FY2022", "FY2023"))),
        ("grounded or honest refusal",
         len(d2["citations"]) > 0 or REFUSAL in d2["answer_markdown"].lower()),
    ])

    print("\n" + "#" * 78)
    print(f"QA SUMMARY: {passed}/{total} cases passed   (follow-up session: {sid})")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
