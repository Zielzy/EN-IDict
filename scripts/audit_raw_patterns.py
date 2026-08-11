"""
audit_raw_patterns.py — Audit pola data bermasalah di seluruh file raw Wiktionary

Scans all 3,979 JSON files under data/wiktionary/raw/ and produces a report
of common problematic patterns that may affect AI translation quality.

Usage:
    python scripts/audit_raw_patterns.py

Output:
    Prints a ranked report to stdout.
    Saves full report to scripts/audit_raw_report.txt
"""

import glob
import json
import re
from collections import Counter, defaultdict

RAW_DIR = "data/wiktionary/raw"
REPORT_FILE = "scripts/audit_raw_report.txt"

# ─── Patterns to detect ──────────────────────────────────────────────────────
PATTERNS = {
    "wikt_t_template":     re.compile(r'\("t=[^"]*"\)'),          # ("t=...")
    "small_citation":      re.compile(r'<small>\[from[^\]]*\]</small>', re.I),
    "ref_tag":             re.compile(r'<ref\b[^>]*?>.*?</ref>|<ref\b[^>]*?/>', re.DOTALL | re.I),
    "literal_dquote":      re.compile(r'"'),                        # literal " in definitions
    "see_only":            re.compile(r'^\d+\.\s+<i>(?:See|see|See also|see also)</i>\s+', re.I),
    "surname_only":        re.compile(r'^\d+\.\s+[Aa]\s+surname\.?\s*$'),
    "placename_only":      re.compile(r'^\d+\.\s+[Aa]\s+(?:city|town|village|municipality|county|country|district)\b', re.I),
    "initialism_only":     re.compile(r'^\d+\.\s+<i>Initialism of</i>\s+', re.I),
    "plural_of":           re.compile(r'^\d+\.\s+<i>(?:plural|Plural)\s+of</i>\s+'),
    "alternative_form":    re.compile(r'^\d+\.\s+<i>(?:Alternative form|Alternative spelling) of</i>\s+', re.I),
    "unknown_tag":         re.compile(r'<(?!/?(?:i|b|br|small)\b)[a-z][^>]{0,40}>', re.I),  # non-whitelisted HTML tag
    "double_br":           re.compile(r'(<br>\s*){2,}'),           # consecutive <br>
    "empty_definition":    re.compile(r'^\s*$'),                    # empty string
}

# ─── Scan ─────────────────────────────────────────────────────────────────────
def scan():
    files = sorted(
        x.replace("\\", "/")
        for x in glob.glob(f"{RAW_DIR}/**/*.json", recursive=True)
        if "manifest" not in x
    )

    total_files = len(files)
    total_entries = 0
    total_defs = 0

    # Counters per pattern
    pattern_hits: dict[str, int] = defaultdict(int)             # # of definitions with this pattern
    pattern_examples: dict[str, list[tuple]] = defaultdict(list) # (word, pos, definition_snippet)
    unknown_tags_seen: Counter = Counter()

    for file_path in files:
        try:
            data = json.load(open(file_path, encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, list):
            continue

        for entry in data:
            if not isinstance(entry, list) or len(entry) <= 5:
                continue

            total_entries += 1
            word = entry[0]
            pos  = entry[2]
            defs = entry[5] if isinstance(entry[5], list) else []

            for d in defs:
                if not isinstance(d, str):
                    continue

                total_defs += 1

                for pat_name, pat in PATTERNS.items():
                    if pat.search(d):
                        pattern_hits[pat_name] += 1
                        if len(pattern_examples[pat_name]) < 5:
                            pattern_examples[pat_name].append(
                                (word, pos, d[:100])
                            )

                # Collect unknown tags specifically
                for tag in re.findall(r'<(/?\w+)[^>]{0,40}>', d, re.I):
                    tag_lower = tag.lower().lstrip("/")
                    if tag_lower not in ("i", "b", "br", "small"):
                        unknown_tags_seen[tag_lower] += 1

    return {
        "total_files": total_files,
        "total_entries": total_entries,
        "total_defs": total_defs,
        "pattern_hits": dict(pattern_hits),
        "pattern_examples": dict(pattern_examples),
        "unknown_tags": unknown_tags_seen,
    }


# ─── Report ───────────────────────────────────────────────────────────────────
def make_report(data: dict) -> str:
    lines = []
    w = lines.append

    w("=" * 60)
    w("AUDIT POLA DATA RAW — EN-IDict Wiktionary")
    w("=" * 60)
    w(f"Total file   : {data['total_files']:,}")
    w(f"Total entri  : {data['total_entries']:,}")
    w(f"Total definisi: {data['total_defs']:,}")
    w("")

    w("─" * 60)
    w("POLA BERMASALAH (diurutkan dari yang paling sering)")
    w("─" * 60)

    friendly_names = {
        "wikt_t_template":  '("t=...") Wiktionary gloss markers',
        "small_citation":   '<small>[from ...]</small> historical citations',
        "ref_tag":          '<ref>...</ref> citation markup',
        "literal_dquote":   'Literal double-quote " in definitions',
        "see_only":         '"See X" cross-ref-only definitions',
        "surname_only":     '"A surname." stub definitions',
        "placename_only":   '"A city/town/..." stub definitions',
        "initialism_only":  '"Initialism of X" abbreviation definitions',
        "plural_of":        '"Plural of X" morphology definitions',
        "alternative_form": '"Alternative form of X" definitions',
        "unknown_tag":      'Non-whitelisted HTML tags (<span>, <sup>, etc.)',
        "double_br":        'Consecutive <br> tags',
        "empty_definition": 'Empty definition strings',
    }

    hits = sorted(data["pattern_hits"].items(), key=lambda x: -x[1])
    for pat_name, count in hits:
        pct = count / data["total_defs"] * 100
        w(f"\n[{count:>7,} defs | {pct:5.1f}%] {friendly_names.get(pat_name, pat_name)}")
        for word, pos, snippet in data["pattern_examples"].get(pat_name, [])[:3]:
            snippet_clean = snippet.replace("\n", " ")
            w(f"    ex: [{word}] ({pos}) → {snippet_clean!r}")

    if data["unknown_tags"]:
        w("")
        w("─" * 60)
        w("TAG HTML TIDAK DIKENAL (top 20)")
        w("─" * 60)
        for tag, cnt in data["unknown_tags"].most_common(20):
            w(f"  <{tag}>  : {cnt:,}")

    w("")
    w("=" * 60)
    w("END OF REPORT")
    w("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    print("Scanning... (ini bisa makan beberapa menit)")
    result = scan()
    report = make_report(result)
    # Write to file first (always succeeds regardless of terminal encoding)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Laporan disimpan ke: {REPORT_FILE}")

    # Print to terminal via utf-8 buffer to avoid Windows cp1252 error
    import sys
    sys.stdout.buffer.write(report.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
