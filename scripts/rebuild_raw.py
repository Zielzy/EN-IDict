"""
rebuild_raw.py — Rebuild raw/ folder from processed/ with full cleanup

Reads all 291,420 entries from data/wiktionary/processed/,
cleans each definition of Wiktionary artifacts, splits into files
of 50 entries each, and writes to data/wiktionary/raw/.

This OVERWRITES the existing raw/ content. Run BEFORE translate_raw_ai.py.

Usage:
    python scripts/rebuild_raw.py [--dry-run]

Output:
    data/wiktionary/raw/term_bank_01/ ... term_bank_30/
    Each folder: up to 200 files of 50 entries each.
    scripts/rebuild_raw_report.txt
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import sys

PROCESSED_DIR = "data/master/processed"
RAW_DIR       = "data/master/raw"
ENTRIES_PER_FILE = 50
DONE_LOG      = "scripts/.translate_done.log"

# ─── Cleanup Regexes ─────────────────────────────────────────────────────────

# 1. Strip <math>...</math> and <ref>...</ref> entirely (content removed)
_STRIP_CONTENT_RE = re.compile(
    r'<(?:math|ref)\b[^>]*>.*?</(?:math|ref)>|<ref\b[^>]*/>\s*',
    re.DOTALL | re.I,
)

# 2. Strip <small>[from ...]</small> historical citation footnotes
_SMALL_FROM_RE = re.compile(
    r'\s*<small>\s*\[(?:from|First attested)[^\]]*\]\s*</small>',
    re.I,
)

# 3. Unwrap non-whitelisted tags (remove tag, keep inner text)
#    Whitelisted: <i>, <b>, <br>
_UNWRAP_TAGS_RE = re.compile(
    r'</?(?:sub|sup|s|u|em|strong|code|span|p|small|font|center|div|td|tr|th|table|dl|dt|dd|ul|ol|li|hr|abbr|cite|q)\b[^>]*>',
    re.I,
)

# 4. Wiktionary <<template>> markup
#    <<river>>             -> "river"
#    <<c/Germany>>         -> "Germany"
#    <<s/Baden-Württ...>>  -> "Baden-Württ..."
#    <<river/Neckar>>      -> "Neckar"
_WIKT_TMPL_RE = re.compile(r'<<([^>]+)>>')

def _resolve_wikt_tmpl(m: re.Match) -> str:
    inner = m.group(1)
    if '/' in inner:
        return inner.split('/', 1)[1].strip()
    # Simple semantic tags: <<river>>, <<city>>, <<tributary>> etc.
    # These are hyperlink labels — keep the word itself
    return inner.strip()

# 5. Wiktionary ("t=gloss") template markers
_T_GLOSS_RE = re.compile(r'\s*\(["\u201c]t=[^)\n]{0,200}["\u201d]\)')

# 6. Duplicate article bug: "A A " or "a a " at start of sentences (data artifact)
_DUP_ARTICLE_RE = re.compile(r'\b([Aa]n?) \1 ', re.I)

# 7. Consecutive <br> (3+) → collapse to <br><br>
_MULTI_BR_RE = re.compile(r'(<br>\s*){3,}', re.I)

# 8. Trailing/leading whitespace around <br>
_BR_SPACE_RE = re.compile(r'\s*(<br>)\s*', re.I)


def clean_definition(text: str) -> str:
    """Apply all cleanup transforms to a single definition string."""
    # Order matters: strip math/ref first (they contain raw HTML we don't want)
    text = _STRIP_CONTENT_RE.sub('', text)
    text = _SMALL_FROM_RE.sub('', text)
    text = _WIKT_TMPL_RE.sub(_resolve_wikt_tmpl, text)
    text = _T_GLOSS_RE.sub('', text)
    text = _UNWRAP_TAGS_RE.sub('', text)
    text = _DUP_ARTICLE_RE.sub(r'\1 ', text)
    text = _MULTI_BR_RE.sub('<br><br>', text)
    text = _BR_SPACE_RE.sub(r'\1', text)
    # Collapse multiple spaces (not inside tags)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def clean_entry(entry: list) -> list:
    """Clean index[5] (definitions list) of a Yomitan entry. Returns new entry."""
    if not isinstance(entry, list) or len(entry) <= 5:
        return entry
    entry = list(entry)  # copy, don't mutate
    defs = entry[5]
    if isinstance(defs, list):
        entry[5] = [clean_definition(d) if isinstance(d, str) else d for d in defs]
    elif isinstance(defs, str):
        entry[5] = clean_definition(defs)
    return entry


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Rebuild raw/ from processed/ with cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without writing files")
    args = parser.parse_args()

    # Load all processed files (sorted numerically, not lexicographically)
    proc_files = sorted(
        [f for f in glob.glob(f"{PROCESSED_DIR}/*.json") if "index" not in f],
        key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group(1))
    )

    print(f"Found {len(proc_files)} processed files.")

    # Stats
    stats = {
        "total_entries": 0,
        "total_defs": 0,
        "strip_content": 0,
        "small_from": 0,
        "wikt_tmpl": 0,
        "t_gloss": 0,
        "unwrap_tags": 0,
        "dup_article": 0,
        "multi_br": 0,
    }

    # Count transforms per pattern for reporting
    def count_transforms(original: str, cleaned: str, orig_defs: list, clean_defs: list):
        for o, c in zip(orig_defs, clean_defs):
            if not isinstance(o, str):
                continue
            stats["total_defs"] += 1
            if _STRIP_CONTENT_RE.search(o):      stats["strip_content"] += 1
            if _SMALL_FROM_RE.search(o):          stats["small_from"]    += 1
            if _WIKT_TMPL_RE.search(o):           stats["wikt_tmpl"]     += 1
            if _T_GLOSS_RE.search(o):             stats["t_gloss"]       += 1
            if _UNWRAP_TAGS_RE.search(o):         stats["unwrap_tags"]   += 1
            if _DUP_ARTICLE_RE.search(o):         stats["dup_article"]   += 1
            if _MULTI_BR_RE.search(o):            stats["multi_br"]      += 1

    # Gather all cleaned entries
    all_entries: list = []

    for bank_idx, proc_file in enumerate(proc_files, 1):
        raw_entries = json.load(open(proc_file, encoding="utf-8"))
        print(f"  [Bank {bank_idx:02d}] {os.path.basename(proc_file)}: {len(raw_entries):,} entries", end="", flush=True)

        cleaned = []
        for entry in raw_entries:
            if not isinstance(entry, list) or len(entry) <= 5:
                cleaned.append(entry)
                continue
            orig_defs = entry[5] if isinstance(entry[5], list) else [entry[5]]
            c = clean_entry(entry)
            clean_defs = c[5] if isinstance(c[5], list) else [c[5]]
            count_transforms(None, None, orig_defs, clean_defs)
            cleaned.append(c)
            stats["total_entries"] += 1

        all_entries.extend(cleaned)
        print(f" -> {len(cleaned):,} cleaned")

    total = len(all_entries)
    n_files = math.ceil(total / ENTRIES_PER_FILE)
    n_banks = len(proc_files)
    files_per_bank = math.ceil(n_files / n_banks)

    print(f"\nTotal entries : {total:,}")
    print(f"Total files   : {n_files:,} (50 entries each)")
    print(f"Banks         : {n_banks:,} (term_bank_01 through term_bank_{n_banks:02d})")
    print(f"Files/bank    : up to {files_per_bank}")

    # Print cleanup stats
    print("\n--- Cleanup Stats ---")
    for key, val in stats.items():
        if key.startswith("total"):
            continue
        pct = val / stats["total_defs"] * 100 if stats["total_defs"] else 0
        print(f"  {key:20s}: {val:7,} defs ({pct:.1f}%)")

    if args.dry_run:
        print("\n[DRY RUN] Tidak ada file yang ditulis.")
        return

    # Clear existing raw/ and rebuild
    print(f"\nMembersihkan {RAW_DIR}/ ...")
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    os.makedirs(RAW_DIR)

    # Split and write
    print("Menulis file raw baru ...")
    chunk_size = ENTRIES_PER_FILE
    file_count = 0

    for bank_idx, proc_file in enumerate(proc_files, 1):
        # How many entries belong to this bank?
        bank_start = (bank_idx - 1) * 10000
        bank_end   = min(bank_start + 10000, len(all_entries))
        bank_entries = all_entries[bank_start:bank_end]

        bank_name = f"term_bank_{bank_idx:02d}"
        bank_dir  = os.path.join(RAW_DIR, bank_name)
        os.makedirs(bank_dir, exist_ok=True)

        for part_idx, start in enumerate(range(0, len(bank_entries), chunk_size), 1):
            chunk = bank_entries[start:start + chunk_size]
            out_file = os.path.join(bank_dir, f"{bank_name}_part_{part_idx:03d}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, separators=(",", ":"))
            file_count += 1

        print(f"  [Bank {bank_idx:02d}] {bank_name}/  {len(bank_entries):,} entries -> {math.ceil(len(bank_entries)/chunk_size)} files")

    # Clear translate done log (fresh start)
    with open(DONE_LOG, "w", encoding="utf-8") as f:
        f.write("")
    print(f"\nTranslate log dikosongkan: {DONE_LOG}")

    print(f"\nSelesai! {file_count:,} file ditulis ke {RAW_DIR}/")
    print("Jalankan berikutnya: python scripts/translate_raw_ai.py --all --resume")


if __name__ == "__main__":
    main()
