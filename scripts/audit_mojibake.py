"""
Comprehensive Mojibake Auditor & Fixer for EN-IDict Yomitan Dictionary Term Banks
"""
import json
import glob
import os


def find_mojibake_in_text(text):
    """
    Finds exact Mojibake sequence matches in a string.
    Returns list of (mojibake_substring, fixed_substring).
    """
    if not isinstance(text, str):
        return []

    issues = []
    i = 0
    n = len(text)

    while i < n:
        char = text[i]
        code = ord(char)

        # Pattern 1: Ã (U+00C3) followed by \u0080-\u00bf (e.g. Ã© -> é)
        if code == 0x00C3 and i + 1 < n and 0x0080 <= ord(text[i + 1]) <= 0x00BF:
            bad_seq = text[i:i + 2]
            try:
                fixed_char = bad_seq.encode('latin1').decode('utf-8')
                issues.append((bad_seq, fixed_char))
            except Exception:
                pass
            i += 2
            continue

        # Pattern 2: Â (U+00C2) followed by \u0080-\u00bf (e.g. Â° -> °)
        elif code == 0x00C2 and i + 1 < n and 0x0080 <= ord(text[i + 1]) <= 0x00BF:
            bad_seq = text[i:i + 2]
            try:
                fixed_char = bad_seq.encode('latin1').decode('utf-8')
                issues.append((bad_seq, fixed_char))
            except Exception:
                pass
            i += 2
            continue

        # Pattern 3: â€ (U+00E2 U+0080) followed by \u0080-\u00bf (e.g. â€™ -> ’)
        elif code == 0x00E2 and i + 2 < n and ord(text[i + 1]) == 0x0080 and 0x0080 <= ord(text[i + 2]) <= 0x00BF:
            bad_seq = text[i:i + 3]
            try:
                fixed_char = bad_seq.encode('latin1').decode('utf-8')
                issues.append((bad_seq, fixed_char))
            except Exception:
                pass
            i += 3
            continue

        # Pattern 4: U+FFFD Replacement Character
        elif code == 0xFFFD:
            issues.append(("\ufffd", ""))
            i += 1
            continue

        i += 1

    return issues


def fix_mojibake_in_text(text):
    if not isinstance(text, str):
        return text

    issues = find_mojibake_in_text(text)
    if not issues:
        return text

    fixed_text = text
    for bad_seq, fixed_char in issues:
        fixed_text = fixed_text.replace(bad_seq, fixed_char)

    return fixed_text


def audit_folder(folder_path, apply_fix=False):
    print(f"=== Auditing {folder_path} for Mojibake ===")
    json_files = sorted(
        glob.glob(os.path.join(folder_path, "term_bank_*.json")),
        key=lambda x: int(os.path.basename(x).replace("term_bank_", "").replace(".json", ""))
    )

    total_scanned = 0
    found_issues = []

    for file_path in json_files:
        fname = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_scanned += len(data)
        file_modified = False

        for idx, entry in enumerate(data):
            term = entry[0]
            defs = entry[5] if isinstance(entry[5], list) else [str(entry[5])]

            # Check term
            term_issues = find_mojibake_in_text(term)
            if term_issues:
                found_issues.append((fname, idx, term, "term", term_issues))
                if apply_fix:
                    entry[0] = fix_mojibake_in_text(term)
                    file_modified = True

            # Check definitions
            new_defs = []
            for d_idx, d in enumerate(defs):
                if isinstance(d, str):
                    d_issues = find_mojibake_in_text(d)
                    if d_issues:
                        found_issues.append((fname, idx, term, f"def[{d_idx}]", d_issues))
                        if apply_fix:
                            d = fix_mojibake_in_text(d)
                            file_modified = True
                new_defs.append(d)

            if apply_fix:
                entry[5] = new_defs

        if apply_fix and file_modified:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"FIXED & SAVED: {fname}")

    print(f"Total entries scanned: {total_scanned:,}")
    print(f"Total Mojibake occurrences found: {len(found_issues)}")

    for fname, idx, term, loc, issues in found_issues:
        print(f"\nFile: {fname} | Index: {idx} | Term: '{term}' | Field: {loc}")
        for bad_seq, fixed in issues:
            print(f"   Corrupted: {repr(bad_seq)}  --->  Fixed: {repr(fixed)}")

    return found_issues


if __name__ == "__main__":
    import sys
    do_fix = "--fix" in sys.argv
    audit_folder("dict", apply_fix=do_fix)
    if os.path.exists("data/wiktionary/processed"):
        audit_folder("data/wiktionary/processed", apply_fix=do_fix)
