"""
Diagnostic Script: Comprehensive Mojibake and Encoding Artifact Scanner for Yomitan Term Banks
"""
import json
import glob
import os
import re

# Specific Mojibake character sequences when UTF-8 bytes were misdecoded as Latin1/CP1252
MOJIBAKE_PATTERNS = [
    # Common double-encoded UTF-8 accented characters
    (re.compile(r'\u00c3[\u0080-\u00bf]'), "Double-encoded UTF-8 character (Ã...)"),
    (re.compile(r'\u00c2[\u0080-\u00bf]'), "Double-encoded UTF-8 character (Â...)"),
    (re.compile(r'\u00e2\u0080[\u0080-\u00bf]'), "Double-encoded UTF-8 punctuation (â€...)"),
    (re.compile(r'\ufffd'), "U+FFFD Replacement Character ()"),
    (re.compile(r'&amp;(?:amp|lt|gt|quot|#39);'), "Double-escaped HTML entity"),
]


def test_fix_mojibake(text):
    """
    Attempt to fix Mojibake by encoding to Latin-1/Windows-1252 and decoding as UTF-8.
    Returns (is_mojibake, fixed_text).
    """
    if not isinstance(text, str):
        return False, text

    # Quick check for suspicious lead characters
    if not any(char in text for char in ['\u00c3', '\u00c2', '\u00e2', '\ufffd']):
        return False, text

    try:
        # Test converting Latin-1 back to UTF-8
        raw_bytes = text.encode('latin1')
        fixed = raw_bytes.decode('utf-8')
        if fixed != text and len(fixed) < len(text):
            return True, fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    return False, text


def scan_dictionary_folder(folder_path):
    print(f"=== Scanning dictionary folder: {folder_path} ===")
    json_files = sorted(
        glob.glob(os.path.join(folder_path, "term_bank_*.json")),
        key=lambda x: int(os.path.basename(x).replace("term_bank_", "").replace(".json", ""))
    )

    total_entries = 0
    mojibake_entries = []

    for file_path in json_files:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            try:
                entries = json.load(f)
            except Exception as e:
                print(f"ERROR reading {filename}: {e}")
                continue

        total_entries += len(entries)

        for entry_idx, entry in enumerate(entries):
            term = entry[0]
            reading = entry[1]
            pos = entry[2]
            defs = entry[5] if isinstance(entry[5], list) else [str(entry[5])]

            has_issue = False
            issue_details = []

            # Check term
            is_m, fixed_term = test_fix_mojibake(term)
            if is_m:
                has_issue = True
                issue_details.append(("term", term, fixed_term))

            # Check definitions
            for d_idx, d_text in enumerate(defs):
                if isinstance(d_text, str):
                    is_m, fixed_d = test_fix_mojibake(d_text)
                    if is_m:
                        has_issue = True
                        issue_details.append((f"def[{d_idx}]", d_text, fixed_d))

            if has_issue:
                mojibake_entries.append({
                    "file": filename,
                    "entry_idx": entry_idx,
                    "term": term,
                    "pos": pos,
                    "issues": issue_details
                })

    print(f"Total entries scanned: {total_entries:,}")
    print(f"Total entries with Mojibake: {len(mojibake_entries):,}")

    return mojibake_entries


def main():
    dict_issues = scan_dictionary_folder("dict")
    
    if dict_issues:
        print("\n--- SAMPLE MOJIBAKE FINDINGS (dict/) ---")
        for idx, item in enumerate(dict_issues[:20]):
            print(f"\n[{idx+1}] File: {item['file']} | Index: {item['entry_idx']} | Term: '{item['term']}'")
            for field, orig, fixed in item["issues"]:
                print(f"   Field '{field}':")
                print(f"     ORIGINAL: {repr(orig[:120])}")
                print(f"     FIXED   : {repr(fixed[:120])}")

    # Also scan data/wiktionary/processed/ if exists
    if os.path.exists("data/wiktionary/processed"):
        proc_issues = scan_dictionary_folder("data/wiktionary/processed")


if __name__ == "__main__":
    main()
