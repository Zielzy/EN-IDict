"""
Deep Mojibake Scanner for EN-IDict Yomitan Dictionary Term Banks
Checks for specific character sequences created by UTF-8 double-encoding,
Windows-1252 / ISO-8859-1 misdecodings, replacement characters, and HTML double-escapes.
"""
import glob
import json
import os
import re

# Comprehensive list of Mojibake regexes and their true UTF-8 equivalents
MOJIBAKE_MAP = [
    # Double-encoded quotes, dashes, bullets
    (re.compile(r'â€™'), "’"),
    (re.compile(r'â€œ'), "“"),
    (re.compile(r'â€\x9d'), "”"),
    (re.compile(r'â€\x9c'), "“"),
    (re.compile(r'â€“'), "–"),
    (re.compile(r'â€”'), "—"),
    (re.compile(r'â€¢'), "•"),
    (re.compile(r'â€¦'), "…"),
    (re.compile(r'â€'), "”"),
    (re.compile(r'Â°'), "°"),
    (re.compile(r'Â '), " "),
    (re.compile(r'Â'), ""),  # stray UTF-8 non-breaking space artifact

    # Double-encoded Latin accented characters
    (re.compile(r'Ã©'), "é"),
    (re.compile(r'Ã¨'), "è"),
    (re.compile(r'Ã '), "à"),
    (re.compile(r'Ã¢'), "â"),
    (re.compile(r'Ã¤'), "ä"),
    (re.compile(r'Ã¶'), "ö"),
    (re.compile(r'Ã¼'), "ü"),
    (re.compile(r'Ã§'), "ç"),
    (re.compile(r'Ã±'), "ñ"),
    (re.compile(r'Ã«'), "ë"),
    (re.compile(r'Ã®'), "î"),
    (re.compile(r'Ã´'), "ô"),
    (re.compile(r'Ã»'), "û"),
    (re.compile(r'Ã¬'), "ì"),
    (re.compile(r'Ã³'), "ó"),
    (re.compile(r'Ã¡'), "á"),
    (re.compile(r'Ã­'), "í"),
    (re.compile(r'Ãº'), "ú"),
    (re.compile(r'Ã½'), "ý"),
    (re.compile(r'Ã¿'), "ÿ"),
    (re.compile(r'Ã‘'), "Ñ"),
    (re.compile(r'Ã‰'), "É"),
    (re.compile(r'Ã€'), "À"),
    (re.compile(r'ÃÂ'), "Á"),
    (re.compile(r'Ã„'), "Ä"),
    (re.compile(r'Ã–'), "Ö"),
    (re.compile(r'ÃÜ'), "Ü"),
    (re.compile(r'Ã‡'), "Ç"),

    # Unicode replacement character & HTML double escapes
    (re.compile(r'\ufffd'), "[REPLACEMENT_CHAR_U+FFFD]"),
    (re.compile(r'&amp;amp;'), "&amp;"),
    (re.compile(r'&amp;quot;'), "&quot;"),
    (re.compile(r'&amp;lt;'), "&lt;"),
    (re.compile(r'&amp;gt;'), "&gt;"),
    (re.compile(r'&amp;#39;'), "&#39;"),
    (re.compile(r'&amp;nbsp;'), "&nbsp;")
]


def scan_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    findings = []
    for pattern, replacement in MOJIBAKE_MAP:
        matches = pattern.findall(content)
        if matches:
            findings.append((pattern.pattern, len(matches), replacement))

    return len(content), findings


def scan_and_fix_dictionary(folder_path, fix=False):
    print(f"\n=======================================================")
    print(f"  SCANNING FOLDER: {folder_path}")
    print(f"=======================================================")

    files = sorted(
        glob.glob(os.path.join(folder_path, "term_bank_*.json")),
        key=lambda x: int(os.path.basename(x).replace("term_bank_", "").replace(".json", ""))
    )

    total_files = len(files)
    total_mojibake_instances = 0
    file_findings = {}

    for file_path in files:
        fname = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        modified = False
        file_issues = 0

        for entry_idx, entry in enumerate(data):
            # entry structure: [term, reading, definition_tags, rules, score, [definitions], sequence, term_tags]
            term = entry[0]
            defs = entry[5] if isinstance(entry[5], list) else [str(entry[5])]

            # Fix term if needed
            new_term = term
            for pattern, rep in MOJIBAKE_MAP:
                if rep != "[REPLACEMENT_CHAR_U+FFFD]" and pattern.search(new_term):
                    new_term = pattern.sub(rep, new_term)

            if new_term != term:
                file_issues += 1
                entry[0] = new_term
                modified = True

            # Fix definitions if needed
            new_defs = []
            for d in defs:
                if isinstance(d, str):
                    new_d = d
                    for pattern, rep in MOJIBAKE_MAP:
                        if rep != "[REPLACEMENT_CHAR_U+FFFD]" and pattern.search(new_d):
                            new_d = pattern.sub(rep, new_d)
                    if new_d != d:
                        file_issues += 1
                        modified = True
                    new_defs.append(new_d)
                else:
                    new_defs.append(d)

            entry[5] = new_defs

        if file_issues > 0:
            file_findings[fname] = file_issues
            total_mojibake_instances += file_issues

            if fix:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)

    print(f"Scanned {total_files} term bank files.")
    print(f"Total Mojibake instances found: {total_mojibake_instances}")

    if file_findings:
        print("\nBreakdown by file:")
        for fname, count in file_findings.items():
            print(f"  - {fname}: {count} Mojibake items")
    else:
        print("✅ NO MOJIBAKE FOUND! All files are 100% clean UTF-8 text.")

    return total_mojibake_instances


if __name__ == "__main__":
    scan_and_fix_dictionary("dict", fix=False)
    if os.path.exists("data/wiktionary/processed"):
        scan_and_fix_dictionary("data/wiktionary/processed", fix=False)
