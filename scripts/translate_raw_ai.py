"""
translate_raw_ai.py  –  Translate raw Yomitan term banks via AI API (OpenAI-compatible)

Architecture:
    Python  = owns Yomitan structure entirely (all 8 fields)
    AI      = only sees {word, definitions} and returns {word, translations}
    Python  = injects translations back into entry[5]

Usage:
    python scripts/translate_raw_ai.py --test          # Translate only the first 1 file (50 entries)
    python scripts/translate_raw_ai.py --all           # Translate ALL files
    python scripts/translate_raw_ai.py --part 01 --all # Translate all files in term_bank_01 only
    python scripts/translate_raw_ai.py --resume --all  # Skip files already translated

Requires:
    pip install openai python-dotenv
    .env file with: COSMOS_API_KEY=sk-cos-xxx
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ─── Config ──────────────────────────────────────────────────────────────────
# Resolve .env from repo root (parent of scripts/) regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")
API_KEY   = os.getenv("COSMOS_API_KEY")
BASE_URL  = "https://api.cosmoshub.tech/v1"
MODEL     = "deepseek-3.2"
RAW_DIR   = "data/wiktionary/raw"
DONE_LOG  = "scripts/.translate_done.log"
# Batches are sized by total definition-text length, not a fixed entry count.
# Some entries (e.g. abbreviations with 50+ senses like "AA") are far longer
# than a typical entry; a fixed entry-count batch can silently overflow
# max_tokens and get the response truncated mid-JSON.
MAX_BATCH_CHARS = 6000

# ─── System Prompt ────────────────────────────────────────────────────────────
# AI only sees word + definitions. Python owns everything else.
SYSTEM_PROMPT = """\
Kamu adalah penerjemah kamus profesional Inggris-Indonesia.

Tugasmu adalah menerjemahkan definisi kamus bahasa Inggris ke bahasa Indonesia yang akurat, natural, dan mudah dipahami oleh penutur asli Indonesia.

INPUT:
Array JSON. Setiap elemen memiliki format:
{"word": "...", "definitions": ["definisi 1", "definisi 2", ...]}

OUTPUT:
Array JSON dengan format yang sama persis, hanya ganti key "definitions" menjadi "translations":
{"word": "...", "translations": ["terjemahan 1", "terjemahan 2", ...]}

ATURAN WAJIB:

1. Jumlah elemen output harus sama persis dengan input.

2. Jumlah string dalam "translations" harus sama persis dengan jumlah string dalam "definitions". Jangan menggabungkan, memecah, menghapus, menambah, atau mengubah urutan definisi.

3. Terjemahkan setiap definisi secara akurat. Jangan meringkas berlebihan. Jangan menghilangkan informasi dari definisi asli.

4. Jangan menambahkan informasi, contoh, sinonim, konteks, atau interpretasi yang tidak ada dalam teks asli.

5. Pertahankan makna, nuansa, negasi, tingkat kepastian, dan informasi gramatikal dari definisi asli.

6. Pertahankan tag HTML: <i>, <b>, <br>. Jangan membuat tag HTML baru.

7. Hapus metadata historis yang tidak membantu pemahaman, seperti:
   <small>[from ca. 1350]</small>, <small>[from 1350\u20131470]</small>

8. Terjemahkan label register ke padanan Indonesia yang natural jika padanannya jelas (contoh: obsolete \u2192 usang/kuno, archaic \u2192 arkais/kuno, informal \u2192 informal, slang \u2192 slang). Jika tidak ada padanan yang jelas, pertahankan label aslinya. Pertahankan label tersebut jika penting untuk memahami konteks penggunaan kata.

9. Pertahankan proper noun, istilah teknis, singkatan, dan nama orang/tempat/organisasi. Terjemahkan hanya jika ada padanan Indonesia yang jelas.

10. Pertahankan referensi silang (see, see also, compare) secara natural tanpa menghilangkan referensinya.

11. Jika definisi ambigu, terjemahkan berdasarkan makna paling langsung yang didukung teks asli. Jangan menebak atau menambah konteks.

12. Gunakan "word" hanya sebagai konteks semantik untuk disambiguasi. Jangan menerjemahkan atau mengubah "word".

13. Output HARUS berupa JSON valid, tanpa markdown code block, tanpa komentar, tanpa teks di luar JSON.

CONTOH INPUT:
[
  {"word": "charge", "definitions": ["to ask someone to pay a particular amount", "to accuse someone formally"]}
]

CONTOH OUTPUT:
[
  {"word": "charge", "translations": ["meminta seseorang membayar sejumlah tertentu", "menuduh seseorang secara resmi"]}
]
"""


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_done_log() -> set:
    if not os.path.exists(DONE_LOG):
        return set()
    with open(DONE_LOG, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(file_path: str):
    with open(DONE_LOG, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")


def make_batches(entries: list, max_chars: int = MAX_BATCH_CHARS) -> list[list]:
    """
    Split entries into batches sized by total definition-text length rather
    than a fixed entry count, so a handful of unusually long entries (e.g.
    abbreviations with dozens of senses) can't blow past max_tokens and get
    the AI response truncated mid-JSON. A single entry longer than max_chars
    still gets its own batch rather than being split (definitions of one
    entry must never be separated across API calls).
    """
    batches: list[list] = []
    current: list = []
    current_len = 0

    for e in entries:
        e_len = sum(len(d) for d in e[5] if isinstance(d, str))
        if current and current_len + e_len > max_chars:
            batches.append(current)
            current, current_len = [], 0
        current.append(e)
        current_len += e_len

    if current:
        batches.append(current)

    return batches


def get_all_files(part: str | None = None) -> list[str]:
    pattern = f"{RAW_DIR}/**/*.json"
    files = sorted(
        x.replace("\\", "/")
        for x in glob.glob(pattern, recursive=True)
        if "manifest" not in x
    )
    if part:
        files = [f for f in files if f"term_bank_{part}/" in f]
    return files


# Wiktionary template gloss marker: ("t=ante meridiem") — not part of the
# definition, contains literal " which breaks AI's JSON output.
# Matches both straight ("...") and curly ("\u201c...\u201d") quote variants so
# it stays consistent with the cleanup regex in call_api().
_WIKT_GLOSS_RE = re.compile(r'\(["\u201c]t=[^)\n]{0,200}["\u201d]\)')

# Wiktionary <ref>...</ref> / <ref .../> citation markup — not part of the
# definition text. Left unstripped, the AI tends to treat it like the
# whitelisted <i>/<b>/<br> tags and copies it verbatim into the translation.
_WIKT_REF_RE = re.compile(r'<ref\b[^>]*?/>|<ref\b[^>]*?>.*?</ref>', re.DOTALL | re.IGNORECASE)


def _sanitize_def(text: str) -> str:
    """
    Strip Wiktionary template artifacts and neutralize remaining literal
    double-quotes before sending definitions to the AI.

    Wiktionary uses ("t=gloss") markers (e.g. ("t=ante meridiem")) that are
    internal template metadata, not actual definition text. When the AI copies
    them verbatim, the embedded " characters break its JSON output. Wiktionary
    also embeds <ref>...</ref> citation markup, which carries no translation
    value and otherwise leaks straight into the output alongside the tags we
    do want to keep (<i>, <b>, <br>).
    """
    # 1. Remove ("t=...") template markers entirely—they add no translation value
    text = _WIKT_GLOSS_RE.sub("", text)
    # 2. Remove <ref>...</ref> / <ref .../> citation markup entirely
    text = _WIKT_REF_RE.sub("", text)
    # 3. Replace any remaining literal " with a typographic apostrophe
    text = text.replace('"', "\u2019")
    return text.strip()


def build_payload(entries: list) -> list[dict]:
    """
    Extract word + definitions for the AI. Python owns everything else.
    Raises ValueError if any entry is structurally invalid — never silently
    drops entries, which would cause translations to shift to the wrong entry.
    """
    payload = []
    for i, e in enumerate(entries):
        if not isinstance(e, list) or len(e) <= 5:
            raise ValueError(f"Entry {i} memiliki struktur Yomitan tidak valid: {e!r:.80}")
        payload.append({
            "word": e[0],
            "definitions": [
                _sanitize_def(d) if isinstance(d, str) else d
                for d in e[5]
            ],
        })
    return payload


def inject_translations(entries: list, results: list[dict]) -> list:
    """Merge AI translations back into original Yomitan entries at index [5]."""
    merged = []
    for orig, res in zip(entries, results):
        entry = list(orig)              # copy; never mutate original
        entry[5] = res["translations"]  # only touch index 5
        merged.append(entry)
    return merged


def call_api(client: OpenAI, payload: list[dict], retries: int = 3) -> list[dict]:
    """Send payload to AI and return parsed response. Retries on transient errors."""
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=16000,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown code blocks if AI adds them anyway
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            # Post-process AI output before JSON parsing:
            # 1. Strip Wiktionary ("t=...") gloss artifacts the AI hallucinates back
            raw = re.sub(r'\(["\u201c]t=[^)\n]{0,200}["\u201d]\)', '', raw)
            # 2. Replace any remaining ("text") unescaped-quote patterns with curly quotes
            raw = re.sub(
                r'\("([^"\n]{0,300})"\)',
                lambda m: '(\u201c' + m.group(1) + '\u201d)',
                raw,
            )

            return json.loads(raw)

        except json.JSONDecodeError:
            preview = raw[:500].replace("\n", " ")
            print(f"\n  [RAW AI OUTPUT]: {preview}...")
            raise  # JSON errors are not transient — don't retry

        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            # Retry on transient server/network errors
            is_transient = any(code in err_str for code in ("502", "503", "429", "timeout", "Connection"))
            if is_transient and attempt < retries:
                wait = 2 ** attempt  # 2s, 4s, 8s
                print(f"\n  [RETRY {attempt}/{retries}] {err_str[:80]} — tunggu {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            raise  # Non-transient or exhausted retries

    raise last_exc  # Should not reach here


def validate(entries: list, results: list[dict]) -> str | None:
    """
    Validate AI output. Returns an error message string on failure, None on success.
    Python guarantees all non-definition fields are untouched (they were never sent).
    This validation checks translation integrity and uses 'word' as a checksum.
    """
    if len(results) != len(entries):
        return f"Jumlah entri berbeda: {len(results)} vs {len(entries)}"

    for i, (orig, res) in enumerate(zip(entries, results)):
        # Word checksum — detects index drift or hallucinated responses
        if res.get("word") != orig[0]:
            return (
                f"Entri {i}: word berubah "
                f"('{orig[0]}' \u2192 '{res.get('word')}')"
            )
        if "translations" not in res:
            return f"Entri {i}: key 'translations' tidak ditemukan"
        if not isinstance(res["translations"], list):
            return f"Entri {i}: 'translations' bukan array"
        if len(res["translations"]) != len(orig[5]):
            return (
                f"Entri {i} ('{orig[0]}'): "
                f"jumlah terjemahan {len(res['translations'])} != "
                f"jumlah definisi asli {len(orig[5])}"
            )
        # All translations must be strings
        if not all(isinstance(x, str) for x in res["translations"]):
            return f"Entri {i} ('{orig[0]}'): semua translation harus berupa string"

    return None


# ─── Core processor ──────────────────────────────────────────────────────────
def process_file(client: OpenAI, file_path: str, dry_run: bool = False) -> bool:
    """Process a single file in sub-batches. Returns True on success."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as e:
        print(f"  [ERROR] Gagal membaca {file_path}: {e}")
        return False

    if not isinstance(entries, list) or not entries:
        print(f"  [SKIP] File kosong atau bukan list: {file_path}")
        return True

    merged: list = []

    for start, sub in enumerate(make_batches(entries)):
        try:
            payload = build_payload(sub)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            return False

        try:
            results = call_api(client, payload)
        except json.JSONDecodeError as e:
            print(f"  [ERROR] AI mengembalikan JSON tidak valid (sub-batch {start}): {e}")
            return False
        except Exception as e:
            print(f"  [ERROR] API error (sub-batch {start}): {e}")
            return False

        err = validate(sub, results)
        if err:
            print(f"  [ERROR] Validasi gagal (sub-batch {start}): {err}")
            return False

        merged.extend(inject_translations(sub, results))

    if not dry_run:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, separators=(",", ":"))

    return True


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Translate raw Yomitan term banks via AI API")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test",  action="store_true", help="Translate hanya 1 file pertama untuk preview")
    group.add_argument("--all",   action="store_true", help="Translate semua file")
    parser.add_argument("--part",    type=str, default=None, help="Hanya proses folder term_bank_XX (misal: --part 01)")
    parser.add_argument("--resume",  action="store_true",    help="Lewati file yang sudah diterjemahkan sebelumnya")
    parser.add_argument("--dry-run", action="store_true",    help="Simulasi tanpa menulis ke file")
    args = parser.parse_args()

    if not API_KEY:
        print("[FATAL] COSMOS_API_KEY tidak ditemukan. Isi file .env:\n  COSMOS_API_KEY=sk-cos-xxx")
        sys.exit(1)

    client   = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    done_set = load_done_log() if args.resume else set()
    all_files = get_all_files(part=args.part)

    if args.test:
        all_files = all_files[:1]
        print(f"[TEST MODE] Memproses 1 file: {all_files[0]}\n")
    else:
        if args.resume:
            before = len(all_files)
            all_files = [f for f in all_files if f not in done_set]
            print(f"[RESUME] {before - len(all_files)} file sudah selesai, {len(all_files)} tersisa.")
        print(f"[INFO] Total file yang akan diproses: {len(all_files)}\n")

    success, failed = 0, []

    for i, file_path in enumerate(all_files, 1):
        print(f"[{i:4d}/{len(all_files)}] {file_path} ... ", end="", flush=True)
        ok = process_file(client, file_path, dry_run=args.dry_run)

        if ok:
            success += 1
            if not args.dry_run:
                mark_done(file_path)
            print("[OK]")
        else:
            failed.append(file_path)
            print("[FAIL]")

        # Preview in test mode
        if args.test and ok and not args.dry_run:
            with open(file_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            print("\n--- Preview 5 Entri Pertama ---")
            for entry in result[:5]:
                print(f"  {entry[0]} ({entry[2]}): {entry[5]}")
            print("-----------------------------------\n")

        # Throttle antara request
        if not args.test and i < len(all_files):
            time.sleep(0.3)

    print(f"\n{'='*50}")
    print(f"Selesai: {success} berhasil, {len(failed)} gagal")
    if failed:
        print("File yang gagal:")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()