"""
translate_raw_ai.py  –  Translate raw Yomitan term banks via AI API (OpenAI-compatible)

Architecture:
    Python  = owns Yomitan structure entirely (all 8 fields)
    AI      = only sees {word, definitions} and returns {word, translations}
    Python  = injects translations back into entry[5]

Usage:
    python scripts/translate_raw_ai.py --test                     # 1 file, preview
    python scripts/translate_raw_ai.py --all --resume             # Full run, skip done
    python scripts/translate_raw_ai.py --all --part 01 --workers 8
    python scripts/translate_raw_ai.py --all --resume --workers 4

Requires:
    pip install openai python-dotenv
    .env file with: AI_API_KEY=sk-cos-xxx
"""

import argparse
import glob
import json
import os
import re
import sys
import threading
import time
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

try:
    from scripts.reorder_pos_priority import sort_entries
except ImportError:
    from reorder_pos_priority import sort_entries

# ─── Configuration ──────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

PROVIDERS = {
    "9router": {
        "env_key": "9ROUTER_API_KEY",
        "base_url": "http://localhost:20128/v1",
        "default_model": "cx/gpt-5.6-luna",
        "max_batch_chars": 3000,
        "default_workers": 4,
    },
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-3.7-flash",
        "max_batch_chars": 10000,
        "default_workers": 2,
    }
}

# These will be populated in main() based on the selected provider
API_KEY         = ""
BASE_URL        = ""
MODEL           = ""
MAX_BATCH_CHARS = 1000
WORKERS         = 2
PROVIDER_NAME   = ""

RAW_DIR         = "data/master/raw"
DONE_LOG        = "scripts/.translate_done.log"

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
Kamu adalah sistem penyusun kamus saku dwibahasa Inggris-Indonesia (EN-ID Pocket Dictionary) yang presisi, ringkas, dan natural untuk pop-up dictionary.

TUGAS:
1. Berikan terjemahan langsung atau padanan kata (glosarium) dalam bahasa Indonesia berdasarkan kata target (`word`) dan kelas katanya (`pos`).
2. Berikan nilai integer `score` (10 - 100) yang mencerminkan tingkat kepopuleran/frekuensi penggunaan kata tersebut dalam kelas kata (`pos`) tersebut di percakapan/literatur bahasa Inggris nyata:
   - 100 = Makna utama/sangat dominan (contoh: "about" (adp) = 100, "run" (verb) = 100, "book" (noun) = 100)
   - 60-80 = Makna umum sekunder (contoh: "about" (adv) = 80, "run" (noun) = 70, "book" (verb) = 60)
   - 10-30 = Makna yang sangat jarang/kuno/kiasan khusus (contoh: "about" (adj) = 15, "above" (noun) = 20)

INPUT:
Array JSON: [{"word": "...", "pos": "..."}]

OUTPUT (JSON):
{"results": [{"word": "...", "pos": "...", "score": 100, "definitions": ["makna 1", "makna 2", ...]}]}

PEDOMAN KETAT DEFINISI:
1. RINGKAS & LANGSUNG (Direct Translations Only):
   - Setiap elemen dalam array "definitions" HARUS berupa kata atau frasa padanan langsung (1 - 4 kata).
   - DILARANG membuat kalimat penjelasan panjang, definisi ensiklopedia, atau deskripsi bertele-tele.
   - Contoh BENAR: ["rumah", "tempat tinggal", "kediaman"]
   - Contoh SALAH: ["bangunan tempat tinggal manusia yang memiliki dinding dan atap"]

2. KESELARASAN KELAS KATA (Part of Speech):
   - noun  -> kata benda (contoh: "buku", "penyimpanan")
   - verb  -> kata kerja berimbuhan yang pas (contoh: "memesan", "berlari", "menjalankan")
   - adj   -> kata sifat (contoh: "cepat", "ramah", "indah")
   - adv   -> kata keterangan (contoh: "dengan cepat", "secara diam-diam")
   - adp/prep -> kata depan/preposisi (contoh: "tentang", "di atas", "dengan")
   - intj  -> kata seru / ungkapan (contoh: "astaga!", "jaga ucapanmu!")

3. PRIORITAS & CAKUPAN:
   - Berikan 2 sampai 6 padanan kata yang paling sering dipakai (makna primer di awal, lalu makna sekunder/idiomatik populer).

4. FORMAT BERSIH:
   - Gunakan huruf kecil (kecuali singkatan/nama khusus).
   - Jangan beri tanda titik koma (;) di ujung kata.
   - Jangan sertakan contoh kalimat, nomor urut, atau bullet point.
   - Gunakan bahasa Indonesia baku dan natural (KBBI/EYD).

5. PEMETAAN 1-KE-1 PERSIS & KELENGKAPAN ENTRI (SANGAT PENTING):
   - Jumlah elemen array "results" pada output HARUS PERSIS SAMA dengan jumlah elemen input.
   - Jika ada kata yang sama muncul berulang dengan pos berbeda (misal: "run" verb dan "run" noun), JANGAN PERNAH DIGABUNG ATAU DILEWATI! Terjemahkan masing-masing sebagai entri terpisah sesuai urutan input.

6. OUTPUT HARUS BERUPA JSON VALID MURNI TANPA TEKS LAIN.
"""

# ─── Thread-safe locks ───────────────────────────────────────────────────────
_done_lock  = threading.Lock()
_print_lock = threading.Lock()

class DailyLimitExceeded(Exception):
    pass

_key_lock = threading.Lock()
ACTIVE_KEYS = []
KEY_MODEL_INDEX = {}
FALLBACK_MODELS = {
    "gemini": [
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite"
    ],
    "9router": [
        "cx/gpt-5.6-luna",
        "cosmoshub/glm-5"
    ]
}

def auto_downgrade_key(api_key: str, failed_model: str):
    with _key_lock:
        if api_key not in KEY_MODEL_INDEX:
            return
        current_idx = KEY_MODEL_INDEX[api_key]
        fallbacks = FALLBACK_MODELS.get(PROVIDER_NAME, [])
        if fallbacks:
            if current_idx < len(fallbacks) and fallbacks[current_idx] != failed_model:
                # Already downgraded by another thread
                return
            
            next_idx = current_idx + 1
            if next_idx < len(fallbacks):
                KEY_MODEL_INDEX[api_key] = next_idx
                next_model = fallbacks[next_idx]
                with _print_lock:
                    print(f"\n  [KEY DOWNGRADE] Kunci (...{api_key[-4:]}) habis di {failed_model} -> Langsung beralih ke {next_model}!\n", flush=True)
            else:
                if api_key in ACTIVE_KEYS:
                    ACTIVE_KEYS.remove(api_key)
                with _print_lock:
                    print(f"\n  [KEY RETIRED] Kunci (...{api_key[-4:]}) telah menghabiskan SEMUA model fallback. Sisa kunci aktif: {len(ACTIVE_KEYS)}\n", flush=True)
                
                if not ACTIVE_KEYS:
                    with _print_lock:
                        print(f"\n  [FATAL] SEMUA API KEY TELAH MENGHABISKAN SELURUH MODEL FALLBACK! Menghentikan proses...\n", flush=True)
                    os._exit(1)
        else:
            if api_key in ACTIVE_KEYS:
                ACTIVE_KEYS.remove(api_key)
            if not ACTIVE_KEYS:
                with _print_lock:
                    print(f"\n  [FATAL] SEMUA API KEY TELAH HABIS LIMIT HARIANNYA! Menghentikan proses...\n", flush=True)
                os._exit(1)

def get_next_key_and_model() -> tuple[str, str]:
    with _key_lock:
        if not ACTIVE_KEYS:
            with _print_lock:
                print("\n  [FATAL] Tidak ada kunci aktif yang tersisa!\n", flush=True)
            os._exit(1)
        key = ACTIVE_KEYS[0]
        ACTIVE_KEYS.append(ACTIVE_KEYS.pop(0))
        fallbacks = FALLBACK_MODELS.get(PROVIDER_NAME, [])
        if fallbacks:
            model = fallbacks[KEY_MODEL_INDEX[key]]
        else:
            model = MODEL
        return key, model

# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_done_log() -> set:
    if not os.path.exists(DONE_LOG):
        return set()
    with open(DONE_LOG, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_done(file_path: str):
    with _done_lock:
        with open(DONE_LOG, "a", encoding="utf-8") as f:
            f.write(file_path + "\n")


def make_batches(entries: list, batch_size: int = 25) -> list[list]:
    """
    Split entries into fixed-size batches (e.g. 25 entries per batch).
    """
    return [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]


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


# ─── Input sanitizers ────────────────────────────────────────────────────────
_GLOSS_RE = re.compile(r'\(["\u201c]t=[^)\n]{0,200}["\u201d]\)')
_REF_RE   = re.compile(r'<ref\b[^>]*?/>|<ref\b[^>]*?>.*?</ref>', re.DOTALL | re.IGNORECASE)


def _sanitize_def(text: str) -> str:
    text = _GLOSS_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = text.replace('"', "\u2019")
    return text.strip()


def build_payload(entries: list) -> list[dict]:
    payload = []
    for i, e in enumerate(entries):
        if not isinstance(e, list) or len(e) <= 2:
            raise ValueError(f"Entry {i} memiliki struktur Yomitan tidak valid: {e!r:.80}")
        payload.append({
            "word": e[0],
            "pos": e[2],
        })
    return payload


def inject_translations(entries: list, results: list[dict]) -> list:
    merged = []
    for orig, res in zip(entries, results):
        entry = list(orig)
        try:
            score_val = int(res.get("score", 0))
        except (ValueError, TypeError):
            score_val = 0
        if len(entry) > 4:
            entry[4] = score_val
        defs = res.get("definitions", ["<terjemahan gagal>"])
        
        items = []
        if isinstance(defs, list):
            for d in defs:
                if isinstance(d, str):
                    for part in d.split(","):
                        p = part.strip()
                        if p:
                            items.append(p)
        elif isinstance(defs, str):
            items = [p.strip() for p in defs.split(",") if p.strip()]
        
        if not items:
            items = ["<terjemahan gagal>"]

        entry[5] = [{
            "type": "structured-content",
            "content": {
                "tag": "ul",
                "style": {"listStyleType": "circle"},
                "content": [{"tag": "li", "content": item} for item in items]
            }
        }]
        merged.append(entry)
    return merged


# ─── API call with retry ──────────────────────────────────────────────────────
def call_api(client: OpenAI, payload: list[dict], model: str, retries: int = 3) -> list[dict]:
    """Send payload to AI and return parsed response. Retries on transient errors."""
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=16000,
                response_format={"type": "json_object"},
                timeout=60.0
            )

            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "results" in parsed:
                return parsed["results"]
            return parsed

        except json.JSONDecodeError:
            preview = raw[:500].replace("\n", " ")
            with _print_lock:
                print(f"\n  [RAW AI OUTPUT ({model})]: {preview}...")
            raise

        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            
            if any(k in err_str for k in ("GenerateRequestsPerDay", "RESOURCE_EXHAUSTED", "PerDay", "quota exceeded", "limit reached", "rate_limit", "quota")):
                if hasattr(client, 'api_key'):
                    auto_downgrade_key(client.api_key, model)
                raise DailyLimitExceeded(err_str)

            is_transient = any(code in err_str for code in ("502", "503", "429", "timeout", "Connection"))
            if is_transient and attempt < retries:
                wait = 2 ** attempt
                # If API tells us how long to wait, respect it
                m = re.search(r"'retry_after':\s*(\d+)", err_str)
                m2 = re.search(r"'retryDelay':\s*'(\d+)s'", err_str)
                if m:
                    wait = int(m.group(1)) + 1
                elif m2:
                    wait = int(m2.group(1)) + 1
                with _print_lock:
                    print(f"\n  [RETRY {attempt}/{retries} on {model}] {err_str[:80]} -- tunggu {wait}s...", flush=True)
                time.sleep(wait)
                continue
            raise

    raise last_exc


# ─── Validation ───────────────────────────────────────────────────────────────
def validate(entries: list, results: list[dict]) -> str | None:
    if len(results) != len(entries):
        return f"Jumlah entri berbeda: {len(results)} vs {len(entries)}"
    for i, (orig, res) in enumerate(zip(entries, results)):
        if "definitions" not in res:
            return f"Entri {i}: key 'definitions' tidak ditemukan"
        if not isinstance(res["definitions"], list):
            return f"Entri {i}: 'definitions' bukan array"
        if not all(isinstance(x, str) for x in res["definitions"]):
            return f"Entri {i} ('{orig[0]}'): semua translation harus berupa string"
    return None


# ─── Core processor ──────────────────────────────────────────────────────────
def process_file(client: OpenAI, file_path: str, model: str, dry_run: bool = False) -> bool:
    """Process a single file in character-sized batches. Returns True on success."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as e:
        with _print_lock:
            print(f"  [ERROR] Gagal membaca {file_path}: {e}")
        return False

    if not isinstance(entries, list) or not entries:
        return True  # empty/skip

    merged: list = []

    for batch_idx, sub in enumerate(make_batches(entries)):

        def _try_batch(batch: list) -> list[dict] | None:
            """Try one batch, return results or None on failure."""
            try:
                payload = build_payload(batch)
                results = call_api(client, payload, model=model)
                err = validate(batch, results)
                if err:
                    with _print_lock:
                        print(f"  [VALIDATION ERROR] {err}")
                    return None
                return results
            except DailyLimitExceeded:
                raise
            except (json.JSONDecodeError, ValueError, Exception) as exc:
                with _print_lock:
                    print(f"  [API/DECODE ERROR] {exc}")
                return None

        try:
            results = _try_batch(sub)
        except DailyLimitExceeded:
            # Propagate up to trigger retry with downgraded key/model
            raise

        if results is None and len(sub) > 1:
            mid = len(sub) // 2
            split_batches = [sub[:mid], sub[mid:]]
            with _print_lock:
                print(f"  [RETRY] batch {batch_idx} ({len(sub)} entri) -> split menjadi 2 bagian ...", flush=True)
            
            results = []
            for b in split_batches:
                try:
                    r = _try_batch(b)
                except DailyLimitExceeded:
                    raise
                if r is None:
                    with _print_lock:
                        print(f"  [ERROR] Sebagian entri dalam batch ini gagal diterjemahkan. Menggugurkan seluruh file.")
                    return False
                else:
                    results.extend(r)

        elif results is None:
            # Single-entry batch still failed — abort file
            with _print_lock:
                print(f"  [ERROR] Entri '{sub[0][0]}' gagal (1-entry batch). Menggugurkan seluruh file.")
            return False

        merged.extend(inject_translations(sub, results))

    if not dry_run:
        merged = sort_entries(merged)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, separators=(",", ":"))

    return True



# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Translate raw Yomitan term banks via AI API")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test",    action="store_true", help="Translate hanya 1 file pertama untuk preview")
    group.add_argument("--all",     action="store_true", help="Translate semua file")
    parser.add_argument("--provider", type=str, choices=list(PROVIDERS.keys()), default="9router", help="API Provider (9router, gemini)")
    parser.add_argument("--model",  type=str, default=None, help="Override default model for the provider")
    parser.add_argument("--part",   type=str,  default=None,    help="Hanya proses folder term_bank_XX (misal: --part 01)")
    parser.add_argument("--resume", action="store_true",        help="Lewati file yang sudah diterjemahkan sebelumnya")
    parser.add_argument("--dry-run",action="store_true",        help="Simulasi tanpa menulis ke file")
    parser.add_argument("--workers",type=int,  default=None,    help="Jumlah parallel workers (default tergantung provider)")
    args = parser.parse_args()

    global API_KEYS, BASE_URL, MODEL, MAX_BATCH_CHARS, WORKERS, PROVIDER_NAME, ACTIVE_KEYS, KEY_MODEL_INDEX
    PROVIDER_NAME = args.provider
    cfg = PROVIDERS[args.provider]
    raw_key = os.getenv(cfg["env_key"], "")
    API_KEYS = [k.strip() for k in raw_key.split(",") if k.strip()]
    BASE_URL = cfg["base_url"]
    MODEL = args.model if args.model else cfg["default_model"]
    MAX_BATCH_CHARS = cfg["max_batch_chars"]
    WORKERS = args.workers if args.workers is not None else cfg["default_workers"]

    if not API_KEYS:
        print(f"[FATAL] {cfg['env_key']} tidak ditemukan. Silakan tambahkan di file .env")
        sys.exit(1)

    ACTIVE_KEYS = list(API_KEYS)
    KEY_MODEL_INDEX = {k: 0 for k in API_KEYS}

    print(f"[INFO] Memuat {len(API_KEYS)} API Key untuk {PROVIDER_NAME} (Per-Key Independent Downgrade Mode)")

    done_set  = load_done_log() if args.resume else set()
    all_files = get_all_files(part=args.part)

    # ── Test mode: single file, single worker ─────────────────────────────────
    if args.test:
        f = all_files[0]
        print(f"[TEST MODE] Memproses 1 file: {f}\n")
        key, model = get_next_key_and_model()
        client = OpenAI(api_key=key, base_url=BASE_URL)
        ok = process_file(client, f, model=model, dry_run=args.dry_run)
        print("[OK]" if ok else "[FAIL]")
        if ok and not args.dry_run:
            if not args.dry_run:
                mark_done(f)
            with open(f, "r", encoding="utf-8") as fh:
                result = json.load(fh)
            print("\n--- Preview 5 Entri Pertama ---")
            for entry in result[:5]:
                print(f"  {entry[0]} ({entry[2]}) [score={entry[4]}]: {entry[5]}")
            print("-----------------------------------\n")
        return

    # ── Full mode: parallel ───────────────────────────────────────────────────
    if args.resume:
        before = len(all_files)
        all_files = [f for f in all_files if f not in done_set]
        print(f"[RESUME] {before - len(all_files)} file sudah selesai, {len(all_files)} tersisa.")
    print(f"[INFO] Total: {len(all_files)} file | Workers: {WORKERS}\n")

    total   = len(all_files)
    counter = {"success": 0, "failed": 0, "done": 0}
    failed  = []
    c_lock  = threading.Lock()

    def run_one(file_path: str) -> tuple:
        max_retries = 5
        for _ in range(max_retries):
            key, model = get_next_key_and_model()
            client = OpenAI(api_key=key, base_url=BASE_URL)
            try:
                ok = process_file(client, file_path, model=model, dry_run=args.dry_run)
                return file_path, ok
            except DailyLimitExceeded:
                # Key already downgraded in call_api, loop to retry with newly active key/model
                continue
        return file_path, False

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, fp): fp for fp in all_files}
        for future in as_completed(futures):
            file_path, ok = future.result()
            with c_lock:
                counter["done"] += 1
                done_n = counter["done"]
                if ok:
                    counter["success"] += 1
                    if not args.dry_run:
                        mark_done(file_path)
                else:
                    counter["failed"] += 1
                    failed.append(file_path)
            status = "[OK]" if ok else "[FAIL]"
            with _print_lock:
                print(f"[{done_n:5d}/{total}] {file_path} ... {status}", flush=True)

    print(f"\n{'='*50}")
    print(f"Selesai: {counter['success']} berhasil, {counter['failed']} gagal")
    if failed:
        print("File yang gagal:")
        for f in sorted(failed):
            print(f"  - {f}")


if __name__ == "__main__":
    main()