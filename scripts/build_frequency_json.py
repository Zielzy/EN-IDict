"""
Convert Unigram frequency CSV to Yomitan / JLPT frequency JSON format.
Format: [["word", "reading"], ...]
For English, term and reading are identical.
"""
import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "frequency"))
INPUT_FILE = os.path.join(DATA_DIR, "unigram_freq.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "english_freq_jlpt_format.json")


def main():
    print(f"Reading {INPUT_FILE}...")
    jlpt_format_data = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header (word, count)

        count = 0
        for row in reader:
            if not row:
                continue
            word = row[0].strip()
            if word:
                jlpt_format_data.append([word, word])
                count += 1

    print(f"Loaded {count} words. Saving to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(jlpt_format_data, f, ensure_ascii=False, indent=2)

    print("Conversion complete!")


if __name__ == "__main__":
    main()

