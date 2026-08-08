"""
Merge Unigram frequency CSV and Subtlex frequency JSON datasets into a deduplicated word list JSON.
"""
import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "frequency"))
UNIGRAM_PATH = os.path.join(DATA_DIR, "unigram_freq.csv")
SUBTLEX_PATH = os.path.join(DATA_DIR, "subtlex_top_200k.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "merged_unigram_subtlex.json")


def main():
    unique_words = set()

    print(f"Reading {os.path.basename(UNIGRAM_PATH)}...")
    with open(UNIGRAM_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if row and row[0].strip():
                unique_words.add(row[0].strip().lower())

    print(f"Total Unigram words loaded: {len(unique_words)}")

    print(f"Reading {os.path.basename(SUBTLEX_PATH)}...")
    with open(SUBTLEX_PATH, "r", encoding="utf-8") as f:
        subtlex_words = json.load(f)
        for word in subtlex_words:
            if isinstance(word, str) and word.strip():
                unique_words.add(word.strip().lower())

    unique_list = sorted(list(unique_words))
    print(f"Total unique merged words: {len(unique_list)}")

    print(f"Saving merged output to {os.path.basename(OUTPUT_PATH)}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_list, f, ensure_ascii=False)

    print("Merge complete!")


if __name__ == "__main__":
    main()

