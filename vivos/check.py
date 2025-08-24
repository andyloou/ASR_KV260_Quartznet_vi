from pathlib import Path
import pandas as pd

df = pd.read_csv("/home/andyloou/Downloads/vivos/test.csv")
all_text = ''.join(df['transcript']).lower()
unique_chars = sorted(set(all_text))

with open("/home/andyloou/DeepSpeech/data/alphabet.txt", "r", encoding="utf-8") as f:
    alphabet = f.read().splitlines()

missing = [ch for ch in unique_chars if ch not in alphabet]
print("Thiếu các ký tự:", missing)