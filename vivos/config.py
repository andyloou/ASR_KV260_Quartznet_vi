import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

def build_wav_lookup(wav_root):
    """Trả về dict {file_id: full_path} từ thư mục gốc chứa waves/"""
    wav_paths = {}
    for root, _, files in os.walk(wav_root):
        for file in files:
            if file.endswith('.wav'):
                file_id = os.path.splitext(file)[0]
                full_path = os.path.join(root, file)
                wav_paths[file_id] = full_path
    return wav_paths

def create_deepspeech_csv(wav_root, prompts_file, output_csv):
    wav_lookup = build_wav_lookup(wav_root)
    rows = []

    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            file_id, transcript = parts
            if file_id not in wav_lookup:
                print(f"❌ Không tìm thấy file âm thanh: {file_id}")
                continue
            wav_path = wav_lookup[file_id]
            size = os.path.getsize(wav_path)
            rows.append([wav_path, size, transcript.lower()])

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['wav_filename', 'wav_filesize', 'transcript'])
        writer.writerows(rows)

# Dùng cho train/test
create_deepspeech_csv(
    wav_root='./train/waves',
    prompts_file='./train/prompts.txt',
    output_csv='train_full.csv'
)

create_deepspeech_csv(
    wav_root='./test/waves',
    prompts_file='./test/prompts.txt',
    output_csv='test.csv'
)

# Chia train_full.csv thành train.csv và dev.csv
df = pd.read_csv('train_full.csv')
train_df, dev_df = train_test_split(df, test_size=0.1, random_state=42)
train_df.to_csv('train.csv', index=False)
dev_df.to_csv('dev.csv', index=False)
