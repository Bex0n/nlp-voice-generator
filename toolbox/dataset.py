import os

from typing import List

from datasets import Dataset, load_dataset

BLUE = '\033[94m'
RESET = '\033[0m'

def log(message: str):
    print(BLUE + message + RESET)

def get_vctk_cache_dir() -> str:
    dataset = load_dataset("vctk", trust_remote_code=True)
    if dataset['train'].cache_files:
        cache_file_path = dataset['train'].cache_files[0]['filename']
        cache_dir = os.path.dirname(cache_file_path)
        return cache_dir
    else:
        return None

def load_vctk_dataset() -> Dataset:
    dataset = load_dataset("vctk", trust_remote_code=True)
    return dataset

def get_vctk_speaker_ids() -> list:
    _ = load_vctk_dataset()
    speaker_ids = []
    try:
        with open(f"{get_vctk_cache_dir()}/speaker-info.txt", "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split()
                if parts:
                    speaker_id = parts[0]
                    speaker_ids.append(speaker_id)
    except FileNotFoundError:
        print(f"File not found: {get_vctk_cache_dir()}/speaker-info.txt")
    except Exception as e:
        print(f"An error occurred: {e}")

def fetch_speaker_audio(speaker_id, text_id):
    with open(f"{get_vctk_cache_dir()}/wav48_silence_trimmed/{speaker_id}/{speaker_id}_{text_id}_mic2.flac", "rb") as f:
        return f.read()

def fetch_text(speaker_id, text_id):
    with open(f"{get_vctk_cache_dir()}/txt/{speaker_id}/{speaker_id}_{text_id}.txt", "r") as f:
        return f.read()
