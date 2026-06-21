# Maimai ChartGPT

Train a model that generates maimai charts (notes + timing) from audio.

## Project Structure

```
src/
  tokenizer.py   – Vocabulary + ChartTokenizer (encode/decode)
  dataloader.py  – MaiMaiDataset, audio loading, mel spectrograms
  model.py       – ChartGPT model (audio encoder + transformer decoder)
  train.py       – Training loop, validation, checkpointing
```

## Setup

```bash
uv sync
```

Requires `ffmpeg` on PATH for audio decoding.

## Tokenizer

```python
from src.tokenizer import ChartTokenizer, VOCAB_SIZE

# Encode chart notes to tokens
bpm10_list = [{"bpm10": 1750, "change_timestamp_ms": 0}]
tokenizer = ChartTokenizer(bpm10_list)
tokens = tokenizer.encode_notes(notes)      # list[int]
notes = tokenizer.decode_tokens(tokens)      # list[dict]

print(f"Vocabulary size: {VOCAB_SIZE}")      # 13267
```

Run built-in tests:
```bash
python src/tokenizer.py
```

## Pre-compute Mel Statistics

Before training, compute per-channel mel spectrogram mean/std (80 mel bands,
10 ms hop, 16 kHz mono).  **Re-run this after changing any audio parameter.**

```bash
python -c "
from src.dataloader import compute_mel_stats
import torch
mean, std = compute_mel_stats('./dataset', max_files=200, device='cpu')
torch.save({'mean': mean, 'std': std}, './dataset/mel_stats.pt')
print('Saved mel_stats.pt')
"
```

If you already have a stale `mel_stats.pt` from a different `n_mels` value,
delete it first:

```bash
rm ./dataset/mel_stats.pt
```

## Training

```bash
python -m src.train \
  --data_dir ./dataset \
  --stats_file ./dataset/mel_stats.pt \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --device cuda \
  --checkpoint_dir ./checkpoints
```

Key flags:
| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `./dataset` | Root of dataset tree |
| `--stats_file` | `./dataset/mel_stats.pt` | Pre-computed mel stats |
| `--batch_size` | `4` | Per-GPU batch size |
| `--epochs` | `50` | Training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--device` | `cuda` | `cuda`, `cpu`, or `mps` |
| `--checkpoint_dir` | `./checkpoints` | Checkpoint directory |
| `--num_workers` | `4` | DataLoader workers |

Training logs include per-step loss, token accuracy, and periodic
validation. Checkpoints are saved every 5 epochs and on best validation
loss.

## Inference

Generate a chart from audio.  Choose one BPM source:

```bash
# Constant BPM
python main.py --checkpoint checkpoints/best.pt --audio track.mp3 \
    --bpm 175.0 --constant "12,5" --title "My Song"

# Variable BPM from a metadata file
python main.py --checkpoint checkpoints/best.pt --audio track.mp3 \
    --chart_metadata meta.json --constant "12,5"
```

The ``--chart_metadata`` JSON format:

```json
{
  "title": "Song Title",
  "artist": "Artist Name",
  "bpm10_list": [
    {"bpm10": 1750, "change_timestamp_ms": 0},
    {"bpm10": 2000, "change_timestamp_ms": 60000}
  ]
}
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Model checkpoint (.pt) |
| `--audio` | *(required)* | Input audio file (mp3/wav) |
| `--bpm` | — | Constant BPM (mutually exclusive with `--chart_metadata`) |
| `--chart_metadata` | — | JSON with `title`/`artist`/`bpm10_list` |
| `--constant` | *(required)* | Target difficulty, e.g. `"12,5"` |
| `--max_tokens` | `8000` | Max tokens to generate |
| `--temperature` | `1.0` | `0.0` = greedy, `1.0` = sampling |
| `--device` | `cuda` | `cuda`, `cpu`, or `mps` |
| `--output` | *(stdout)* | Save path for generated chart JSON |
| `--mel_stats` | `./dataset/mel_stats.pt` | Mel normalization stats |
| `--title` | — | Song title (embedded in output) |
| `--artist` | — | Song artist (embedded in output) |

The output JSON follows the same ``processed.json`` schema (includes ``title``,
``artist``, ``cabinet``, ``version``, ``charts`` with ``constant``,
``designer``, ``bpm10_list``, and ``notes``).

### Speed note

Generation is currently **O(n²)** in token count (no KV‑caching).  It runs
fast for the first ~500 tokens and slows down progressively.  For a
full‑length chart (~5 000 tokens), expect several minutes on GPU.
KV‑caching will be added in a future update.
