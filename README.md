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

Before training, compute per-channel mel spectrogram mean/std:

```bash
python -c "
from src.dataloader import compute_mel_stats
import torch
mean, std = compute_mel_stats('./dataset', max_files=200, device='cpu')
torch.save({'mean': mean, 'std': std}, './dataset/mel_stats.pt')
print('Saved mel_stats.pt')
"
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

```python
from src.model import ChartGPT
from src.dataloader import MaiMaiDataset

# Load model
model = ChartGPT()
model.load_state_dict(torch.load("checkpoints/best.pt")["model_state_dict"])
model.eval()

# Load a sample
ds = MaiMaiDataset("./dataset", split="val", mel_stats=(mean, std))
sample = ds[0]

# Generate
tokens = model.generate(
    sample["spectrogram"].unsqueeze(0),
    sample["bpm_signal"].unsqueeze(0),
    sample["chart_constant"].unsqueeze(0),
    max_len=8000,
    temperature=1.0,
)

# Decode
from src.tokenizer import ChartTokenizer
notes = ChartTokenizer._decode_notes_standalone(tokens)
```
