# Maimai ChartGPT

Two‑stage model that generates maimai charts from audio:
**Stage 1** detects onsets; **Stage 2** generates notes within each onset block.

## Setup

```bash
uv sync
```

Requires `ffmpeg` on PATH for audio decoding.

## 0. Pre‑process the data

```bash
# Compute mel spectrogram normalization stats (per‑channel mean/std)
uv run python -c "from src.dataloader import compute_mel_stats; import torch; mean, std = compute_mel_stats('./dataset', max_files=200, device='cpu'); torch.save({'mean': mean, 'std': std}, './dataset/mel_stats.pt'); print('Saved mel_stats.pt')"
```

## 1. Train the model

**Stage 1 — Onset detector** (pretrains base CNN, then adds DifficultyGate):

```bash
uv run python -m src.stage1_train --data_dir ./dataset --stats_file ./dataset/mel_stats.pt --pretrain_batch_size 4 --gate_batch_size 4 --pretrain_epochs 20 --gate_epochs 30 --device cuda --checkpoint_dir ./checkpoints/stage1
```

**Stage 2 — Note generator** (scheduled sampling with trained Stage 1):

```bash
uv run python -m src.stage2_train --data_dir ./dataset --stats_file ./dataset/mel_stats.pt --stage1_checkpoint ./checkpoints/stage1/gate_best.pt --batch_size 4 --epochs 50 --device cuda --checkpoint_dir ./checkpoints/stage2
```

### Pause and resume

Press **Ctrl‑C** to pause safely (writes `latest.pt`). Resume with `--resume`:

```bash
uv run python -m src.stage1_train --resume ...  # or stage2_train
```

## 2. Inference

```bash
# Two-stage (onset detection → note generation)
uv run python main.py --stage2_checkpoint ./checkpoints/stage2/best.pt --stage1_checkpoint ./checkpoints/stage1/gate_best.pt --audio track.mp3 --bpm 175.0 --constant "12.5" --title "My Song" --output chart.json
```

`--constant` uses dot notation: `12.5` = 12★5, `9.0` = 9★0.

`--chart_metadata` is available for variable-BPM songs (mutually exclusive with `--bpm`):

```bash
uv run python main.py --stage2_checkpoint ./checkpoints/stage2/best.pt --stage1_checkpoint ./checkpoints/stage1/gate_best.pt --audio track.mp3 --chart_metadata meta.json --constant "13.2"
```

## Validate difficulty‑density correlation

A diagnostic script that checks whether higher chart constants produce more notes (Pearson ρ). Useful for verifying the model's difficulty conditioning works.

```bash
# Ground truth only (fast, no GPU needed)
uv run python -m src.validate_density --data_dir ./dataset

# Include trained model evaluation (slower, needs GPU)
uv run python -m src.validate_density --data_dir ./dataset --stage2_checkpoint ./checkpoints/stage2/best.pt
```

## Project Structure

```
src/
  tokenizer.py          Vocabulary + ChartTokenizer (encode/decode)
  dataloader.py          MaiMaiDataset, mel spectrograms, collation
  model.py               ChartGPT (audio encoder + transformer decoder)
  train.py               Original single‑stage training loop
  stage1_model.py        OnsetDetector + DifficultyGate → Stage1Model
  stage1_train.py        Two‑phase Stage 1 training
  stage2_model.py        Stage2Encoder + Stage2Decoder → Stage2Model
  stage2_train.py        Stage 2 training with scheduled onset sampling
  token_validator.py     ChartValidator (Stage 1) + Stage2Validator
  inference.py           Two‑stage inference pipeline
  validate_density.py    Difficulty‑density correlation (ρ metric)
main.py                  Unified CLI entrypoint
```
