# Model Checkpoint

This directory contains the trained checkpoint from the final Mahabharata GPT training run.

## Checkpoint

| Item | Value |
|---|---:|
| File | `mahabharata_30m_weights.pt` |
| Size | ~125 MB (130,581,125 bytes) |
| Model | Mahabharata GPT |
| Parameters | ~32.1M trainable |
| Context length | 256 tokens |
| Architecture | Decoder-only Transformer |
| Tokenizer | GPT-2 BPE |

The checkpoint is managed using **Git LFS** because of its file size.

## Download

Clone the repository normally:

```bash
git clone https://github.com/Rahul-cs50/mahabharat-gpt.git
cd mahabharat-gpt
```

Make sure Git LFS is installed and enabled:

```bash
git lfs install
git lfs pull
```

The checkpoint will then be available at:

```text
checkpoints/mahabharata_30m_weights.pt
```

## Generate Text

Use the trained checkpoint with:

```bash
python src/generate.py \
  --checkpoint checkpoints/mahabharata_30m_weights.pt \
  --prompt "धर्म"
```

You can replace the prompt with another Sanskrit or Mahabharata-related prompt.

## Evaluate

To evaluate the checkpoint on a text corpus:

```bash
python src/evaluate.py \
  --checkpoint checkpoints/mahabharata_30m_weights.pt \
  --data path/to/corpus.txt
```

The original training corpus is not included in this repository.

## Training Results

The final training run achieved approximately:

| Metric | Result |
|---|---:|
| Parameters | ~32.1M |
| Training loss | ~2.58 |
| Validation loss | ~3.05 |
| Training perplexity | 13.15 |
| Validation perplexity | 21.26 |

These results correspond to the final checkpoint.

## Verification

The checkpoint is tracked using Git LFS. After cloning the repository, run:

```bash
git lfs install
git lfs pull
```

Then verify that the checkpoint exists locally before running generation or evaluation.

## Notes

The checkpoint contains the learned model weights. The model configuration is available in:

```text
configs/32m_config.json
```

The training and inference implementation is available under:

```text
src/
```

See the main repository README for the full architecture, training methodology, experiments, and limitations.
