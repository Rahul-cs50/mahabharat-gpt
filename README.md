# Mahabharata GPT

Decoder-only Transformer language model trained from scratch on a self-curated Mahabharata/Sanskrit corpus using PyTorch.

This repository packages the training notebook into reusable Python modules, records the final 32M-parameter configuration, and includes experiment outputs from the training runs.

## Highlights

| Metric | Result |
|---|---:|
| Architecture | Decoder-only Transformer |
| Parameters | ~32.1M trainable |
| Vocabulary | 50,257 |
| Context length | 256 |
| Embedding dimension | 256 |
| Transformer layers | 8 |
| Attention heads | 8 |
| Tokenizer | GPT-2 BPE |
| Framework | PyTorch |
| Training GPU | RTX 4090 / RunPod |
| Final train perplexity | 13.15 |
| Final validation perplexity | 21.26 |

## Project Structure

```text
mahabharata-gpt/
├── configs/
│   └── 32m_config.json
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── model.py
│   ├── tokenizer.py
│   └── train.py
├── notebooks/
│   └── mahabharata_gpt_experiments.ipynb
├── results/
│   ├── gita_gpt_los.png
│   ├── gita_gpt_loss_overfit.png
│   ├── loss-plot.pdf
│   └── samples.txt
├── checkpoints/
│   └── README.md
├── requirements.txt
└── README.md
```

## Dataset

The model was trained on a self-curated Sanskrit/Mahabharata text corpus. The corpus was processed as a continuous token sequence using the GPT-2 BPE tokenizer and split into training and validation portions.

The dataset is not included in this repository. Source text licensing should be reviewed before publishing any full corpus extracts.

## Architecture

The model follows a GPT-style causal language-modeling stack:

```text
Input tokens
  -> token embeddings + positional embeddings
  -> Transformer blocks x 8
      -> LayerNorm
      -> multi-head causal self-attention
      -> residual connection
      -> LayerNorm
      -> feed-forward network
      -> residual connection
  -> final LayerNorm
  -> linear language-model head
  -> next-token logits
```

The final run uses the configuration in `configs/32m_config.json`.

## Training

Training uses next-token prediction with cross-entropy loss, AdamW optimization, gradient clipping, periodic train/validation evaluation, checkpoint saving, and autoregressive sample generation during training.

## Results

### Final Run

| Metric | Result |
|---|---:|
| Train loss | ~2.58 |
| Validation loss | ~3.05 |
| Train perplexity | 13.15 |
| Validation perplexity | 21.26 |

### Overfitting Experiment

The first small-corpus experiment strongly overfit:

| Metric | Result |
|---|---:|
| Training loss | ~0.04 |
| Validation loss | ~6.08 |

Scaling the corpus substantially improved validation behavior, showing how important dataset size is when training even a small decoder-only model from scratch.

## Loss Curves

Available artifacts:

![Bhagavad Gita training curve](results/gita_gpt_los.png)

![Small-corpus overfitting curve](results/gita_gpt_loss_overfit.png)

The original PDF loss curve is also included at `results/loss-plot.pdf`.

Additional Mahabharata loss plots and generation screenshots were supplied for the project write-up, but Windows file permissions blocked direct copying from the provided paths during packaging. They can be added under `results/` with names such as `mahabharata_32m_loss.png` and linked here.

## Generation

The model supports autoregressive generation with temperature sampling, top-k filtering, configurable context length, and custom prompts.

Example outputs are saved in `results/samples.txt`.

```text
Prompt: धर्म
Generated: धर्म, with austerities and self-control, learned in this world...
```

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Train from a text file:

```bash
python src/train.py --data path/to/corpus.txt --config configs/32m_config.json
```

Generate from a checkpoint:

```bash
python src/generate.py --checkpoint path/to/mahabharata_30m_weights.pt --prompt "धर्म"
```

Evaluate a checkpoint:

```bash
python src/evaluate.py --checkpoint path/to/mahabharata_30m_weights.pt --data path/to/corpus.txt
```

## Checkpoints

The trained checkpoint `mahabharata_30m_weights.pt` is intentionally not committed because it is large. Use Git LFS, a GitHub Release, or Hugging Face Hub for publishing model weights.

## Limitations

This is an educational/research-scale model trained on a relatively small domain-specific corpus. It is not intended to provide authoritative interpretations of the Mahabharata or Sanskrit texts. Generated text may be ungrammatical, repetitive, or hallucinated.

## Future Work

- Rotary positional embeddings
- Grouped-query or multi-query attention
- Longer context windows
- Larger model configurations
- Mixture-of-Experts variants
- Improved tokenization and evaluation benchmarks

## License

Add an appropriate license after reviewing the licensing terms of the source corpus and tokenizer dependencies.
