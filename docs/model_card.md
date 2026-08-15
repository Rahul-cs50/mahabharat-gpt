# Mahabharata GPT Model Card

## Model Details

Mahabharata GPT is a decoder-only Transformer language model trained from scratch on a self-curated Mahabharata/Sanskrit corpus.

| Item | Value |
|---|---:|
| Parameters | ~32.1M trainable |
| Context length | 256 tokens |
| Vocabulary | 50,257 |
| Tokenizer | GPT-2 BPE |
| Layers | 8 |
| Attention heads | 8 |
| Embedding dimension | 256 |
| Framework | PyTorch |
| Training GPU | RTX 4090 / RunPod |

## Intended Use

This model is intended as an educational and research artifact for studying small-scale language-model training, Sanskrit/Mahabharata domain modeling, loss behavior, and autoregressive generation.

## Not Intended For

The model should not be treated as an authoritative source for Sanskrit translation, scriptural interpretation, historical claims, or religious guidance.

## Training Data

The model was trained on a self-curated text corpus related to the Mahabharata/Sanskrit domain. The corpus is not included in this repository pending source licensing review.

## Evaluation

Final reported metrics:

| Metric | Result |
|---|---:|
| Train loss | ~2.58 |
| Validation loss | ~3.05 |
| Train perplexity | 13.15 |
| Validation perplexity | 21.26 |

An earlier small-corpus experiment showed severe overfitting, with training loss around `0.04` and validation loss around `6.08`.

## Training Configuration

| Parameter | Value |
|---|---:|
| Architecture | Decoder-only Transformer |
| Layers | 8 |
| Attention heads | 8 |
| Embedding dimension | 256 |
| Context length | 256 |
| Vocabulary size | 50,257 |
| Tokenizer | GPT-2 BPE |
| Framework | PyTorch |
| Training GPU | RTX 4090 / RunPod |

## Limitations

Generated text may be repetitive, ungrammatical, factually incorrect, or stylistically inconsistent. The model is small and domain-specific, and it has not been instruction-tuned or safety-tuned.

## Checkpoint

The trained checkpoint is stored at:

```text
checkpoints/mahabharata_30m_weights.pt
```

It is tracked using Git LFS.

## Reproducibility

The repository contains the model configuration, training and inference code, experiment notebook, results, and final checkpoint. The training corpus itself is not included because its source licensing is still under review.

See the main repository README and `checkpoints/README.md` for setup and checkpoint usage instructions.

## License

No project license is currently specified. Source-corpus licensing should be reviewed before publishing corpus extracts or selecting a final project license.
