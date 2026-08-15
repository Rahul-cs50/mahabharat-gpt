# Checkpoints

Large model files are intentionally excluded from git.

The trained checkpoint used for the final run was named:

```text
mahabharata_30m_weights.pt
```

Recommended publishing options:

- GitHub Release
- Git LFS
- Hugging Face Hub

After downloading the checkpoint locally, use it with:

```bash
python src/generate.py --checkpoint path/to/mahabharata_30m_weights.pt --prompt "धर्म"
python src/evaluate.py --checkpoint path/to/mahabharata_30m_weights.pt --data path/to/corpus.txt
```
