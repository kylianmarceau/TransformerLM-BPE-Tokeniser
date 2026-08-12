# CSC3043S Assignment 1 - Transformer Language-Model Training

This repository contains the implementation and experiment artifacts for
CSC3043S Assignment 1. Task 1 implements an incremental byte-level BPE
tokenizer, studies vocabulary-size compression, and encodes TinyStories as
memory-mappable `uint16` NumPy arrays.

## Repository layout

- `src/tokenizer.py`: byte-level BPE training, encoding, decoding, and
  tokenizer serialization.
- `tests/test_tokenizer.py`: tokenizer correctness and determinism checks.
- `script/vocab_study.py`: Section 3.4 vocabulary-size study.
- `script/encode_corpus.py`: Section 3.5 full-corpus tokenizer training and
  streaming corpus encoding.
- `results/vocab_study/`: vocabulary-study measurements, tokenizer files, and
  the compression figure.
- `tokenizers/`: the tokenizers trained on the full training corpus.
- `results/corpus_encoding.json`: corpus-encoding timings, token counts,
  environment information, and artifact hashes.

The raw datasets and encoded `.npy` arrays are intentionally excluded from Git.

## Data setup

Obtain the TinyStories V2 GPT-4 files from the course resources and place them
at:

```text
datasets/TinyStoriesV2-GPT4-train.txt
datasets/TinyStoriesV2-GPT4-valid.txt
```

Documents are expected to be separated by `<|endoftext|>`. The final 2,000
validation documents are excluded from the vocabulary-size study. They must
remain reserved for the final test evaluation described in the assignment.

## Installation

Task 1 requires Python, NumPy, and the third-party `regex` package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy regex
```

The recorded Task 1 corpus-encoding run used:

- Python 3.14.6
- NumPy 2.5.1
- macOS 27.0 on arm64

The `regex` package version was not captured in the current manifest. PyTorch
is not used by Task 1; its exact version and installation command must be added
here once the model-training environment is fixed.

## Reproducing Task 1

Run the tokenizer checks from the repository root:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Run the vocabulary-size study on the eligible validation documents:

```bash
python3 script/vocab_study.py
```

This trains tokenizers with vocabulary sizes 1,000, 2,000, 4,000, 8,000, and
16,000. It writes the measurements to
`results/vocab_study/vocab_study.csv`, saves the study tokenizers under
`results/vocab_study/tokenizers/`, and creates
`results/vocab_study/vocab_compression.svg`.

Train the selected full-corpus tokenizers and encode both corpus splits:

```bash
python3 script/encode_corpus.py
```

The default main vocabulary has 4,000 tokens and the comparison vocabulary has
1,000 tokens. Existing tokenizer and encoded-array artifacts are reused. To
deliberately replace them and record fresh timings, run:

```bash
python3 script/encode_corpus.py --force
```

The outputs are:

```text
tokenizers/vocab_4000/{vocab.tsv,merges.tsv}
tokenizers/vocab_1000/{vocab.tsv,merges.tsv}
datasets/encoded/train_vocab_4000.npy
datasets/encoded/valid_vocab_4000.npy
datasets/encoded/train_vocab_1000.npy
datasets/encoded/valid_vocab_1000.npy
results/corpus_encoding.json
```

The `.npy` arrays can be opened without loading them fully into memory:

```python
tokens = np.load("datasets/encoded/train_vocab_4000.npy", mmap_mode="r")
```

## Recorded Task 1 outputs

The current full-corpus run recorded the following values in
`results/corpus_encoding.json`:

| Item | Vocabulary | Recorded value |
| --- | ---: | ---: |
| BPE training on the full training corpus | 4,000 | 541.69 s |
| Training-corpus encoding | 4,000 | 573.71 s |
| Validation-corpus encoding | 4,000 | 5.75 s |
| Training tokens | 4,000 | 559,840,775 |
| Validation tokens | 4,000 | 5,652,645 |
| Training-corpus encoding | 1,000 | 578.12 s |
| Validation-corpus encoding | 1,000 | 5.88 s |
| Training tokens | 1,000 | 703,090,431 |
| Validation tokens | 1,000 | 7,096,737 |

The vocabulary-study figure used for Task 1 Q3 is generated directly from the
measurements in `results/vocab_study/vocab_study.csv` by
`script/vocab_study.py`:

![Vocabulary size versus compression](results/vocab_study/vocab_compression.svg)

## Reproducibility notes

- IDs 0-255 represent the corresponding byte values; ID 256 is
  `<|endoftext|>` in the saved Task 1 tokenizers.
- Vocabulary and merge files use hexadecimal byte strings so arbitrary bytes
  can be serialized without loss.
- `results/corpus_encoding.json` records SHA-256 hashes for the full-corpus
  tokenizer artifacts.
- The saved encoded arrays are one-dimensional little-endian `uint16` arrays
  compatible with `np.load(..., mmap_mode="r")`.
- Before final submission, add the exact PyTorch version, commands for Tasks
  2-5, and the source command for every additional report figure.
