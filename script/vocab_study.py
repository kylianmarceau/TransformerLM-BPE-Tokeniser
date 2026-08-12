"""Run the Section 3.4 BPE vocabulary-size study.

The default experiment trains byte-level BPE tokenizers with vocabulary
sizes 1,000, 2,000, 4,000, 8,000, and 16,000 on the TinyStories validation
documents, excluding the final 2,000 documents reserved for final testing.

Example:
    python3 scripts/vocab_study.py

For a quick smoke test:
    python3 scripts/vocab_study.py --vocab-sizes 300 400 --max-documents 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer import (  # noqa: E402
    BPETokenizer,
    save_merges,
    save_vocab,
    train_bpe,
)


EOT = "<|endoftext|>"
DEFAULT_VOCAB_SIZES = (1_000, 2_000, 4_000, 8_000, 16_000)
DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT / "datasets" / "TinyStoriesV2-GPT4-valid.txt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "vocab_study"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Assignment 1 Section 3.4 vocabulary study."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
        help="TinyStories validation file used for the study.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for results and trained tokenizer files.",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_VOCAB_SIZES),
        help="Vocabulary sizes to compare.",
    )
    parser.add_argument(
        "--reserved-test-documents",
        type=int,
        default=2_000,
        help="Number of documents at the end of validation to exclude.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional fixed prefix of eligible documents for a quick run.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Model width used to calculate embedding and untied LM-head parameters.",
    )
    return parser.parse_args()


def load_study_corpus(
    input_path: Path,
    reserved_test_documents: int,
    max_documents: int | None,
) -> tuple[str, int]:
    """Return eligible validation documents joined by hard EOT boundaries."""
    if reserved_test_documents < 0:
        raise ValueError("reserved_test_documents cannot be negative")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive")
    if not input_path.is_file():
        raise FileNotFoundError(f"Validation file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    documents = [document for document in text.split(EOT) if document]

    if reserved_test_documents >= len(documents):
        raise ValueError(
            "reserved_test_documents must be smaller than the number "
            f"of documents ({len(documents)})"
        )

    eligible = (
        documents[:-reserved_test_documents]
        if reserved_test_documents
        else documents
    )
    if max_documents is not None:
        eligible = eligible[:max_documents]

    # Delimiters are included in both the encoded token count and the raw
    # character/byte counts. This makes the stated compression ratios use
    # exactly the same text that is passed to encode().
    return EOT.join(eligible), len(eligible)


def tokenizer_paths(output_dir: Path, vocab_size: int) -> tuple[Path, Path]:
    tokenizer_dir = output_dir / "tokenizers" / f"vocab_{vocab_size}"
    return tokenizer_dir / "vocab.tsv", tokenizer_dir / "merges.tsv"


def display_path(path: Path) -> str:
    """Prefer a project-relative artifact path, otherwise use its full path."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def train_and_measure(
    corpus_path: Path,
    corpus_text: str,
    document_count: int,
    requested_vocab_size: int,
    d_model: int,
    output_dir: Path,
) -> dict[str, int | float | str]:
    """Train one tokenizer, encode the corpus, and return its measurements."""
    if not 257 <= requested_vocab_size <= 65_536:
        raise ValueError(
            f"vocab size {requested_vocab_size} must be between 257 and 65,536"
        )

    train_started = time.perf_counter()
    vocab, merges = train_bpe(
        str(corpus_path),
        requested_vocab_size,
        [EOT],
    )
    training_seconds = time.perf_counter() - train_started

    tokenizer = BPETokenizer(vocab, merges, [EOT])
    encode_started = time.perf_counter()
    token_ids = tokenizer.encode(corpus_text)
    encoding_seconds = time.perf_counter() - encode_started

    if not token_ids:
        raise RuntimeError("The study corpus encoded to zero tokens")

    vocab_path, merges_path = tokenizer_paths(output_dir, requested_vocab_size)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab, str(vocab_path))
    save_merges(merges, str(merges_path))

    byte_count = len(corpus_text.encode("utf-8"))
    character_count = len(corpus_text)
    token_count = len(token_ids)

    return {
        "requested_vocab_size": requested_vocab_size,
        "actual_vocab_size": len(vocab),
        "merge_count": len(merges),
        "document_count": document_count,
        "character_count": character_count,
        "byte_count": byte_count,
        "token_count": token_count,
        "bytes_per_token": byte_count / token_count,
        "characters_per_token": character_count / token_count,
        "training_seconds": training_seconds,
        "encoding_seconds": encoding_seconds,
        "embedding_parameters": len(vocab) * d_model,
        "lm_head_parameters": len(vocab) * d_model,
        "vocab_dependent_parameters": 2 * len(vocab) * d_model,
        "vocab_path": display_path(vocab_path),
        "merges_path": display_path(merges_path),
    }


def save_csv(rows: list[dict[str, int | float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(
    rows: list[dict[str, int | float | str]],
    path: Path,
    metadata: dict[str, int | str | bool | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": rows}
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def save_svg_plot(
    rows: list[dict[str, int | float | str]],
    path: Path,
) -> None:
    """Create a dependency-free compression plot as an SVG file."""
    width, height = 900, 560
    left, right, top, bottom = 90, 40, 55, 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [int(row["requested_vocab_size"]) for row in rows]
    byte_values = [float(row["bytes_per_token"]) for row in rows]
    char_values = [float(row["characters_per_token"]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min = 0.0
    y_max = max(byte_values + char_values) * 1.12

    def x_position(value: float) -> float:
        if x_max == x_min:
            return left + plot_width / 2
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.grid{stroke:#dce2ea;stroke-width:1}.axis{stroke:#172033;stroke-width:1.5}.bytes{stroke:#1769aa;fill:none;stroke-width:3}.chars{stroke:#d1495b;fill:none;stroke-width:3}</style>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="20" font-weight="bold">BPE vocabulary size vs compression</text>',
    ]

    for tick in range(6):
        value = y_max * tick / 5
        y = y_position(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="13">{value:.2f}</text>')

    for value in x_values:
        x = x_position(value)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-size="13">{value:,}</text>')

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 22}" text-anchor="middle" font-size="15">Vocabulary size</text>',
            f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-size="15" transform="rotate(-90 24 {top + plot_height / 2})">Compression ratio</text>',
        ]
    )

    for values, css_class, colour in (
        (byte_values, "bytes", "#1769aa"),
        (char_values, "chars", "#d1495b"),
    ):
        points = " ".join(
            f"{x_position(x):.2f},{y_position(y):.2f}"
            for x, y in zip(x_values, values)
        )
        parts.append(f'<polyline class="{css_class}" points="{points}"/>')
        for x, y in zip(x_values, values):
            parts.append(f'<circle cx="{x_position(x):.2f}" cy="{y_position(y):.2f}" r="4.5" fill="{colour}"/>')

    legend_x = left + 18
    parts.extend(
        [
            f'<line class="bytes" x1="{legend_x}" y1="{top + 18}" x2="{legend_x + 36}" y2="{top + 18}"/>',
            f'<text x="{legend_x + 45}" y="{top + 23}" font-size="14">Bytes per token</text>',
            f'<line class="chars" x1="{legend_x + 175}" y1="{top + 18}" x2="{legend_x + 211}" y2="{top + 18}"/>',
            f'<text x="{legend_x + 220}" y="{top + 23}" font-size="14">Characters per token</text>',
            "</svg>",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def print_result(row: dict[str, int | float | str]) -> None:
    print(
        f"vocab={int(row['requested_vocab_size']):>5,}  "
        f"tokens={int(row['token_count']):>10,}  "
        f"bytes/token={float(row['bytes_per_token']):.4f}  "
        f"chars/token={float(row['characters_per_token']):.4f}  "
        f"vocab-params={int(row['vocab_dependent_parameters']):>10,}  "
        f"train={float(row['training_seconds']):.2f}s  "
        f"encode={float(row['encoding_seconds']):.2f}s"
    )


def main() -> None:
    args = parse_args()
    vocab_sizes = sorted(set(args.vocab_sizes))
    output_dir = args.output_dir.resolve()
    if args.d_model <= 0:
        raise ValueError("d_model must be positive")

    corpus_text, document_count = load_study_corpus(
        args.input.resolve(),
        args.reserved_test_documents,
        args.max_documents,
    )
    print(
        f"Study corpus: {document_count:,} documents, "
        f"{len(corpus_text):,} characters, "
        f"{len(corpus_text.encode('utf-8')):,} bytes"
    )
    print("The EOT delimiters are included in compression measurements.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | float | str]] = []

    # train_bpe accepts a path, so create the exact eligible corpus once and
    # reuse it for every vocabulary size to keep the comparison controlled.
    with tempfile.TemporaryDirectory(prefix="vocab-study-") as temporary:
        corpus_path = Path(temporary) / "study_corpus.txt"
        corpus_path.write_text(corpus_text, encoding="utf-8", newline="")

        for vocab_size in vocab_sizes:
            print(f"\nTraining vocabulary size {vocab_size:,}...")
            row = train_and_measure(
                corpus_path,
                corpus_text,
                document_count,
                vocab_size,
                args.d_model,
                output_dir,
            )
            rows.append(row)
            print_result(row)

    metadata: dict[str, int | str | bool | None] = {
        "section": "3.4",
        "input_path": str(args.input.resolve()),
        "reserved_test_documents": args.reserved_test_documents,
        "max_documents": args.max_documents,
        "document_count": document_count,
        "special_token": EOT,
        "d_model": args.d_model,
        "untied_lm_head": True,
        "delimiters_included_in_metrics": True,
    }
    csv_path = output_dir / "vocab_study.csv"
    json_path = output_dir / "vocab_study.json"
    plot_path = output_dir / "vocab_compression.svg"
    save_csv(rows, csv_path)
    save_json(rows, json_path, metadata)
    save_svg_plot(rows, plot_path)

    print("\nSaved results:")
    for path in (csv_path, json_path, plot_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
