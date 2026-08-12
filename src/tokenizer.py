
# imports

from __future__ import annotations
from collections import Counter
from typing import Iterable, Iterator

import regex

# new imports for 3.2 
from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor, wait,)
import heapq # supports best pair selection
import multiprocessing
import os



END_OF_TEXT = "<|endoftext|>"

# gpt2 pre-tokenizer regex from appendix A
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_pretoken_regex = regex.compile(GPT2_PAT)

pre_token_chunksize = 8*1024*1024
parallel_pretoken_threshold = 16*1024*1024
max_pretoken_workers = 8
Pair = tuple[bytes, bytes]



def validate_special_tokens(special_tokens: list[str] | None, ) -> list[str]:
    # validate and copy the configured special token list
    tokens = list(special_tokens or [])

    if any(token == "" for token in tokens):
        raise ValueError("Special tokens cannot be empty")

    if len(tokens) != len(set(tokens)):
        raise ValueError("Special tokens must be unique")

    return tokens

def split_on_special_tokens(text: str, special_tokens: list[str], keep: bool,) -> list[str]:
    # split at the special tokens, toptionallu retaining them 
    if not special_tokens:
        return[text] 

    #macth longer special tokens first 
    ordered = sorted(special_tokens ,key=lambda token: (-len(token), token),)
    alternatives = "|".join(regex.escape(token) for token in ordered)

    if keep:
        pattern = f"({alternatives})"
    else:
        pattern = f"(?:{alternatives})"

    return regex.split(pattern, text)

def pretokenize(text: str, ) -> list[tuple[bytes, ...]]:
    # convert gpt2 pretokens into tuples of individual bytes
    pretokens = []
    for match in GPT2_pretoken_regex.finditer(text):
        raw_bytes = match.group(0).encode("utf-8")
        symbols = tuple(bytes([byte_value])for byte_value in raw_bytes) 
        pretokens.append(symbols)

    return pretokens

def count_pretokens(text: str, special_tokens: list[str], ) -> Counter:
    # count byte pre tokens wihtout counting the special tokens
    counts: Counter = Counter()

    for chunk in split_on_special_tokens(text, special_tokens, keep=False,):
        if not chunk:
            continue

        counts.update(pretokenize(chunk))

    return counts

def count_pairs(word_freqs: dict[tuple[bytes, ...], int], ) -> Counter: #CHANGE signature so refers bytes
    """
    Count how often each adjacent pair of symbols occurs, weighted by pre-token frequency.
    params:
        word_freqs: dict mapping a pre-token (tuple of symbol strings) -> its corpus count
    returns:
        Counter mapping (left_symbol, right_symbol) -> total count
    """

    # count adjacebnt byte symbol paris weighted by frequency
    pair_counts: Counter = Counter()
    for symbols, frequency in word_freqs.items():
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += frequency
    return pair_counts

def merge_word(symbols: tuple[bytes, ...], pair: tuple[bytes, bytes], )->tuple[bytes, ...]: # CHANGE to bytesz
    """
    Replace every occurrence of `pair` in `symbols` with the single merged symbol.
    E.g. merge_word(("n", "e", "w", "e", "s", "t"), ("s", "t")) -> ("n", "e", "w", "e", "st")
    params:
        symbols: tuple of symbol strings representing one pre-token
        pair:    (left, right) tuple of symbols to merge
    returns:
        new tuple of symbols
    """
    merged = []
    index = 0
    while index < len(symbols):
        # If the pair starts here, append the concatenated symbol and skip forward by 2.
        if index < len(symbols) - 1 and (symbols[index], symbols[index + 1]) == pair:
            merged.append(symbols[index] + symbols[index+ 1])
            index += 2
        # Otherwise keep the current symbol and advance by 1.
        else:
            merged.append(symbols[index])
            index += 1
    return tuple(merged)

# NEW 3.2-------------------------------------------------------------------------------------------------
class _PairHeapEntry:
    """Pair ordered by highest count, then greatest pair."""

    __slots__ = ("count", "pair")

    def __init__(
        self,
        count: int,
        pair: Pair,
    ):
        self.count = count
        self.pair = pair

    def __lt__(
        self,
        other: "_PairHeapEntry",
    ) -> bool:
        # heapq is normally a minimum heap, so reverse
        # the comparisons to obtain maximum behaviour.
        if self.count != other.count:
            return self.count > other.count

        # Required tie-break:
        # lexicographically greatest pair.
        return self.pair > other.pair


def _build_pair_indexes(
    words: list[tuple[bytes, ...]],
    frequencies: list[int],
) -> tuple[
    Counter,
    dict[Pair, set[int]],
    list[_PairHeapEntry],
]:
    """Count pairs once and record which words contain them."""

    pair_counts: Counter = Counter()
    pair_to_word_ids: dict[
        Pair,
        set[int],
    ] = {}

    for word_id, symbols in enumerate(words):
        frequency = frequencies[word_id]

        # Local Counter is important because a pair can
        # occur more than once in the same pre-token.
        local_counts = Counter(
            zip(symbols, symbols[1:])
        )

        for pair, occurrences in local_counts.items():
            pair_counts[pair] += (
                occurrences * frequency
            )

            pair_to_word_ids.setdefault(
                pair,
                set(),
            ).add(word_id)

    heap = [
        _PairHeapEntry(count, pair)
        for pair, count in pair_counts.items()
    ]
    heapq.heapify(heap)

    return pair_counts, pair_to_word_ids, heap


def _pop_best_pair(
    pair_counts: Counter,
    heap: list[_PairHeapEntry],
) -> Pair | None:
    """Return the best pair, ignoring outdated entries."""

    while heap:
        candidate = heapq.heappop(heap)

        current_count = pair_counts.get(
            candidate.pair
        )

        if current_count == candidate.count:
            return candidate.pair

    return None


def _merge_pair_incrementally(
    best_pair: Pair,
    words: list[tuple[bytes, ...]],
    frequencies: list[int],
    pair_counts: Counter,
    pair_to_word_ids: dict[Pair, set[int]],
    heap: list[_PairHeapEntry],
) -> None:
    """Update only words containing the selected pair."""

    affected_word_ids = tuple(
        pair_to_word_ids.get(
            best_pair,
            (),
        )
    )

    if not affected_word_ids:
        raise RuntimeError(
            "Pair index is inconsistent "
            "with pair counts"
        )

    # Collect the net changes across all affected
    # pre-tokens before updating the global counts.
    count_deltas: Counter = Counter()

    for word_id in affected_word_ids:
        old_symbols = words[word_id]
        frequency = frequencies[word_id]

        # Step 1: count the affected word's old pairs.
        old_local_counts = Counter(
            zip(
                old_symbols,
                old_symbols[1:],
            )
        )

        # Step 2: apply the selected merge.
        new_symbols = merge_word(
            old_symbols,
            best_pair,
        )

        # Step 3: count the affected word's new pairs.
        new_local_counts = Counter(
            zip(
                new_symbols,
                new_symbols[1:],
            )
        )

        # Remove the old pair contributions.
        for pair, occurrences in (
            old_local_counts.items()
        ):
            count_deltas[pair] -= (
                occurrences * frequency
            )

            indexed_words = (
                pair_to_word_ids.get(pair)
            )

            if indexed_words is not None:
                indexed_words.discard(word_id)

                if not indexed_words:
                    del pair_to_word_ids[pair]

        # Add the new pair contributions.
        for pair, occurrences in (
            new_local_counts.items()
        ):
            count_deltas[pair] += (
                occurrences * frequency
            )

            pair_to_word_ids.setdefault(
                pair,
                set(),
            ).add(word_id)

        words[word_id] = new_symbols

    # Apply the net changes to the global counts.
    for pair, delta in count_deltas.items():
        if delta == 0:
            continue

        new_count = (
            pair_counts.get(pair, 0)
            + delta
        )

        if new_count < 0:
            raise RuntimeError(
                "Incremental pair count "
                "became negative"
            )

        if new_count == 0:
            pair_counts.pop(pair, None)
        else:
            pair_counts[pair] = new_count

            heapq.heappush(
                heap,
                _PairHeapEntry(
                    new_count,
                    pair,
                ),
            )

def train_bpe_OLD(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    """Train a byte-level BPE tokenizer.

    Returns:
        vocab: token ID -> token bytes
        merges: learned byte pairs in learning order
    """
    special_tokens = validate_special_tokens(
        special_tokens
    )

    if vocab_size > 65_536:
        raise ValueError(
            "vocab_size must not exceed 65,536 "
            "when using uint16 token IDs"
        )

    with open(input_path,encoding="utf-8",newline="",) as stream:
        text = stream.read()

    # IDs 0–255 are the corresponding individual bytes.
    vocab: dict[int, bytes] = {
        byte_value: bytes([byte_value])
        for byte_value in range(256)
    }

    # Special tokens follow the base byte vocabulary.
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode(
            "utf-8"
        )

    number_of_merges = vocab_size - len(vocab)

    if number_of_merges < 0:
        raise ValueError(
            f"vocab_size={vocab_size} is too small; "
            f"{len(vocab)} initial tokens are required"
        )

    pretoken_counts = count_pretokens(
        text,
        special_tokens,
    )

    # count_pretokens already returns tuples of bytes.
    word_freqs = dict(pretoken_counts)
    merges: list[tuple[bytes, bytes]] = []

    for _ in range(number_of_merges):
        # This full recount is intentionally the slow tutorial
        # implementation. Section 3.2 optimises it.
        pair_counts = count_pairs(word_freqs)

        if not pair_counts:
            break

        # Highest frequency first; lexicographically greatest
        # pair when frequencies tie.
        best_pair = max(
            pair_counts.items(),
            key=lambda item: (item[1], item[0]),
        )[0]

        vocab[len(vocab)] = (
            best_pair[0] + best_pair[1]
        )
        merges.append(best_pair)

        new_word_freqs: Counter = Counter()

        for symbols, frequency in word_freqs.items():
            merged_symbols = merge_word(
                symbols,
                best_pair,
            )
            new_word_freqs[merged_symbols] += frequency

        word_freqs = new_word_freqs

    return vocab, merges

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    """Train BPE using incremental pair counting."""

    special_tokens = validate_special_tokens(
        special_tokens
    )

    if vocab_size > 65_536:
        raise ValueError(
            "vocab_size must not exceed 65,536 "
            "when using uint16 token IDs"
        )

    with open(
        input_path,
        encoding="utf-8",
        newline="",
    ) as stream:
        text = stream.read()

    # IDs 0-255 represent their corresponding bytes.
    vocab: dict[int, bytes] = {
        byte_value: bytes([byte_value])
        for byte_value in range(256)
    }

    # Special tokens come after the byte vocabulary.
    for special_token in special_tokens:
        vocab[len(vocab)] = (
            special_token.encode("utf-8")
        )

    number_of_merges = (
        vocab_size - len(vocab)
    )

    if number_of_merges < 0:
        raise ValueError(
            f"vocab_size={vocab_size} "
            f"is too small; {len(vocab)} "
            "initial tokens are required"
        )

    pretoken_counts = count_pretokens(
        text,
        special_tokens,
    )

    # Assign a stable integer ID to each distinct
    # pre-token. Its symbol tuple will change during
    # merging, but its ID remains unchanged.
    words = list(pretoken_counts)

    frequencies = [
        pretoken_counts[word]
        for word in words
    ]

    # Build the pair counts and inverted index once.
    (
        pair_counts,
        pair_to_word_ids,
        heap,
    ) = _build_pair_indexes(
        words,
        frequencies,
    )

    merges: list[Pair] = []

    for _ in range(number_of_merges):
        best_pair = _pop_best_pair(
            pair_counts,
            heap,
        )

        if best_pair is None:
            break

        vocab[len(vocab)] = (
            best_pair[0] + best_pair[1]
        )
        merges.append(best_pair)

        # This replaces the full pair recount and
        # the loop over every distinct pre-token.
        _merge_pair_incrementally(
            best_pair,
            words,
            frequencies,
            pair_counts,
            pair_to_word_ids,
            heap,
        )

        # Old heap entries are retained temporarily.
        # Rebuild occasionally to control memory use.
        if len(heap) > max(
            100_000,
            2 * len(pair_counts),
        ):
            heap = [
                _PairHeapEntry(count, pair)
                for pair, count
                in pair_counts.items()
            ]
            heapq.heapify(heap)

    return vocab, merges

def save_vocab(
    vocab: dict[int, bytes],
    path: str,
) -> None:
    """Save token IDs and byte values using hexadecimal."""
    with open(path, "w", encoding="utf-8") as stream:
        for token_id in sorted(vocab):
            stream.write(
                f"{token_id}\t{vocab[token_id].hex()}\n"
            )


def save_merges(
    merges: list[tuple[bytes, bytes]],
    path: str,
) -> None:
    """Save ordered byte-pair merges using hexadecimal."""
    with open(path, "w", encoding="utf-8") as stream:
        for left, right in merges:
            stream.write(
                f"{left.hex()}\t{right.hex()}\n"
            )


def load_vocab(path: str) -> dict[int, bytes]:
    """Load the hexadecimal vocabulary format."""
    vocab: dict[int, bytes] = {}

    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip("\n")

            if not line:
                continue

            fields = line.split("\t", 1)

            if len(fields) != 2:
                raise ValueError(
                    f"Invalid vocabulary line {line_number}: {line!r}"
                )

            token_id_text, token_hex = fields
            vocab[int(token_id_text)] = bytes.fromhex(token_hex)

    return vocab


def load_merges(
    path: str,
) -> list[tuple[bytes, bytes]]:
    """Load hexadecimal byte-pair merges in learned order."""
    merges: list[tuple[bytes, bytes]] = []

    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.rstrip("\n")

            if not line:
                continue

            fields = line.split("\t")

            if len(fields) != 2:
                raise ValueError(
                    f"Invalid merge line {line_number}: {line!r}"
                )

            left_hex, right_hex = fields
            merges.append(
                (
                    bytes.fromhex(left_hex),
                    bytes.fromhex(right_hex),
                )
            )

    return merges

class BPETokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = validate_special_tokens(
            special_tokens
        )

        # Validate the required base vocabulary.
        for byte_value in range(256):
            if self.vocab.get(byte_value) != bytes(
                [byte_value]
            ):
                raise ValueError(
                    "Vocabulary IDs 0–255 must contain "
                    "their corresponding byte values"
                )

        self.token_to_id = {
            token: token_id
            for token_id, token in self.vocab.items()
        }

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        self.special_to_id: dict[str, int] = {}

        for special_token in self.special_tokens:
            special_bytes = special_token.encode(
                "utf-8"
            )

            if special_bytes not in self.token_to_id:
                raise ValueError(
                    f"Special token {special_token!r} "
                    "is missing from the vocabulary"
                )

            self.special_to_id[special_token] = (
                self.token_to_id[special_bytes]
            )

        self._cache: dict[
            tuple[bytes, ...],
            list[int],
        ] = {}

    @classmethod
    def from_files(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        vocab = load_vocab(vocab_path)
        merges = load_merges(merges_path)

        return cls(
            vocab,
            merges,
            special_tokens=special_tokens,
        )

    def _apply_merges(
        self,
        symbols: tuple[bytes, ...],
    ) -> list[bytes]:
        """Apply the earliest-learned applicable merge."""
        symbols = list(symbols)

        while len(symbols) > 1:
            best_rank = None
            best_index = None

            for index, pair in enumerate(
                zip(symbols, symbols[1:])
            ):
                rank = self.merge_ranks.get(pair)

                if rank is not None and (
                    best_rank is None
                    or rank < best_rank
                ):
                    best_rank = rank
                    best_index = index

            if best_index is None:
                break

            symbols[
                best_index : best_index + 2
            ] = [
                symbols[best_index]
                + symbols[best_index + 1]
            ]

        return symbols

    def _encode_pretoken(
        self,
        pretoken: tuple[bytes, ...],
    ) -> list[int]:
        if pretoken not in self._cache:
            merged_symbols = self._apply_merges(
                pretoken
            )

            self._cache[pretoken] = [
                self.token_to_id[symbol]
                for symbol in merged_symbols
            ]

        return self._cache[pretoken]

    def encode(self, text: str) -> list[int]:
        """Encode text into byte-BPE token IDs."""
        token_ids: list[int] = []

        for chunk in split_on_special_tokens(
            text,
            self.special_tokens,
            keep=True,
        ):
            if chunk in self.special_to_id:
                token_ids.append(
                    self.special_to_id[chunk]
                )
            elif chunk:
                for pretoken in pretokenize(chunk):
                    token_ids.extend(
                        self._encode_pretoken(
                            pretoken
                        )
                    )

        return token_ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """Lazily encode an iterable of text chunks.

        For identical segmentation to encode(), chunks should
        end at document or pre-token boundaries.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """Decode IDs without crashing on incomplete UTF-8."""
        pieces = []

        for token_id in ids:
            integer_id = int(token_id)

            if integer_id not in self.vocab:
                raise ValueError(
                    f"Unknown token ID: {integer_id}"
                )

            pieces.append(
                self.vocab[integer_id]
            )

        raw_bytes = b"".join(pieces)

        return raw_bytes.decode(
            "utf-8",
            errors="replace",
        )

    