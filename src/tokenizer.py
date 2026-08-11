
# imports

from __future__ import annotations
from collections import Counter
from typing import Iterable, Iterator

import regex

END_OF_TEXT = "<|endoftext|>"

'''WORD_START = "▁"      # ▁ marks the start of a word (written '_' in the lecture slides)
UNK = "<|unk|>"

def count_pretokens(text, special_tokens=(END_OF_TEXT,)):
    """
    Count how often each pre-token occurs in the corpus.
    params:
        text:           the raw training corpus
        special_tokens: strings that act as hard boundaries (never merged across)
    returns:
        Counter mapping pre-token string -> count
    """
    counts = Counter()
    split_pattern = "|".join(re.escape(s) for s in special_tokens)
    for document in re.split(split_pattern, text):
        counts.update(pretokenize(document))
    return counts


_toy_corpus = ("low low low low low\n"
               "lower lower widest widest widest\n"
               "newest newest newest newest newest newest")
print(count_pretokens(_toy_corpus))'''

# gpt2 pre-tokenizer regex from appendix A
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
(?!\S)|\s+"""

GPT2_pretoken_regex = regex.compile(GPT2_PAT)

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

def count_pairs(word_freqs):
    """
    Count how often each adjacent pair of symbols occurs, weighted by pre-token frequency.
    params:
        word_freqs: dict mapping a pre-token (tuple of symbol strings) -> its corpus count
    returns:
        Counter mapping (left_symbol, right_symbol) -> total count
    """
    pair_counts = Counter()
    for symbols, freq in word_freqs.items():
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += freq
    return pair_counts

def merge_word(symbols, pair):
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
    i = 0
    while i < len(symbols):
        # If the pair starts here, append the concatenated symbol and skip forward by 2.
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        # Otherwise keep the current symbol and advance by 1.
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)

def train_bpe(text, vocab_size, special_tokens=(END_OF_TEXT, UNK), verbose=False):
    """
    Train a character-level BPE tokenizer (unoptimised reference implementation).
    params:
        text:           raw training corpus
        vocab_size:     desired final vocabulary size
        special_tokens: tokens added to the vocabulary before anything else
        verbose:        print progress every 200 merges
    returns:
        vocab:  dict mapping token ID -> token string
        merges: list of (left, right) pairs, in the order they were learned
    """
    # Represent each distinct pre-token as a tuple of single characters, with its corpus count.
    pretoken_counts = count_pretokens(text)
    word_freqs = {tuple(pretoken): count for pretoken, count in pretoken_counts.items()}

    # Initial vocabulary: the special tokens, then every character seen in the corpus.
    characters = sorted({ch for word in word_freqs for ch in word})
    vocab = list(special_tokens) + characters
    merges = []

    num_merges = vocab_size - len(vocab)
    if num_merges < 0:
        raise ValueError(f"vocab_size={vocab_size} is smaller than the {len(vocab)} "
                         f"special tokens + characters")
    if verbose:
        print(f"{len(word_freqs):,} distinct pre-tokens, {len(characters)} characters, "
              f"{num_merges} merges to learn")

    start = time.time()
    for step in range(num_merges):
        # Step 1: count every adjacent pair of symbols in the corpus.
        pair_counts = count_pairs(word_freqs)
        if not pair_counts:
            break

        # Step 2: pick the most frequent pair, breaking ties lexicographically-greatest.
        best_count = max(pair_counts.values())
        best_pair = max(pair for pair, count in pair_counts.items() if count == best_count)

        # Step 3: apply the merge everywhere in the corpus.
        word_freqs = {merge_word(word, best_pair): freq for word, freq in word_freqs.items()}

        # Step 4: record the merge and add the new symbol to the vocabulary.
        merges.append(best_pair)
        vocab.append(best_pair[0] + best_pair[1])

        if verbose and (step + 1) % 200 == 0:
            print(f"  merge {step + 1:5d}: {best_pair[0]!r} + {best_pair[1]!r} -> "
                  f"{vocab[-1]!r} (count {best_count:,})   [{time.time() - start:.1f}s]")

    return {i: token for i, token in enumerate(vocab)}, merges

class CharBPETokenizer:
    """
    Character-level BPE tokenizer.
    params:
        vocab:          dict mapping token ID -> token string
        merges:         list of (left, right) pairs in the order they were learned
        special_tokens: strings that encode to a single ID and are never split
    """

    def __init__(self, vocab, merges, special_tokens=(END_OF_TEXT, UNK)):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = list(special_tokens)
        self.token_to_id = {token: i for i, token in vocab.items()}
        # merge_ranks[pair] = the step at which this merge was learned (lower = applied earlier)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._cache = {}

    def _apply_merges(self, symbols):
        """
        Greedily apply learned merges to one pre-token, earliest-learned merge first.
        params:
            symbols: list of single-character strings
        returns:
            list of token strings
        """
        symbols = list(symbols)
        while len(symbols) > 1:
            # Find the applicable merge with the lowest rank (i.e. learned earliest).
            best_rank, best_index = None, None
            for i, pair in enumerate(zip(symbols, symbols[1:])):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, i
            # No merge in our list applies to this pre-token, so we are done.
            if best_index is None:
                break
            # Apply it: replace the two symbols at best_index with their concatenation.
            symbols[best_index:best_index + 2] = [symbols[best_index] + symbols[best_index + 1]]
        return symbols

    def _encode_pretoken(self, pretoken):
        if pretoken not in self._cache:
            # Replace any character we have never seen with the unknown token.
            safe = [c if c in self.token_to_id else UNK for c in pretoken]
            tokens = self._apply_merges(safe)
            self._cache[pretoken] = [self.token_to_id[t] for t in tokens]
        return self._cache[pretoken]

    def encode(self, text):
        """Encode a string into a list of integer token IDs."""
        ids = []
        # Keep the special tokens as delimiters so we can emit their IDs directly.
        split_pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        for chunk in re.split(split_pattern, text):
            if chunk in self.special_tokens:
                ids.append(self.token_to_id[chunk])
            else:
                for pretoken in pretokenize(chunk):
                    ids.extend(self._encode_pretoken(pretoken))
        return ids

    def decode(self, ids):
        """Decode a list of integer token IDs back into a string."""
        text = "".join(self.vocab.get(i, UNK) for i in ids)
        return text.replace(WORD_START, " ")

    