import os
from collections import defaultdict
from typing import Dict, Set

import regex

def word_has_pair(
        word_bytes_tuple : tuple[bytes],
        pair : tuple[bytes, bytes]
) -> bool:
    for i in range(len(word_bytes_tuple) - 1) :
        if (pair[0], pair[1]) == (word_bytes_tuple[i], word_bytes_tuple[i + 1]) :
            return True 
    return False

def merge_word_with_best_pair(
        word_bytes_tuple : tuple[bytes],
        best_pair : tuple[bytes, bytes]
) -> tuple[bytes]:
    res = []
    i = 0
    while i < len(word_bytes_tuple):
        if i < len(word_bytes_tuple) - 1 and (best_pair == (word_bytes_tuple[i], word_bytes_tuple[i + 1])):
            res.append(best_pair[0] + best_pair[1])
            i += 2
        else:
            res.append(word_bytes_tuple[i])
            i += 1
    return tuple(res)             


def train_bpe(
        input_path : str | os.PathLike, 
        vocab_size : int,
        special_tokens : list[str], 
        **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
        output its vocabulary and merges.
    
        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.
    
        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
                vocab:
                    The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes)
                merges:
                    BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Merges are ordered by order of creation.
        """
    vocab : Dict[int , bytes] = {i : bytes([i]) for i in range(256)}
    cur_token_id = 256
    words_set : Set[bytes] = set(vocab.values())
    for special_token in special_tokens:
        sp_token_bytes = special_token.encode('utf-8')
        if sp_token_bytes not in words_set:
            words_set.add(sp_token_bytes)
            vocab[cur_token_id] = sp_token_bytes
            cur_token_id += 1
    print("当前工作目录:", os.getcwd())
    print("文件是否存在:", os.path.exists(input_path))
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        corpus = f.read()

    word_counts = defaultdict(int)
    pair_counts = defaultdict(int)

    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" #实现单词拆分的正则
    paragraphs = regex.split('|'.join(map(regex.escape, special_tokens)), corpus) #按照特殊字符分开，注意规避正则
    for paragraph in paragraphs:
        words =  regex.findall(pat, paragraph)
        for word in words:
            word_bytes_list = [bytes([ch]) for ch in word.encode("utf-8")] #注意这里的写法
            word_counts[tuple(word_bytes_list)] += 1

    for word_bytes_tuple, counts in word_counts.items():
        for i in range(len(word_bytes_tuple) - 1):
            pair1, pair2 = word_bytes_tuple[i], word_bytes_tuple[i + 1]
            pair_counts[(pair1, pair2)] += counts 

    merges = []
    while cur_token_id < vocab_size and len(pair_counts) > 0:
        max_pair_count = max(pair_counts.values())
        best_pair_candidates = [k for k, v in pair_counts.items() if v == max_pair_count]
        best_pair = max(best_pair_candidates)
        vocab[cur_token_id] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        word_with_best_pair = {}
        for word_bytes_tuple, counts in word_counts.items():
            if word_has_pair(word_bytes_tuple, best_pair):
                word_with_best_pair[word_bytes_tuple] = counts 

        for word_bytes_tuple, counts in word_with_best_pair.items():
            word_merged = merge_word_with_best_pair(word_bytes_tuple, best_pair)
            for i in range(len(word_bytes_tuple) - 1):
                pair1, pair2 = word_bytes_tuple[i], word_bytes_tuple[i + 1]
                if  (pair1, pair2) in pair_counts:
                    pair_counts[(pair1, pair2)] -= counts 
                    if pair_counts[(pair1, pair2)] == 0:
                        del pair_counts[(pair1, pair2)] 

            del word_counts[word_bytes_tuple]
            word_counts[tuple(word_merged)] += counts 
            for i in range(len(word_merged) - 1):
                pair1, pair2 = word_merged[i], word_merged[i + 1]
                pair_counts[(pair1, pair2)] += counts 
        cur_token_id += 1    

    return vocab, merges    
    
if __name__ == "__main__":
   special_tokens = ["<|endoftext|>"]
   input_path = "tests/fixtures/tinystories_sample.txt"
   vocab, merges = train_bpe(input_path, 500, special_tokens) 
   print(f"vocab is {vocab}")
   print(f"merges is {merges}")
