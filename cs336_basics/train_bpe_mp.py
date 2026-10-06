import os
from collections import defaultdict
from typing import Dict, Set

import regex
from tests.common import gpt2_bytes_to_unicode 
import cProfile
import pstats
from collections import Counter
import multiprocessing as mp 
from cs336_basics.pretokenization_example import find_chunk_boundaries
from functools import partial 
import json

def count_one(text, pat_re, special_tokens):
    local = Counter()
    paragraphs = regex.split('|'.join(map(regex.escape, special_tokens)), text) 
    for paragraph in paragraphs:
        for m in pat_re.finditer(paragraph): # finditer返回的是match对象的迭代器
            word = m.group(0)
            local[tuple(bytes([ch]) for ch in word.encode("utf-8"))] += 1
    return local

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
    # profiler = cProfile.Profile()
    # profiler.enable()

    vocab : dict[int , bytes] = {i : bytes([i]) for i in range(256)}
    cur_token_id = 256
    words_set : set[bytes] = set(vocab.values())
    for special_token in special_tokens:
        sp_token_bytes = special_token.encode('utf-8')
        if sp_token_bytes not in words_set:
            words_set.add(sp_token_bytes)
            vocab[cur_token_id] = sp_token_bytes
            cur_token_id += 1

    num_processes = 10 
    corpus = []
    with open(input_path, "rb") as f:  # 因为要在非开始处f.seek,因此必须用二进制模式打开
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore") 
            corpus.append(chunk) 

    print(f"corpus len is {len(corpus)}")        

    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" #实现单词拆分的正则 
    PAT_RE = regex.compile(pat)
    worker = partial(count_one, pat_re=PAT_RE, special_tokens=special_tokens)

    num_process = len(corpus)
    with mp.Pool(num_process) as pool:
        counters = pool.map(worker, corpus)
    word_counts = Counter()
    for item in counters:
        word_counts.update(item)

    pair_counts: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)    
    pair2word: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set) 
    id2word: dict[int, tuple[bytes]] = dict()
    
    for word_id, (word_bytes_tuple, counts) in enumerate(word_counts.items()):
        for i in range(len(word_bytes_tuple) - 1):
            pair1, pair2 = word_bytes_tuple[i], word_bytes_tuple[i + 1]
            pair_counts[(pair1, pair2)] += counts 
            pair2word[(pair1, pair2)].add(word_id) 
        id2word[word_id] = word_bytes_tuple 

    merges = []
    while cur_token_id < vocab_size and len(pair_counts) > 0: # 一定要加pair_counts为空的条件
        best_pair = max(pair_counts, key=lambda item: (pair_counts[item], item)) #首先以频率排序，频率相同的取字典序
        vocab[cur_token_id] = best_pair[0] + best_pair[1]
        merges.append(best_pair) 

        affected_word_ids = pair2word[best_pair]         
        affected_word_ids_list = [id for id in affected_word_ids] # 循环列表，因为要改变set所以不能循环set
        for id in affected_word_ids_list:
            word_bytes_tuple = id2word[id] 
            counts = word_counts[word_bytes_tuple]
            for i in range(len(word_bytes_tuple) - 1):
                pair1, pair2 = word_bytes_tuple[i], word_bytes_tuple[i + 1]
                if  (pair1, pair2) in pair_counts:
                    pair_counts[(pair1, pair2)] -= counts 
                    #if id in pair2word[(pair1, pair2)]:
                    pair2word[(pair1, pair2)].discard(id) # 注意这里用discard而不是remove,因为pair会有重复导致remove前已经删掉了
                    if pair_counts[(pair1, pair2)] == 0:
                        del pair_counts[(pair1, pair2)] 
                        del pair2word[(pair1, pair2)]
            del word_counts[word_bytes_tuple] 
            word_merged = merge_word_with_best_pair(word_bytes_tuple, best_pair)
            word_counts[word_merged] += counts 
            id2word[id] = word_merged  #id2word重置
            for i in range(len(word_merged) - 1):
                pair1, pair2 = word_merged[i], word_merged[i + 1]
                pair_counts[(pair1, pair2)] += counts 
                pair2word[(pair1, pair2)].add(id)
        cur_token_id += 1     

    # profiler.disable()
    # print("=== 整体耗时 ===")
    # pstats.Stats(profiler).sort_stats('cumulative').print_stats(10)
    return vocab, merges    
    
if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    #input_path = "tests/fixtures/corpus.en"
    vocab, merges = train_bpe(input_path, 10000, special_tokens) 
    # print(f"vocab is {vocab}")
    # print(f"merges is {merges}")
    vocab_w = {"".join([gpt2_bytes_to_unicode()[ch] for ch in v]): k for k, v in vocab.items()} 
    merges_w = [("".join([gpt2_bytes_to_unicode()[ch] for ch in pair1]), "".join([gpt2_bytes_to_unicode()[ch] for ch in pair2])) for (pair1, pair2) in merges]
    with open("tests/fixtures/train-bpe-tinystory-vocab.json", 'w', encoding='utf-8') as fp: 
        json.dump(vocab_w, fp, ensure_ascii=False, indent=4) 
    with open("tests/fixtures/train-bpe-tinystory-merges.txt", 'w', encoding='utf-8') as fp:
        for merge in merges_w:
            fp.write(" ".join(merge) + "\n") 

    
   
