import regex
from collections import defaultdict
from typing import Iterable, Iterator, List, Set, Tuple
import json 
import regex 
from tests.common import gpt2_bytes_to_unicode

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes] , 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None
                 ):
        self.vocab = vocab
        self.merges = merges 
        self.special_tokens = special_tokens 
        self.vocab_bytes_id = {v: k for k, v in vocab.items()}
        self.merges_priority = {k: i for i, k in enumerate(merges)} 
    
    def encode_text(self, word_bytes: bytes) -> list[int]:
        word_seg = [bytes([ch]) for ch in word_bytes]
        while len(word_seg) > 1:
            #print(f"word seg is {word_seg}")
            merge_candidates = set()
            for i in range(len(word_seg) - 1):
                if (word_seg[i], word_seg[i + 1]) in self.merges:
                    merge_candidates.add((word_seg[i], word_seg[i + 1]))
            if not merge_candidates:
                break 
            best_merge = min(merge_candidates, key = lambda x : self.merges_priority[x])
            new_seg = []
            #print(f"best merge is {best_merge}")
            i = 0
            while i < len(word_seg):
                if i < len(word_seg) - 1 and ((word_seg[i], word_seg[i + 1]) == best_merge):
                    new_seg.append(word_seg[i] + word_seg[i + 1])
                    i += 2
                else :
                    new_seg.append(word_seg[i])
                    i += 1
            word_seg = new_seg 
        #print(f"new seg is {word_seg}")
        return [self.vocab_bytes_id[seg] for seg in word_seg] 
    
    def encode(self, text: str) -> list[int]:
        result = [] 
        if not self.special_tokens:
            paragraphs = [text] 
        else :
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            paragraphs = regex.split(f'({'|'.join(map(regex.escape, sorted_special_tokens))})', text)
        for paragraph in paragraphs:
            if self.special_tokens and paragraph in self.special_tokens:
                result.append(self.vocab_bytes_id[paragraph.encode('utf-8')])
            else :
                words = regex.findall(PAT, paragraph)
                word_bytes_list = [word.encode('utf-8') for word in words]
                for word_bytes in word_bytes_list:
                    result.extend(self.encode_text(word_bytes)) 
        return result
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        result_bytes = b''.join([self.vocab[id] for id in ids])
        return result_bytes.decode('utf-8', errors = 'replace')

        

