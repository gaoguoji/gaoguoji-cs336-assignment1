from cs336_basics.tokenizer import Tokenizer 
from tests.common import gpt2_bytes_to_unicode 
from cs336_basics.pretokenization_example import find_chunk_boundaries
import json 
from functools import partial 
import multiprocessing as mp

def encode_one(text, encoder):
    tokens = encoder(text)
    return tokens

if __name__ == "__main__": 
    merges_path = "tests/fixtures/train-bpe-tinystory-merges.txt"
    vocab_path = "tests/fixtures/train-bpe-tinystory-vocab.json"
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(merges_path, encoding="utf-8") as f:
            gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                    bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in gpt2_reference_merges
            ] 

    with open(vocab_path, encoding="utf-8") as f:
            gpt2_reference_vocab = json.load(f)
            vocab = {
                gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
                for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
            } 
    print(f"merges len is {len(merges)}")
    print(f"vocab len is {len(vocab)}") 
    print(f"longest token is {max(gpt2_reference_vocab, key=lambda item:len(item))}")

    tokenizer = Tokenizer(vocab, merges, ['<|endoftext|>'])   

    #input_path = "tests/fixtures/tinystories_sample_5M.txt" 
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    num_processes = 10
    corpus = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore") 
            corpus.append(chunk) 
   
    worker = partial(encode_one, encoder=tokenizer.encode)

    num_process = len(corpus)
    print(f"corpus size is {num_process}") 
    with mp.Pool(num_process) as pool:
        tokens_list = pool.map(worker, corpus)
    result = [] 
    for tokens in tokens_list:
        result.extend(tokens)
    # with open(input_path, 'r', encoding='utf-8') as f:
    #     corpus = f.read()
    # result = tokenizer.encode(corpus) 
    print(f"tokens size is {len(result)}") 
    # print(f"first 100 tokens is {result[:100]}") 
    # print(f"first 100 tokens word is {tokenizer.decode(result[:100])}")
