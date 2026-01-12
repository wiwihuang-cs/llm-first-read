"""
# === concept ===
1. embedding : convert data to vector form
- word2Vec : method of word embedding
- sentence or paragraph embedding is popular with RAG 
"""

import tiktoken  # BPE algorithm


text_noInWordTable = "just for test"

# Build a tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Encode the text
integers = tokenizer.encode(text_noInWordTable)
print(integers)

# Decode the integers
strings = tokenizer.decode(integers)
print(strings)