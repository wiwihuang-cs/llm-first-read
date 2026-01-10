"""
# === concept ===
1. embedding : convert data to vector form
- word2Vec : method of word embedding
- sentence or paragraph embedding is popular with RAG 
"""

import tiktoken  # BPE algorithm

# Prepare the text
"""
text = "I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that, in the height of his glory, he had dropped his painting, married a rich widow, and established himself in a villa on the Riviera. (Though I rather thought it would have been Rome or Florence.)" 
"""

text_noInWordTable = "Akwirw ier"

"""
with open("the-verdict.txt", "r", encoding= "utf-8") as f:
    text = f.read() 
"""

# Build a tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Encode the text
integers = tokenizer.encode(text_noInWordTable)
print(integers)

# Decode the integers
strings = tokenizer.decode(integers)
print(strings)