'''
Tokenizer

- Performs token segmentation and maps each token to a uniquo token ID
- Requires because the embedding layer accepts only integer indices as input

'''


import tiktoken  # BPE algorithm by GPT models




# Build a tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Encode the text
text = "just for test"
integers = tokenizer.encode(text)
# print(integers) # DEBUG: temporary output to verify the tokenizer encoding

# Decode the integers
strings = tokenizer.decode(integers)
# print(strings) # DEBUG: tempoary output to verify the tokenizer decoding