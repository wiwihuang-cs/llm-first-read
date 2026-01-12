import tiktoken


text = "just for test"

'''
Build a tokenizer and Encode the text to integer
'''

# Build a tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Encode the text
enc_text = tokenizer.encode(text)

# print(enc_text)  # DEBUG: temporary output to verify the tokenization

'''
Build the input-target pairs
'''

context_size = 4
input = enc_text[:context_size]
target = enc_text[1:context_size+1]

# print(f"input: {input}")  # DEBUG: temporary output to verify the tokenization
# print(f"target: {target}")

# for i in range(1, context_size+1):  # DEBUG: temporary output to verify the tokenization
#     context = enc_text[:i]
#     desired = enc_text[i]
#     print(f"{context} ---> {desired}")