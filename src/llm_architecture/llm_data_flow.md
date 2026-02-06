# model purpose
- predict the next token based on the previous tokens
- P(next_token | previous_tokens) : based on attention mechanism

# data flow
- Input text
- Tokenization : tokenize text into tokens(strings), and then convert to token_ids (integers)
- Embedding Layer: convert token_ids to dense vectors, which is like a loopup table that maps each token_id to a vector representation
- Transformer Blocks: the output is still a sequence of vectors representing token_ids
    - Multi-Head Self-Attention: capture relationships between token_ids, and **adjust token_ids representations based on the context of the previous tokens.**
    - Feed-Forward Neural Network: process attention outputs
- Output Linear Layer: produces logits, which represent the unnormalized **probabilities between the token_id and all possible next token_ids**