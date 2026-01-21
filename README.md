# llm-first-read
My first systematic reading notes and code experiments while learning Large Language Models (LLMs).

## === ML vs DL ===
- ml needs to choose the characters manually but DL not. 

## History of LLM (in the field of NLP)

- Self-attention (2015–2016):  
  Researchers explored whether each token could directly attend to other tokens within the same sentence.  
  This led to self-attention, a mechanism that computes context-vector representations by weighting interactions between tokens, allowing each token’s representation to depend on the entire sequence rather than a single hidden state.
  But, during this period, self-attention was typically used in combination with RNNs or LSTMs, rather than replacing them.

- Transformer (2017):  
  In the paper *Attention Is All You Need*, self-attention is used as the core computation to replace recurrence in RNNs and LSTMs.  
  The architecture relies entirely on self-attention and feed-forward networks, without any recurrent or convolutional structures.

- LLM (2018–):  
  Before the paper *Improving Language Understanding by Generative Pre-Training* was published, self-attention and the Transformer architecture were primarily used in encoder–decoder settings, especially for machine translation tasks.  
  GPT-style models adopt a decoder-only Transformer with causal self-attention, making them well suited for autoregressive language modeling and text generation.  
  GPT-1 also explicitly introduced the two-stage training paradigm of large-scale pretraining followed by task-specific fine-tuning.

- Generative AI (GenAI):  
  Along with the development of large language models, the term Generative AI (GenAI) became widely used to describe AI systems capable of generating new content, such as text, images, and code.  
  LLMs represent the text-generation branch of GenAI and played a key role in popularizing the concept.
