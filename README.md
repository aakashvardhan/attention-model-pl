# English-Italian Translator using the Transformer model

Note: The Attention model is taken from 'Attention is all you need' (https://arxiv.org/pdf/1706.03762.pdf)


## [Transformer Architecture](https://github.com/aakashvardhan/attention-model-pl/blob/main/models/model.py)
![imgs](https://github.com/aakashvardhan/attention-model-pl/blob/main/attention_research_1.webp)
### 1. Layer Normalization (`LayerNormalization`)

- **Purpose**: Normalizes the input across the features.
- **Computation**: 
  \[
  \text{output} = \alpha \times \frac{x - \text{mean}}{\text{std} + \epsilon} + \text{bias}
  \]
- **Parameters**: `alpha`, `bias` are learnable. `eps` is a small constant.

---

### 2. Feed Forward Block (`FeedForwardBlock`)

- **Purpose**: A simple feed-forward neural network.
- **Computation**: 
  \[
  \text{output} = W_2 \times \text{ReLU}(W_1 \times x + b_1) + b_2
  \]
- **Parameters**: \(W_1\), \(b_1\), \(W_2\), \(b_2\)

---

### 3. Input Embeddings (`InputEmbeddings`)

- **Purpose**: Embeds tokens into a high-dimensional space.
- **Parameters**: Embedding matrix

---

### 4. Positional Encoding (`PositionalEncoding`)

- **Purpose**: Adds position information to the input embeddings.
- **Computation**: Uses sine and cosine functions of different frequencies.

---

### 5. Residual Connection (`ResidualConnection`)

- **Purpose**: Helps in training deep networks via shortcut connections.
- **Computation**: 
  \[
  \text{output} = x + \text{sublayer}(x)
  \]
---

### 6. Multi-Head Attention (`MultiHeadAttentionBlock`)

- **Purpose**: Attends to different parts of the input simultaneously.
- **Computation**: Consists of Query, Key, and Value transformations, followed by scaled dot-product attention.

---

### 7. Encoder & Decoder Blocks (`EncoderBlock`, `DecoderBlock`)

- **Purpose**: Fundamental building blocks of the Transformer architecture.
- **Components**: Each block contains self-attention layers, feed-forward neural networks, and layer normalization.

---

### 8. Encoder & Decoder (`Encoder`, `Decoder`)

- **Purpose**: Stack of Encoder and Decoder Blocks.

---

### 9. Projection Layer (`ProjectionLayer`)

- **Purpose**: Maps the decoder output to the target vocabulary.
- **Computation**: 
  \[
  \text{output} = \log(\text{Softmax}(W \times x + b))
  \]

---

## Computation Graph

1. **Encoding Phase**: Input -> InputEmbeddings -> PositionalEncoding -> Encoder
2. **Decoding Phase**: Output of Encoder, Target Input -> InputEmbeddings -> PositionalEncoding -> Decoder
3. **Projection**: Output of Decoder -> ProjectionLayer

## Building the Model (`build_transformer`)

The function `build_transformer` is used to assemble these components into a complete Transformer model. It initializes the model's weights with Xavier Uniform
