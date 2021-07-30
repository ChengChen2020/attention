# Attention Is All You Need

Implementation reorganized from [d2l](https://d2l.ai/_modules/d2l/torch.html)

### Architecture

<details>
  <summary>[Click to expand]</summary>

- Transformer
	- TransformerEncoder
		- Embedding
		- PositionalEncoding
		- EncoderBlock (N$\times$)
			- MultiHeadAttention (Self)
			- Add & Norm
			- PositionWiseFFN
			- Add & Norm
	- TransformerDecoder
		- Embedding
		- PositionalEncoding
		- DecoderBlock (N$\times$)
			- MultiHeadAttention (Self)
			- Add & Norm
			- MultiHeadAttention (Encoder-Decoder)
			- Add & Norm
			- PositionWiseFFN
			- Add & Norm
		- Linear

</details>

### Demo
`./experiments/scripts/demo.sh`