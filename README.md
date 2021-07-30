# "Attention Is All You Need"

Implementation reorganized from [d2l](https://d2l.ai/_modules/d2l/torch.html)

### Architecture

<details>
  <summary>[Click to expand]</summary>

- Transformer (**transformer.py**)
	- TransformerEncoder (**transformer.py**)
		- Embedding
		- PositionalEncoding (**embeddings.py**)
		- EncoderBlock (Nx) (**layers.py**)
			- MultiHeadAttention (Self) (**sublayers.py**)
			- Add & Norm (**sublayers.py**)
			- PositionWiseFFN (**sublayers.py**)
			- Add & Norm
	- TransformerDecoder (**transformer.py**)
		- Embedding
		- PositionalEncoding
		- DecoderBlock (Nx) (**layers.py**)
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