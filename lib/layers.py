import torch
import torch.nn as nn

from lib.sublayers import MultiHeadAttention, PositionWiseFFN, AddNorm

class EncoderBlock(nn.Module):
	def __init__(self, key_size, query_size, value_size, num_hiddens,
				 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
				 dropout, use_bias=False, **kwargs):
		super(EncoderBlock, self).__init__(**kwargs)
		self.attention = MultiHeadAttention(
			key_size, query_size, value_size, num_hiddens, num_heads, dropout,
			use_bias)
		self.addnorm1 = AddNorm(norm_shape, dropout)
		self.ffn = PositionWiseFFN(
			ffn_num_input, ffn_num_hiddens, num_hiddens)
		self.addnorm2 = AddNorm(norm_shape, dropout)

	def forward(self, X, valid_lens):
		Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
		return self.addnorm2(Y, self.ffn(Y))

class DecoderBlock(nn.Module):
	# The `i`-th block in the decoder
	def __init__(self, key_size, query_size, value_size, num_hiddens,
				 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
				 dropout, i, **kwargs):
		super(DecoderBlock, self).__init__(**kwargs)
		self.i = i
		self.attention1 = MultiHeadAttention(
			key_size, query_size, value_size, num_hiddens, num_heads, dropout)
		self.addnorm1 = AddNorm(norm_shape, dropout)
		self.attention2 = MultiHeadAttention(
			key_size, query_size, value_size, num_hiddens, num_heads, dropout)
		self.addnorm2 = AddNorm(norm_shape, dropout)
		self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
								   num_hiddens)
		self.addnorm3 = AddNorm(norm_shape, dropout)

	def forward(self, X, state):
		enc_outputs, enc_valid_lens = state[0], state[1]
		# During training, all the tokens of any output sequence are processed
		# at the same time, so `state[2][self.i]` is `None` as initialized.
		# When decoding any output sequence token by token during prediction,
		# `state[2][self.i]` contains representations of the decoded output at
		# the `i`-th block up to the current time step
		if state[2][self.i] is None:
			key_values = X
		else:
			key_values = torch.cat((state[2][self.i], X), axis=1)
		state[2][self.i] = key_values
		if self.training:
			batch_size, num_steps, _ = X.shape
			# Shape of `dec_valid_lens`: (`batch_size`, `num_steps`), where
			# every row is [1, 2, ..., `num_steps`]
			dec_valid_lens = torch.arange(
				1, num_steps + 1, device=X.device).repeat(batch_size, 1)
		else:
			dec_valid_lens = None

		# Self-attention
		X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
		Y = self.addnorm1(X, X2)
		# Encoder-decoder attention. Shape of `enc_outputs`:
		# (`batch_size`, `num_steps`, `num_hiddens`)
		Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
		Z = self.addnorm2(Y, Y2)
		return self.addnorm3(Z, self.ffn(Z)), state

if __name__ == '__main__':
	X = torch.ones((2, 100, 24))
	valid_lens = torch.tensor([3, 2])
	
	encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
	encoder_blk.eval()
	print(encoder_blk(X, valid_lens).shape)

	decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
	decoder_blk.eval()
	state = [encoder_blk(X, valid_lens), valid_lens, [None]]
	print(decoder_blk(X, state)[0].shape)

