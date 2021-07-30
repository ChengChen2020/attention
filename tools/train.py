import _init_paths

import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from datasets.dataset import load_data_nmt
from lib.utils import Accumulator, Timer, MaskedSoftmaxCELoss, try_gpu, grad_clipping
from lib.transformer import TransformerEncoder, TransformerDecoder, Transformer

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_steps', type=int, default=10, help='num steps')
parser.add_argument('--num_layers', type=int, default=2, help='repeating layers')
parser.add_argument('--num_heads', type=int, default=4, help='num heads')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--feature_dim', type=int, default=32, help='feature dimension')
parser.add_argument('--ffn_hidden_dim', type=int, default=64, help='ffn hidden feature dimension')
parser.add_argument('--num_epochs', type=int, default=200, help='max number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--log_dir', type=str, default='./experiments/logs')
parser.add_argument('--resume', type=str, default='')
opt = parser.parse_args()

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    for epoch in tqdm(range(num_epochs)):
        timer = Timer()
        metric = Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            
            # Insert <bos> symbol
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing

            # params: enc, dec, enc_valid_len
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)

            break

    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

    torch.save(net.state_dict(), '{0}/model_{1}.pth'.format(opt.log_dir, f'{metric[0] / metric[1]:.3f}'))

if __name__ == '__main__':
	os.makedirs(opt.log_dir, exist_ok=True)

	num_hiddens, num_layers, dropout, batch_size, num_steps = \
		opt.feature_dim, opt.num_layers, opt.dropout, opt.batch_size, opt.num_steps
	lr, num_epochs, device = \
		opt.lr, opt.num_epochs, try_gpu()
	ffn_num_input, ffn_num_hiddens, num_heads = \
		opt.feature_dim, opt.ffn_hidden_dim, opt.num_heads
	key_size, query_size, value_size = \
		opt.feature_dim, opt.feature_dim, opt.feature_dim,
	norm_shape = [opt.feature_dim]

	train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)

	encoder = TransformerEncoder(
		len(src_vocab), key_size, query_size, value_size, num_hiddens,
		norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
		num_layers, dropout)
	decoder = TransformerDecoder(
		len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
		norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
		num_layers, dropout)
	net = Transformer(encoder, decoder)

	train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
