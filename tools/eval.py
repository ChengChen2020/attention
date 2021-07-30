import _init_paths

import os
import argparse

import torch

from datasets.dataset import load_data_nmt
from lib.utils import truncate_pad, bleu, try_gpu
from lib.transformer import TransformerEncoder, TransformerDecoder, Transformer

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence."""
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(
        torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device),
        dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_steps', type=int, default=10, help='num steps')
parser.add_argument('--num_layers', type=int, default=2, help='repeating layers')
parser.add_argument('--num_heads', type=int, default=4, help='num heads')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--feature_dim', type=int, default=32, help='feature dimension')
parser.add_argument('--ffn_hidden_dim', type=int, default=64, help='ffn hidden feature dimension')
parser.add_argument('--log_dir', type=str, default='./experiments/logs')
parser.add_argument('--resume', type=str, default='model_0.031.pth')
opt = parser.parse_args()

if __name__ == '__main__':
    num_hiddens, num_layers, dropout, batch_size, num_steps = \
        opt.feature_dim, opt.num_layers, opt.dropout, opt.batch_size, opt.num_steps
    ffn_num_input, ffn_num_hiddens, num_heads = \
        opt.feature_dim, opt.ffn_hidden_dim, opt.num_heads
    key_size, query_size, value_size = \
        opt.feature_dim, opt.feature_dim, opt.feature_dim,
    norm_shape = [opt.feature_dim]
    device = try_gpu()

    _, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    
    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = Transformer(encoder, decoder)
    net.load_state_dict(torch.load('{0}/{1}'.format(opt.log_dir, opt.resume)))

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ',
              f'bleu {bleu(translation, fra, k=2):.3f}')


