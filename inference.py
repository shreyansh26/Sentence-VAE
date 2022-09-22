import os
import json
import torch
import argparse

from model import SentenceVAEModel
from utils import idx2word, interpolate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    with open(args.data_dir+'/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    special_tokens = {
        'sos_token': w2i['<sos>'],
        'eos_token': w2i['<eos>'],
        'pad_token': w2i['<pad>'],
        'unk_token': w2i['<unk>']
    }

    model = SentenceVAEModel(
            vocab_size=len(w2i),
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            word_dropout_rate=args.word_dropout_rate,
            embedding_dropout_rate=args.embedding_dropout_rate,
            latent_dim=args.latent_dim,
            special_tokens=special_tokens,
            max_seq_length=args.max_seq_length,
            model_type=args.model_type,
            bidirectional=args.bidirectional,
            num_layers=args.num_layers
        )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(args.load_checkpoint, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(args.load_checkpoint))

    print("Model loaded from %s" % args.load_checkpoint)

    model.to(device)
    model.eval()

    samples, z = model.inference(n=args.num_samples)
    print('----------SAMPLES----------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    z1 = torch.randn([args.latent_dim]).numpy()
    z2 = torch.randn([args.latent_dim]).numpy()
    z = torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float().to(device)
    samples, _ = model.inference(z=z)
    print('-------INTERPOLATION-------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-ms', '--max_seq_length', type=int, default=60)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--model_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout_rate', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout_rate', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_dim', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', '--bidirectional', action='store_true')

    args = parser.parse_args()

    args.model_type = args.model_type.lower()

    assert args.model_type in ['rnn', 'gru']
    assert 0 <= args.word_dropout_rate <= 1
    assert 0 <= args.embedding_dropout_rate <= 1

    main(args)