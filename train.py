import os
import sys
import json
import torch
import argparse
import numpy as np
from pytz import timezone 
from datetime import datetime
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from utils import kl_anneal_function, idx2word, experiment_name

from ptb import PTB
from model import SentenceVAEModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    ts = datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S')

    splits = ['train', 'valid'] + (['test'] if args.test else [])

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_seq_length,
            min_occ=args.min_occ
        )

    special_tokens = {
        'sos_token': datasets['train'].sos_idx,
        'eos_token': datasets['train'].eos_idx,
        'pad_token': datasets['train'].pad_idx,
        'unk_token': datasets['train'].unk_idx
    }

    params = dict(
        vocab_size=datasets['train'].vocab_size,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        word_dropout_rate=args.word_dropout_rate,
        embedding_dropout_rate=args.embedding_dropout_rate,
        latent_dim=args.latent_dim,
        special_tokens=special_tokens,
        max_seq_length=args.max_seq_length,
        model_type=args.model_type,
        bidirectional=args.bidirectional,
        num_layers=args.num_layers,
    )

    model = SentenceVAEModel(**params)
    model.to(device)
    print(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, experiment_name(args, ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_nll = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, reduction='sum')

    def loss_fn(logp, target, length, mean, log_sigmasquared, anneal_function, step, k, annealing_till):
        # till the max sequence length followed by flattening
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        nll_loss = loss_nll(logp, target)

        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + log_sigmasquared - mean**2 - torch.exp(log_sigmasquared))
        kl_weight = kl_anneal_function(anneal_function, step, k, annealing_till)

        return nll_loss, kl_loss, kl_weight

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0

    # Epoch loop
    for epoch in range(args.epochs):
        # For train, valid and test splits
        for split in splits:
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle = True if split=='train' else False,
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            # Iterate over each batch
            for iteration, batch in enumerate(data_loader):
                batch_size = batch['input'].shape[0]
                for k, v in batch.items():
                    if k != 'length' and torch.is_tensor(v):
                        batch[k] = v.to(device)

                # Forward pass
                logp, mean, log_sigmasquared, z = model(batch['input'], batch['length'])

                # Loss calculation
                nll_loss, kl_loss, kl_weight = loss_fn(logp, batch['target'], batch['length'], mean, 
                                                log_sigmasquared, args.anneal_function, step, args.k, args.annealing_till)

                loss = (nll_loss + kl_weight * kl_loss) / batch_size # Since 'sum' reduction of NLL and KL loss also gets added over whole batch

                # Backward pass + Optimizer step
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.view(1, -1)), dim=0)

                ## Logging begin
                if args.tensorboard_logging:
                    writer.add_scalar("%s/ELBO" % split.upper(), loss.item(), epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/NLL Loss" % split.upper(), nll_loss.item() / batch_size,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Loss" % split.upper(), kl_loss.item() / batch_size,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Weight" % split.upper(), kl_weight,
                                      epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                          % (split.upper(), iteration, len(data_loader)-1, loss.item(), nll_loss.item()/batch_size,
                          kl_loss.item()/batch_size, kl_weight))

                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    tracker['target_sents'] += idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(),
                                                        pad_idx=datasets['train'].pad_idx)
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            # Iterating over split complete
            print("%s Epoch %02d/%i, Mean ELBO %9.4f" % (split.upper(), epoch, args.epochs, tracker['ELBO'].mean()))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

            # Save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/'+ts)
                with open(os.path.join('dumps/'+ts+'/valid_E%i.json' % epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # Save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "epoch_%i.pt" % epoch)
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_seq_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--model_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_dim', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout_rate', type=float, default=0)   # Helps quite a lot
    parser.add_argument('-ed', '--embedding_dropout_rate', type=float, default=0.5) # Not that helpful

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-at', '--annealing_till', type=int, default=2500)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')

    args = parser.parse_args()

    args.model_type = args.model_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.model_type in ['rnn', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout_rate <= 1
    assert 0 <= args.embedding_dropout_rate <= 1

    main(args)