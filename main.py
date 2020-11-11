import argparse
import time
import math
import os
import torch
import torch.nn as nn

import pickle

import data
import models


def options():
    parser = argparse.ArgumentParser(description='VN Network: Embedding Newly Emerging Entities with Virtual Neighbors')
    parser.add_argument('--data', type=str, default='./dataset',
                        help='location of the data corpus')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--dict', type=str, default='./models/dict.pkl',
                        help='path to (save/load) the dictionary')
    parser.add_argument('--save', type=str, default='./models/model',
                        help='prefix to save the final model')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--optim_type', type=str, default='SGD',
                        help='type of the optimizer')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')


    parser.add_argument('--glove', type=str, default='',
                        help='path to the glove embedding')
    parser.add_argument('--direction', type=str, default='left2right',
                        help='type of language model direction (left2right, right2left, both)')
    parser.add_argument('--wo_tok', action='store_true',
                        help='without token embeddings')
    parser.add_argument('--wo_char', action='store_true',
                        help='without character embeddings')
    parser.add_argument('--tok_emb', type=int, default=200,
                        help='The dimension size of word embeddings')
    parser.add_argument('--char_emb', type=int, default=50,
                        help='The dimension size of character embeddings')
    parser.add_argument('--char_kmin', type=int, default=1,
                        help='minimum size of the kernel in the character encoder')
    parser.add_argument('--char_kmax', type=int, default=5,
                        help='maximum size of the kernel in the character encoder')
    parser.add_argument('--tok_hid', type=int, default=250,
                        help='number of hidden units of the token level rnn layer')
    parser.add_argument('--char_hid', type=int, default=50,
                        help='number of hidden units of the character level rnn layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--cut_freq', type=int, default=10,
                        help='cut off tokens in a corpus less than this value')
    parser.add_argument('--max_vocab_size', type=int, default=100000,
                        help='cut off low-frequencey tokens in a corpus if the vocabulary size exceeds this value')
    parser.add_argument('--max_length', type=int, default=300,
                        help='skip sentences more than this value')
    parser.add_argument('--init_range', type=float, default=0.1,
                        help='initialization range of the weights')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='report interval')

    opts = parser.parse_args()
    return opts


def evaluate(opts, device, corpus, model, criterion, epoch):
    """
    Parameters
    ----------
        opts: command line arguments
        device: device type
        corpus: Corpus
        model: Model
        criterion: loss function
        epoch: current epoch
    Return
    ------
        total_loss: float
    """
    epoch_start_time = time.time()

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_id, batch in enumerate(data.data2batch(corpus.valid, corpus.dictionary, opts.batch_size, flag_shuf=True)):

            input = model.batch2input(batch, device)

            target = model.batch2target(batch, device)

            model.zero_grad()

            output = model(input)

            total_loss += criterion(output, target).item()

            total_num = batch_id + 1
    total_loss /= total_num
    print('-' * 89)
    try:
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), total_loss, math.exp(total_loss)))
    except:
        print("Warning: math error")
    print('-' * 89)
    return total_loss


def train(opts, device, corpus, model, criterion, optimizer, lr, epoch):
    """
    Parameters
    ----------
        opts: command line arguments
        device: device type
        corpus: Corpus
        model: Model
        criterion: loss function
        optimizer: optimizer
        lr: learning rate (float)
        epoch: current epoch
    """
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch_id, batch in enumerate(data.data2batch(corpus.train, corpus.dictionary, opts.batch_size, flag_shuf=True)):

        input = model.batch2input(batch, device)

        target = model.batch2target(batch, device)
        # clear previous gradients
        model.zero_grad()

        output = model(input)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()


        if batch_id % opts.log_interval == 0 and batch_id > 0:
            cur_loss = total_loss / opts.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_id, len(corpus.train) // opts.batch_size, lr,
                                 elapsed * 1000 / opts.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def main():
    ###############################################################################
    # Load command line options.
    ###############################################################################

    opts = options()
    # Set the random seed manually for reproducibility.
    torch.manual_seed(opts.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(opts)

    corpus.load_data(opts.data)

    ###############################################################################
    # Build a model
    ###############################################################################

    # convert to parameters
    params = models.opts2params(opts, corpus.dictionary)
    # construct model
    model = models.WeightedGraphConvolutionNetwork(params)

    # save parameters
    with open(opts.save + ".params", mode='wb') as f:
        pickle.dump(params, f)

    if torch.cuda.is_available():
        if not opts.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        if opts.cuda:
            print("Error: No CUDA device. Remove the option --cuda")
    device = torch.device("cuda" if opts.cuda else "cpu")
    model = model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    ###############################################################################
    # Train the  model
    ###############################################################################

    # Loop over epochs.
    lr = opts.lr
    best_val_loss = None

    # Select an optimizer
    try:
        optimizer = getattr(torch.optim, opts.optim_type)(model.parameters(), lr=lr)
    except:
        raise ValueError("""An invalid option for `--optim_type` was supplied.""")

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, opts.epochs + 1):
            train(opts, device, corpus, model, criterion, optimizer, lr, epoch)
            val_loss = evaluate(opts, device, corpus, model, criterion, epoch)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                torch.save(model.state_dict(), opts.save + ".pt")
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
            optimizer.lr = lr
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


if __name__ == "__main__":
    main()
