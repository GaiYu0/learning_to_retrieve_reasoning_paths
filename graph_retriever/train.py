from argparse import ArgumentParser
from functools import partial
import gc
import os
import pickle
import time

import numpy as np
import numpy.random as npr
from tensorboardX import SummaryWriter
import torch as th
from torch.optim import Adam
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, BertForNextSentencePrediction, get_linear_schedule_with_warmup

parser = ArgumentParser()
parser.add_argument('--train-batch-size', type=int, required=True)
parser.add_argument('--test-data', type=str, default='data/hotpot/test.pickle')
parser.add_argument('--test-batch-size', type=int, required=True)
parser.add_argument('--logdir', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n-iters', type=int, required=True)
parser.add_argument('--num-warmup-steps', type=int, default=0)
parser.add_argument('--ratio', type=int, nargs='+', default=[1, 1])
parser.add_argument('--train-data', type=str, default='data/hotpot/train.pickle')
parser.add_argument('--test-freq', type=int, required=True)
parser.add_argument('--weight-decay', type=float, default=0)
args = parser.parse_args()

def batch(xs, ys=None):
    max_length = max(map(len, xs))
    d = tokenizer.batch_encode_plus(xs, add_special_tokens=False, max_length=max_length, pad_to_max_length=True, is_pretokenized=True, return_tensors='pt')
    bert_args = place([d['input_ids'], d['attention_mask'], d['token_type_ids']])
    bert_kwargs = {} if ys is None else {'next_sentence_label' : place(long_tensor(ys))}
    return bert_args, bert_kwargs

def train(bert, xs, ys, z, nz):
    a, b = args.ratio
    c =  a + b
    i = np.hstack([npr.choice(nz, size=a * args.train_batch_size // c),
                   npr.choice(z, size=b * args.train_batch_size // c)])

    bert_args, bert_kwargs = batch([xs[j] for j in i], [not ys[j] for j in i])

    bert.train()
    next_sentence_loss, _ = bert(*bert_args, **bert_kwargs)

    optimizer.zero_grad()
    next_sentence_loss.backward()
    optimizer.step()

    return next_sentence_loss.item()

def test(bert, xs, ys, z, nz):
    i = np.hstack([nz, npr.choice(z, size=len(nz))])

    with th.no_grad():
        bert.eval()
        y_hats = []
        for j in tqdm(np.array_split(i, len(i) // args.test_batch_size + 1)):
            bert_args, bert_kwargs = batch([xs[k] for k in j])
            [seq_relationship_score] = bert(*bert_args, **bert_kwargs)
            _, y_hat = seq_relationship_score.max(1)
            y_hats.append(1 - y_hat.cpu().numpy())

    y_hat = np.hstack(y_hats)
    tp = np.sum(y_hat[:len(nz)])
    fp = np.sum(y_hat[len(nz):]) * len(z) / len(nz)
    p = tp / (tp + fp)
    r = tp / len(nz)
    f1 = 2 * p * r / (p + r)

    return p, r, f1

if __name__ == '__main__':
    device = th.device('cuda') if 'CUDA_VISIBLE_DEVICES' in os.environ else th.device('cpu')
    long_tensor = partial(th.tensor, dtype=th.long)
    place = lambda x: list(map(place, x)) if type(x) is list else x.to(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertForNextSentencePrediction.from_pretrained('bert-base-uncased').to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in bert.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in bert.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.n_iters
    )

    _round = partial(round, ndigits=3)

    print('Loading training data...')
    t = time.time()
    _, xs_train, ys_train = pickle.load(open(args.train_data, 'rb'))
    print(f'{len(ys_train)} training samples loaded ({_round(time.time() - t)}s).')
    y_train = np.array(ys_train)

    print('Loading test data...')
    t = time.time()
    _, xs_test, ys_test = pickle.load(open(args.test_data, 'rb'))
    print(f'{len(ys_test)} test samples loaded ({_round(time.time() - t)}s).')
    y_test = np.array(ys_test)

    [z_train], [nz_train] = np.nonzero(np.logical_not(y_train)), np.nonzero(y_train)
    [z_test], [nz_test] = np.nonzero(np.logical_not(y_test)), np.nonzero(y_test)

    writer = SummaryWriter(args.logdir) if args.logdir else None

    for i in range(args.n_iters):
        if i % args.test_freq == 0:
            p, r, f1 = test(bert, xs_test, ys_test, z_test, nz_test)
            print(f'[{i}/{args.n_iters}]Precision: {_round(p)} | Recall: {_round(r)} | F1: {_round(f1)}')
            if writer is not None:
                writer.add_scalar('precision', p, i)
                writer.add_scalar('recall', r, i)
                writer.add_scalar('f1', f1, i)

        loss = train(bert, xs_train, ys_train, z_train, nz_train)
        if writer is None:
            print(f'[{i + 1}/{args.n_iters}]{_round(loss)}')
        else:
            writer.add_scalar('loss', loss, i + 1)
