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
import torch.nn as nn
from torch.optim import Adam
from torch_scatter import scatter_max
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, BertForNextSentencePrediction, get_linear_schedule_with_warmup

def sample(indptr, size, n, k):
    """
    n : number of questions
    k : number of negative samples per question
    """
    q = npr.choice(len(size), size=n)
    pos = np.reshape(indptr[q], [-1, 1])
    neg = npr.rand(n, k) * np.reshape(size[q] - 1, [-1, 1])
    return np.reshape(np.hstack([pos, 1 + pos + neg.astype(int)]), -1)

def batch(xs, ys=None):
    max_length = max(map(len, xs))
    d = tokenizer.batch_encode_plus(xs, add_special_tokens=False, max_length=max_length, pad_to_max_length=True, is_pretokenized=True, return_tensors='pt')
    bert_args = place([d['input_ids'], d['attention_mask'], d['token_type_ids']])
    bert_kwargs = {} if ys is None else {'next_sentence_label' : place(long_tensor(ys))}
    return bert_args, bert_kwargs

# def train(bert, xs, ys, z, nz):
def train(bert, xs, ys, indptr, size):
    a, b = args.ratio
    c =  a + b
    '''
    i = np.hstack([npr.choice(nz, size=a * args.train_batch_size // c),
                   npr.choice(z, size=b * args.train_batch_size // c)])
    '''
    i = sample(indptr, size, args.train_batch_size // c, b)

    bert_args, bert_kwargs = batch([xs[j] for j in i], [not ys[j] for j in i])

    bert.train()
    next_sentence_loss, seq_relationship_scores = bert(*bert_args, **bert_kwargs)

    '''
    optimizer.zero_grad()
    next_sentence_loss.backward()
    optimizer.step()

    return next_sentence_loss.item()
    '''

    logp = seq_relationship_scores.log_softmax(1)
    pos = logp[:, 0].view(-1, c)[:, 0]
    neg = logp[:, 1].view(-1, c)[:, 1:].sum(1)
    loss = -th.mean(pos + neg)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

'''
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
'''

'''
def test(bert, xs, ys, indptr, size):
    i = sample(indptr, size, 1000, 7)  # 250 batches
    with th.no_grad():
        em = 0
        for j in tqdm(np.array_split(i, len(i) // 32)):
            bert_args, bert_kwargs = batch([xs[k] for k in j])
            [seq_relationship_score] = bert(*bert_args, **bert_kwargs)
            probability = seq_relationship_score.log_softmax(1)[:, 0]
            _, argmax = probability.view(4, 8).max(1)
            em += argmax.eq(0).sum().item()
        return em / 1000
'''

def test(bert, xs, indptr, size):
    ps = []
    with th.no_grad():
        for i in tqdm(range(len(xs) // 32 + 1)):
            bert_args, bert_kwargs = batch([xs[32 * i + j] for j in range(32)])
            [seq_relationship_score] = bert(*bert_args, **bert_kwargs)
            ps.append(seq_relationship_score.softmax(1))
    p = th.cat(ps)
    i = th.arange(len(size), device=device).repeat_interleave(size)
    _, argmax = scatter_max(p, i)
    return argmax.eq(indptr[:-1]) / len(size)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train-batch-size', type=int, default=32)
    parser.add_argument('--test-data', type=str, default='data/hotpot/test.pickle')
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--logdir', type=str, default='')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--n-iters', type=int, required=True)
    parser.add_argument('--num-warmup-steps', type=int, default=0)
    parser.add_argument('--ratio', type=int, nargs='+', default=[1, 1])
    parser.add_argument('--train-data', type=str, default='data/hotpot/train.pickle')
    parser.add_argument('--test-freq', type=int, default=1000)
    parser.add_argument('--weight-decay', type=float, default=0)
    args = parser.parse_args()

    device = th.device('cuda') if 'CUDA_VISIBLE_DEVICES' in os.environ else th.device('cpu')
    long_tensor = partial(th.tensor, dtype=th.long)
    place = lambda x: list(map(place, x)) if type(x) is list else x.to(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertForNextSentencePrediction.from_pretrained('bert-base-uncased').to(device)
    if torch.cuda.device_count() > 1:
        bert = nn.DataParallel(bert)

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
    train_indptr = np.load('data/hotpot/train-indptr.npy')
    print(f'{len(ys_train)} training samples loaded ({_round(time.time() - t)}s).')
    y_train = np.array(ys_train)
    train_size = train_indptr[1:] - train_indptr[:-1]

    print('Loading test data...')
    t = time.time()
    _, xs_test, ys_test = pickle.load(open(args.test_data, 'rb'))
    test_indptr = np.load('data/hotpot/test-indptr.npy')
    print(f'{len(ys_test)} test samples loaded ({_round(time.time() - t)}s).')
    y_test = np.array(ys_test)
    test_size = test_indptr[1:] - test_indptr[:-1]

    [z_train], [nz_train] = np.nonzero(np.logical_not(y_train)), np.nonzero(y_train)
    [z_test], [nz_test] = np.nonzero(np.logical_not(y_test)), np.nonzero(y_test)

    writer = SummaryWriter(args.logdir) if args.logdir else None

    for i in range(args.n_iters):
        if i % args.test_freq == 0:
            '''
            p, r, f1 = test(bert, xs_test, ys_test, z_test, nz_test)
            print(f'[{i}/{args.n_iters}]Precision: {_round(p)} | Recall: {_round(r)} | F1: {_round(f1)}')
            if writer is not None:
                writer.add_scalar('precision', p, i)
                writer.add_scalar('recall', r, i)
                writer.add_scalar('f1', f1, i)
            '''
#           emr = test(bert, xs_test, ys_test, test_indptr, test_size)
            n = 10
            emr = test(bert, xs_test[:test_indptr[n + 1]], test_indptr[:n], test_size[:n])
            print(f'[{i}/{args.n_iters}]EMR: {_round(emr)}')
            if writer is not None:
                writer.add_scalar('emr', emr, i)

#       loss = train(bert, xs_train, ys_train, z_train, nz_train)
        loss = train(bert, xs_train, ys_train, train_indptr, train_size)
        if writer is None:
            print(f'[{i + 1}/{args.n_iters}]{_round(loss)}')
        else:
            writer.add_scalar('loss', loss, i + 1)
