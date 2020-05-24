from argparse import ArgumentParser
from functools import partial
import gc
import os
import pickle
import time

import numpy as np
import numpy.random as npr
from sklearn.metrics import *
import torch as th
from torch.optim import Adam
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, BertForNextSentencePrediction, get_linear_schedule_with_warmup

parser = ArgumentParser()
parser.add_argument('--train-batch-size', type=int, required=True)
parser.add_argument('--test-data', type=str, default='data/hotpot/test.pickle')
parser.add_argument('--test-batch-size', type=int, required=True)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n-iters', type=int, required=True)
parser.add_argument('--num-warmup-steps', type=int, default=0)
parser.add_argument('--train-data', type=str, default='data/hotpot/train.pickle')
parser.add_argument("--weight-decay", type=float, default=0)
args = parser.parse_args()

def batch(xs, ys=None):
    max_length = max(map(len, xs))
    d = tokenizer.batch_encode_plus(xs, add_special_tokens=False, max_length=max_length, pad_to_max_length=True, is_pretokenized=True, return_tensors='pt')
#   d = tokenizer.batch_encode_plus(xs, add_special_tokens=False, max_length=max(map(len, _xs)), pad_to_max_length=True, is_pretokenized=True, return_tensors='pt')
    bert_args = place([d['input_ids'], d['attention_mask'], d['token_type_ids']])
    bert_kwargs = {} if ys is None else {'next_sentence_label' : place(long_tensor(ys))}
    return bert_args, bert_kwargs

def train(bert, xs, ys, z, nz):
#   indices = npr.randint(len(qids), size=args.batch_size)
    indices = np.hstack([npr.choice(z, size=args.train_batch_size // 2),
                         npr.choice(nz, size=args.train_batch_size // 2)])

    bert_args, bert_kwargs = batch([xs[i] for i in indices], [not ys[i] for i in indices])

    bert.train()
    next_sentence_loss, _ = bert(*bert_args, **bert_kwargs)

    optimizer.zero_grad()
    next_sentence_loss.backward()
    optimizer.step()

    return next_sentence_loss.item()

def test(bert, xs, ys, z, nz):
    indices = np.hstack([nz, npr.choice(z, size=len(nz))])

    with th.no_grad():
        bert.eval()
        y_preds = []
        for ii in tqdm(np.array_split(indices, len(indices) // args.test_batch_size + 1)):
            bert_args, bert_kwargs = batch([xs[i] for i in ii])
            [seq_relationship_score] = bert(*bert_args, **bert_kwargs)
            _, y_pred = seq_relationship_score.max(1)
            y_preds.append(y_pred.cpu().numpy())

    y = np.hstack([np.zeros_like(nz), np.ones_like(nz)])
    y_pred = np.hstack(y_preds)
    return precision_recall_fscore_support(y, y_pred, pos_label=0, average='binary')

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

    '''
    print('Loading training data...')
    t = time.time()
    _, xs_train, ys_train = pickle.load(open(args.train_data, 'rb'))
    print(f'Training data loaded ({time.time() - t}).')
    y_train = np.array(ys_train)
    '''

    print('Loading test data...')
    t = time.time()
    _, xs_test, ys_test = pickle.load(open(args.test_data, 'rb'))
    print(f'Test data loaded ({time.time() - t}).')
    y_test = np.array(ys_test)

#   [z_train], [nz_train] = np.nonzero(np.logical_not(y_train)), np.nonzero(y_train)
    [z_test], [nz_test] = np.nonzero(np.logical_not(y_test)), np.nonzero(y_test)

    p, r, f, s = test(bert, xs_test, ys_test, z_test, nz_test)

#   test_qids, test_xs, test_ys = pickle.load(open(args.test_data, 'rb'))

    for i in range(args.n_iters):
        loss = train(bert, xs_train, ys_train, z_train, nz_train)
#       p, r, f, s = test(bert, xs_test, ys_test)
        print(f'[{i + 1}/{args.n_iters}]{round(loss, 3)}')
