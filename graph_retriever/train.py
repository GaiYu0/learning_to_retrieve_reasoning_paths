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
from transformers import AdamW, BertTokenizer, BertForNextSentencePrediction, get_linear_schedule_with_warmup

parser = ArgumentParser()
parser.add_argument('--train-batch-size', type=int, required=True)
parser.add_argument('--test-data', type=str, default='data/hotpot/dev.pickle')
parser.add_argument('--test-batch-size', type=int, required=True)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n-iters', type=int, required=True)
parser.add_argument('--num-warmup-steps', type=int, default=0)
parser.add_argument('--train-data', type=str, default='data/hotpot/train.pickle')
parser.add_argument("--weight-decay", type=float, default=0)
args = parser.parse_args()

def train(bert, xs, ys, z, nz):
#   indices = npr.randint(len(qids), size=args.batch_size)
    indices = np.hstack([npr.choice(z, size=args.train_batch_size // 2),
                         npr.choice(nz, size=args.train_batch_size // 2)])

    d = tokenizer.batch_encode_plus(xs, max_length=max(map(len, xs)), pad_to_max_length=True, is_pretokenized=True, return_tensors='pt')
    bert_args = place([d['input_ids'], d['attention_mask'], d['token_type_ids']])
    bert_kwargs = {'next_sentence_label' : place(long_tensor([not ys[i] for i in indices]))}

    bert.train()
    next_sentence_loss, _ = bert(*bert_args, **bert_kwargs)

    optimizer.zero_grad()
    next_sentence_loss.backward()
    optimizer.step()

    return next_sentence_loss.item()

def test(bert, xs, y):
    with th.no_grad():
        bert.test()
        y_preds = []
        for x in xs:
            _, y_pred = bert(*x).max(1)
            y_preds.append(y_pred.cpu().numpy())

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

    print('Loading training data...')
    t = time.time()
    train_qids, train_xs, train_ys = pickle.load(open(args.train_data, 'rb'))
    print(f'Training data loaded ({time.time() - t}).')
    train_y = np.array(train_ys)

    print('Loading test data...')
    t = time.time()
    [z], [nz] = np.nonzero(np.logical_not(train_y)), np.nonzero(train_y)
    print(f'Test data loaded ({time.time() - t}).')

#   test_qids, test_xs, test_ys = pickle.load(open(args.test_data, 'rb'))

    for i in range(args.n_iters):
        loss = train(bert, train_xs, train_ys, z, nz)
#       p, r, f, s = test(bert, test_xs, test_ys)
        print(f'[{i + 1}/{args.n_iters}]{round(loss, 3)}')
