import sqlite3
import argparse
import time
try:
    from retriever.utils import find_hyper_linked_titles, remove_tags, normalize
except:
    from utils import find_hyper_linked_titles, remove_tags, normalize

import numpy as np

from functools import reduce
from itertools import chain
from multiprocessing.pool import ThreadPool
from operator import *
import re

from pyspark.sql.functions import col
import tqdm

def fetchall(path, fields=[]):
    connection = sqlite3.connect(path, check_same_thread=False)
    cursor = connection.cursor()
    cursor.execute(f"SELECT {', '.join(fields) if len(fields) > 0 else '*'} FROM documents")
    ret = cursor.fetchall()
    cursor.close()
    return ret

def gr(sc, path='models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db'):
    data = fetchall(path, ['original_title', 'linked_title'])

    df = sc.parallelize(data).flatMap(lambda r: [[r[0], x] for x in set(r[1].split('\t'))]).toDF(['src_title', 'dst_title'])

    _title2id = sc.parallelize([[i, title] for i, [title, _] in enumerate(data)]).toDF(['id', 'title'])
    title2id = lambda x: _title2id.withColumnRenamed('id', x).withColumnRenamed('title', x + '_title')

    _df = df.join(title2id('src'), 'src_title', 'inner').join(title2id('dst'), 'dst_title', 'inner')
    src, dst = map(np.array, zip(*_df.rdd.map(lambda r: [r['src'], r['dst']]).collect()))

    prefix = 'models/hotpot_models'
    _title2id.write.parquet(prefix + '/title2id.parquet', mode='overwrite')
    np.save(prefix + '/src', src)
    np.save(prefix + '/dst', dst)

    return _title2id, src, dst

def filter_links(link2sent, k):
    dst = link2sent.groupBy('dst').count().filter(col('count') < k)
    return link2sent.join(dst, 'dst', 'inner')

def closest_docs(sqlCtx, hotpot, ranker, k, db):
    qids, qs = zip(*hotpot.rdd.map(lambda r: [r['_id'], r['question']]).collect())
    xs, _ = zip(*ranker.batch_closest_docs(qs, k))
    df = sqlCtx.createDataFrame(list(chain(*[zip(len(x) * [qid], len(x) * [q], x)
                                             for qid, q, x in zip(qids, qs, xs)])), ['qid', 'q', 'id'])
    return df.join(db, on='id', how='inner').withColumnRenamed('original_title', 'title').select('qid', 'q', 'title').distinct().persist()

'''
def closest_docs(sc, hotpot, ranker, k, db):
    _ranker = sc.broadcast(ranker)
    def flat_mapper(r):
        doc_ids, _ = _ranker.value.closest_docs(r['question'], k)
        return [[r['_id'], r['question'], doc_id] for doc_id in doc_ids]

    df = hotpot.rdd.flatMap(flat_mapper).toDF(['qid', 'q', 'id'])
    return df.join(db, on='id', how='inner').withColumnRenamed('original_title', 'title').select('qid', 'q', 'title').distinct().persist()
'''

def expand(titles, link2sent, ranker, k):
    def flat_mapper(pair):
        [qid, q], values = pair
        us, vs, sents = zip(*values)
        return [[qid, q, us[i], vs[i]] for i in ranker.closest_sents(q, sents, k)]

    mapper = lambda r: [(r['qid'], r['q']), [r['src'], r['dst'], r['sent']]]
    join = lambda col: titles.withColumnRenamed('title', col).join(link2sent, col, 'inner').rdd.map(mapper)
    rdd = join('src').union(join('dst')).groupByKey().flatMap(flat_mapper)
    return rdd.toDF(['qid', 'q', 'src', 'dst']).distinct().persist()

def build_inputs(df, gold, link2sent):
    rdd = df.join(link2sent, ['src', 'dst'], 'inner').rdd
    mapper = lambda r: [r['qid'], r['q'], r['sent'], {r['src'], r['dst']} == set(r['titles'])]
    return df.join(link2sent, ['src', 'dst'], 'inner').join(gold, 'qid', 'inner').rdd.map(mapper).persist()

def filter_inputs(rdd):
    def mapper(r):
        qid, q, sent, y = r
        return qid, tokenizer.encode(q, sent), y

    def filterer(r):
        _, ids, _ = r
        return len(ids) > idenizer.max_model_input_sizes['bert-base-uncased']

    return zip(*rdd.map(mapper).filter(filterer).collect())  # qid, ids, y

class ListOfIDLists:
    def __init__(self, xs, path=''):
        self.path = path
        self.data = np.hstack(list(map(np.array, xs)))
        self.indptr = np.cumsum(np.array([0] + list(map(len, xs))))

    def __getitem__(self, index):
        data = self.data[self.indptr[index] : self.indptr[index + 1]]
        return data.tolist()

    def __getstate__(self):
        if not self.path:
            raise RuntimeError()
        np.save(self.path + '-data', self.data)
        np.save(self.path + '-indptr', self.indptr)
        return {k : v for k, v in self.__dict__.items() if k not in ['data', 'indptr']}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.data, self.indptr = np.load(self.path + '-data.npy'), np.load(self.path + '-indptr.npy')

def link2sent(sqlCtx, enwiki, title2id):
    def flat_mapper(row):
        return sum(([[(row['title'], title), sent] for title in find_hyper_linked_titles(sent_with_links)]
                    for sent, sent_with_links in zip(row['text'], row['text_with_links'])), [])

    def mapper(row):
        [[src, dst], sent] = row
        return src, dst, sent

    rdd = enwiki.rdd.flatMap(flat_mapper).reduceByKey(lambda s, t: ' '.join([s, t])).map(mapper)
    titles = title2id.select('title').withColumnRenamed('title', 'dst')
    df = rdd.toDF(['src', 'dst', 'sent']).join(titles, 'dst', 'inner').persist()
    return df

def link2toks(link2sent, tokenizer):
    return link2sent.rdd.map(lambda r: [r['src'], r['dst'], tokenizer.tokenize(r['sent'])]).toDF(['src', 'dst', 'toks']).persist()

def recall(sc, gold, hops):
    titles = reduce(type(hops[0]).union, hops).distinct()
    gold = sc.parallelize(gold).toDF(['qid', 'titles'])
    pairs = titles.join(gold, on='qid', how='inner')
    _, ns = zip(*pairs.rdd.map(lambda r: [r['qid'], r['title'] in r['titles']]).reduceByKey(add).collect())
    return ns

class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None):
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (doc_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def get_hyper_linked(self, doc_id):
        """Fetch the hyper-linked titles of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT linked_title FROM documents WHERE id = ?",
            (doc_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if (result is None or len(result[0]) == 0) else [normalize(title) for title in result[0].split("\t")]

    def get_original_title(self, doc_id):
        """Fetch the original title name  of the doc."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT original_title FROM documents WHERE id = ?",
            (doc_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    def get_doc_text_hyper_linked_titles_for_articles(self, doc_id):
        """
        fetch all of the paragraphs with their corresponding hyperlink titles.
        e.g., 
        >>> paras, links = db.get_doc_text_hyper_linked_titles_for_articles("Tokyo Imperial Palace_0")
        >>> paras[2]
        'It is built on the site of the old Edo Castle. The total area including the gardens is . During the height of the 1980s Japanese property bubble, the palace grounds were valued by some to be more than the value of all of the real estate in the state of California.'
        >>> links[2]
        ['Edo Castle', 'Japanese asset price bubble', 'Real estate', 'California']
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (doc_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        if result is None:
            return [], []
        else:
            hyper_linked_paragraphs = result[0].split("\n\n")
            paragraphs, hyper_linked_titles = [], []

            for hyper_linked_paragraph in hyper_linked_paragraphs:
                paragraphs.append(remove_tags(hyper_linked_paragraph))
                hyper_linked_titles.append([normalize(title) for title in find_hyper_linked_titles(
                    hyper_linked_paragraph)])

            return paragraphs, hyper_linked_titles
