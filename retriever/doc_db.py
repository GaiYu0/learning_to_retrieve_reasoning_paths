import sqlite3
import argparse
import time
try:
    from retriever.utils import find_hyper_linked_titles, remove_tags, normalize
except:
    from utils import find_hyper_linked_titles, remove_tags, normalize

import numpy as np

from itertools import chain
from multiprocessing.pool import ThreadPool
import re

import tqdm

def fetchall(fields=[], path='models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db'):
    connection = sqlite3.connect(path, check_same_thread=False)
    cursor = connection.cursor()
    cursor.execute(f"SELECT {', '.join(fields) if len(fields) > 0 else '*'} FROM documents")
    ret = cursor.fetchall()
    cursor.close()
    return ret

def gr(sqlCtx):
    fields = ['linked_title', 'original_title']
    ret = fetchall(fields)
    for i in range(len(ret)):
        ret[i] += (i,)
    df = sqlCtx.createDataFrame(ret, fields + ['#'])
    src = df[['#', 'linked_title']].rdd.flatMap(lambda d: [(d['#'], title) for title in d['linked_title'].split('\t')]).toDF(['src', 'title'])
    dst = df[['#', 'original_title']].withColumnRenamed('#', 'dst').withColumnRenamed('original_title', 'title')

    df = src.join(dst, 'title', 'inner')
    np.save('src', np.array(df[['src']].rdd.flatMap(lambda x: x).collect(), dtype=np.long))
    np.save('dst', np.array(df[['dst']].rdd.flatMap(lambda x: x).collect(), dtype=np.long))

    return df

def closest_docs(sqlCtx, hotpot, ranker, k, db, path='models/hotpot_models/closest_docs0.parquet'):
    qids = hotpot[['_id']].rdd.flatMap(lambda x: x).collect()
    qs = hotpot[['question']].rdd.flatMap(lambda x: x).collect()
    doc_ids, _ = zip(*ranker.batch_closest_docs(qs, k))
    _doc_ids = sqlCtx.createDataFrame(list(chain(*[zip(len(x) * [qid], len(x) * [q], x) for qid, q, x in zip(qids, qs, doc_ids)])), ['qid', 'q', 'id'])
    titles = db.join(_doc_ids, on='id', how='inner').withColumnRenamed('original_title', 'title')[['qid', 'q', 'title']]
    titles.write.parquet(path, mode='overwrite')
    return titles

def more_docs(sqlCtx, titles, enwiki, ranker, k, radius, num_workers=None, path='models/hotpot_models/closest_docs%d.parquet'):
    def closest_docs(row):
        indices = ranker.closest_sents(row['q'], row['text'], len(row['text']))
        _titles = []
        for index in indices:
            if len(_titles) == k:
                break
            sent_with_links = row['text_with_links'][index]
            appending = False
            for m, n in row['charoffset_with_links'][index]:
                tok = sent_with_links[m : n]
                if tok.startswith('<a href='):
                    appending = True
                    _title = ''
                elif tok == '</a>':
                    try:
                        _titles.append([row['qid'], row['q'], _title])
                    except UnboundLocalError:
                        pass
                    appending = False
                elif appending:
                    _title = f'{_title} {tok}' if _title else tok

        return _titles

    _titles = enwiki.join(titles, on='title', how='inner').rdd.flatMap(closest_docs).toDF(['qid', 'q', 'title'])

    _titles.write.parquet(path % radius, mode='overwrite')
    return _titles

def recall(hops):
    titles = None
    gold = sc.parallelize(gold).toDF(['qid', 'titles'])
    pairs = titles.join(gold, on='qid', how='inner')
    ret = pairs.rdd.map(lambda r: [r['qid'], r['title'] in r['titles']]).reduceByKey(add).collect()

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
