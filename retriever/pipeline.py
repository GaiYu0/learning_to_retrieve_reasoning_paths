db = DocDB('models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db')
ranker = TfidfDocRanker(tfidf_path='models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz')
data = json.load('data/hotpot/hotpot_train_v1.1.json')
