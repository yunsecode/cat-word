#!/usr/bin/env python3

import pandas as pd
import re
from collections import defaultdict
import numpy as np
import nltk
from nltk.corpus import wordnet
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# NLTK WordNet 데이터 다운로드 (최초 1회)
nltk.download('wordnet')
nltk.download('omw-1.4')

pd.set_option("display.max_colwidth", None)

# -----------------------------
# STEP 1: 데이터 로딩 & 전처리
# -----------------------------
def load_documents(csv_path):
    df = pd.read_csv(csv_path)
    documents = []
    for _, row in df.iterrows():
        text = f"{row['Title']} {row['Tag']} {row['Content']}"
        text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text.lower())
        documents.append(text)
    return df, documents

# -----------------------------
# STEP 2: 역색인 구축 (Boolean 검색용)
# -----------------------------
def build_inverted_index(docs):
    index = defaultdict(set)
    for doc_id, text in enumerate(docs):
        for word in set(text.split()):
            index[word].add(doc_id)
    return index

# -----------------------------
# STEP 3: Boolean 검색 엔진
# -----------------------------
def boolean_search(query, index, total_docs):
    def tokenize(q):
        q = q.lower()
        q = re.sub(r'([()])', r' \1 ', q)
        return q.split()
    def precedence(op):
        return {'NOT': 3, 'AND': 2, 'OR': 1}.get(op, 0)
    def apply_op(op, vals):
        if op == 'NOT':
            v = vals.pop()
            return set(range(total_docs)) - v
        r = vals.pop(); l = vals.pop()
        return (l & r) if op == 'AND' else (l | r)
    def eval_query(tokens):
        vals, ops = [], []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == '(': ops.append(t)
            elif t == ')':
                while ops and ops[-1] != '(': vals.append(apply_op(ops.pop(), vals))
                ops.pop()
            elif t.upper() in {'AND','OR','NOT'}:
                while ops and precedence(ops[-1]) >= precedence(t.upper()):
                    vals.append(apply_op(ops.pop(), vals))
                ops.append(t.upper())
            else:
                vals.append(index.get(t, set()))
            i += 1
        while ops:
            vals.append(apply_op(ops.pop(), vals))
        return vals[-1] if vals else set()
    return eval_query(tokenize(query))

# -----------------------------
# STEP 4: LSI 모델 준비
# -----------------------------
def build_lsi(docs, n_components=100):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(docs)
    svd = TruncatedSVD(n_components=n_components)
    lsi_matrix = svd.fit_transform(tfidf)
    return vectorizer, svd, lsi_matrix

# -----------------------------
# STEP 5: LSI 검색 함수
# -----------------------------
def lsi_search(query, vectorizer, svd, lsi_matrix, top_k=10):
    q = re.sub(r'[^a-zA-Z0-9 ]', ' ', query.lower())
    q_tfidf = vectorizer.transform([q])
    q_lsi = svd.transform(q_tfidf)
    sims = cosine_similarity(q_lsi, lsi_matrix)[0]
    ranked = np.argsort(-sims)[:top_k]
    return list(ranked), sims

# -----------------------------
# STEP 6: 유사어 추출 함수
# -----------------------------
# WordNet 기반 동의어
# gensim Word2Vec 기반 유사어

# 사전 학습된 Word2Vec 모델 로드 (최초 1회, 메모리 주의)
print("Loading Word2Vec model...")
word2vec = api.load('word2vec-google-news-300')
print("Model loaded.")

def get_synonyms(term):
    synsets = wordnet.synsets(term)
    synonyms = set(
        lemma.name().lower().replace('_', ' ')
        for syn in synsets
        for lemma in syn.lemmas()
    )
    synonyms.discard(term)
    return list(synonyms)


def get_similar_terms(term, topn=5):
    try:
        sims = word2vec.most_similar(term, topn=topn)
        return [w.lower() for w, _ in sims]
    except KeyError:
        return []

# -----------------------------
# STEP 7: 쿼리 확장
# -----------------------------
def expand_query(query):
    tokens = re.findall(r"\b\w+\b|[()]", query)
    expanded = []
    for token in tokens:
        if token.upper() in {'AND', 'OR', 'NOT', '(', ')'}:
            expanded.append(token.upper())
        else:
            t = token.lower()
            syns = get_synonyms(t)
            sims = get_similar_terms(t, topn=3)
            terms = set([t] + syns + sims)
            if len(terms) > 1:
                grp = ' OR '.join(terms)
                expanded.append(f"({grp})")
            else:
                expanded.append(t)
    return ' '.join(expanded)

# -----------------------------
# STEP 8: 실행 및 비교
# -----------------------------
if __name__ == '__main__':
    csv_path = './Financial.csv'
    df, documents = load_documents(csv_path)
    index = build_inverted_index(documents)
    vectorizer, svd, lsi_matrix = build_lsi(documents)

    raw_queries = [
        'automobile AND electric',
        # 'battery AND car',
        # 'electric NOT fire',
        # 'auto OR vehicle',
        # 'NOT tesla'
    ]

    for q in raw_queries:
        exp_q = expand_query(q)
        bool_res = sorted(boolean_search(exp_q, index, len(documents)))
        lsi_res, _ = lsi_search(exp_q, vectorizer, svd, lsi_matrix, top_k=20)

        print(f"\n▶ 원본: {q}")
        print(f"▶ 확장된 쿼리: {exp_q}")
        print(f" Boolean: {len(bool_res)}건, 예시 IDs: {bool_res[:5]}")
        print(f" LSI: 상위 20 IDs: {lsi_res}")

        set_b, set_l = set(bool_res), set(lsi_res)
        only_b = set_b - set_l
        only_l = set_l - set_b
        common = set_b & set_l
        print(f" - Boolean 전용: {len(only_b)}")
        print(f" - LSI 전용: {len(only_l)}")
        print(f" - 공통: {len(common)}")
