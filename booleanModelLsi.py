#!/usr/bin/env python3

import pandas as pd
import re
import nltk
from nltk.corpus import wordnet
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# NLTK WordNet 데이터 다운로드 (최초 1회)
nltk.download('wordnet')
nltk.download('omw-1.4')

pd.set_option("display.max_colwidth", None)

# -----------------------------
# STEP 1: 데이터 로딩 & 전처리
# -----------------------------
def load_documents(csv_path):
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        text = f"{row['Title']} {row['Tag']} {row['Content']}"
        text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text.lower())
        docs.append(text)
    return df, docs

# -----------------------------
# STEP 2: 역색인 구축
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

    def apply_op(op, values):
        if op == 'NOT':
            val = values.pop()
            return set(range(total_docs)) - val
        right = values.pop()
        left = values.pop()
        if op == 'AND': return left & right
        if op == 'OR': return left | right
        return set()

    def eval_query(tokens):
        values, ops = [], []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == '(': ops.append(token)
            elif token == ')':
                while ops and ops[-1] != '(': values.append(apply_op(ops.pop(), values))
                ops.pop()
            elif token.upper() in {'AND', 'OR', 'NOT'}:
                while ops and precedence(ops[-1]) >= precedence(token.upper()):
                    values.append(apply_op(ops.pop(), values))
                ops.append(token.upper())
            else:
                values.append(index.get(token, set()))
            i += 1
        while ops:
            values.append(apply_op(ops.pop(), values))
        return values[-1] if values else set()

    tokens = tokenize(query)
    return eval_query(tokens)

# -----------------------------
# STEP 4: LSI 모델 준비
# -----------------------------
def build_lsi(docs, n_components=200):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        max_df=0.8,
        min_df=2
    )
    tfidf = vectorizer.fit_transform(docs)
    svd = TruncatedSVD(n_components=n_components)
    lsi_matrix = svd.fit_transform(tfidf)
    normalizer = Normalizer(copy=False)
    lsi_norm = normalizer.fit_transform(lsi_matrix)
    return vectorizer, svd, lsi_norm, normalizer

# -----------------------------
# STEP 5: 유사어 & 동의어 추출
# -----------------------------
print("Loading Word2Vec model...")
word2vec = api.load('word2vec-google-news-300')
print("Model loaded.")

def get_synonyms(term):
    synsets = wordnet.synsets(term)
    synonyms = set(
        lemma.name().lower().replace('_', ' ')
        for syn in synsets for lemma in syn.lemmas()
    )
    synonyms.discard(term)
    return synonyms


def get_similar_terms(term, topn=3):
    try:
        sims = word2vec.most_similar(term, topn=topn)
        return {w.lower() for w, _ in sims}
    except KeyError:
        return set()


def expand_terms(terms):
    expanded = set()
    for t in terms:
        expanded.add(t)
        expanded |= get_synonyms(t)
        expanded |= get_similar_terms(t)
    return expanded

# -----------------------------
# STEP 6: LSI 재랭킹 함수
# -----------------------------
def lsi_rerank(terms, vectorizer, svd, lsi_norm, normalizer, subset_ids, top_k=10, threshold=0.1):
    q = " ".join(terms)
    q = re.sub(r'[^a-zA-Z0-9 ]', ' ', q.lower())
    q_tfidf = vectorizer.transform([q])
    q_lsi = svd.transform(q_tfidf)
    q_norm = normalizer.transform(q_lsi)
    sims = cosine_similarity(q_norm, lsi_norm)[0]
    candidates = [(i, sims[i]) for i in subset_ids if sims[i] >= threshold]
    candidates.sort(key=lambda x: -x[1])
    ranked = [i for i, _ in candidates][:top_k]
    return ranked, sims

# -----------------------------
# STEP 7: 실행 및 결과 출력
# -----------------------------
if __name__ == '__main__':
    csv_path = './Financial.csv'
    df, documents = load_documents(csv_path)
    index = build_inverted_index(documents)
    vectorizer, svd, lsi_norm, normalizer = build_lsi(documents)

    raw_queries = ['car AND electric']
    for q in raw_queries:
        # Boolean 검색으로 후보 문서 필터링
        bool_ids = boolean_search(q, index, len(documents))
        print(f"\nBoolean filter matched {len(bool_ids)} docs")

        # 쿼리에서 실제 키워드 추출
        terms = [t.lower() for t in re.findall(r"\b\w+\b", q)
                 if t.upper() not in {'AND', 'OR', 'NOT'}]
        expanded = sorted(expand_terms(terms))
        print(f"Original terms: {terms}")
        print(f"Expanded terms: {expanded}")

        # LSI로 재랭킹
        lsi_ids, sims = lsi_rerank(expanded, vectorizer, svd, lsi_norm, normalizer,
                                    subset_ids=bool_ids, top_k=10, threshold=0.1)
        print("\nTop LSI-ranked docs within boolean results:")
        for doc_id in lsi_ids:
            row = df.iloc[doc_id]
            print(f"--- ID {doc_id} | score={sims[doc_id]:.3f}")
            print(f"Title : {row['Title']}")
            print(f"Content : {row['Content']}")
            print()
