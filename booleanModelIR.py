#!/usr/bin/env python3

import pandas as pd
import re
from collections import defaultdict

pd.set_option("display.max_colwidth", None)

# -----------------------------
# STEP 1: 데이터 로딩 & 전처리
# -----------------------------
def load_documents(csv_path):
    df = pd.read_csv(csv_path)
    documents = []
    for i, row in df.iterrows():
        text = f"{row['Title']} {row['Tag']} {row['Content']}"
        text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text.lower())  # 특수문자 제거 + 소문자화
        documents.append(text)
    return df, documents

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
# STEP 3: Boolean 검색 엔진 (우선순위 완전 반영)
# -----------------------------
def boolean_search(query, index, total_docs):
    def tokenize(query):
        query = query.lower()
        query = re.sub(r'([()])', r' \1 ', query)
        return query.split()

    def precedence(op):
        return {'NOT': 3, 'AND': 2, 'OR': 1}.get(op, 0)

    def apply_op(op, values):
        if op == 'NOT':
            val = values.pop()
            return set(range(total_docs)) - val
        right = values.pop()
        left = values.pop()
        if op == 'AND':
            return left & right
        if op == 'OR':
            return left | right
        return set()

    def eval_query(tokens):
        values = []
        ops = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == '(':
                ops.append(token)
            elif token == ')':
                while ops and ops[-1] != '(':
                    values.append(apply_op(ops.pop(), values))
                ops.pop()
            elif token in {'AND', 'OR', 'NOT'}:
                while ops and precedence(ops[-1]) >= precedence(token):
                    values.append(apply_op(ops.pop(), values))
                ops.append(token)
            else:
                values.append(index.get(token, set()))
            i += 1

        while ops:
            values.append(apply_op(ops.pop(), values))
        return values[-1] if values else set()

    tokens = tokenize(query)
    return eval_query(tokens)

# -----------------------------
# STEP 4: 실행
# -----------------------------
if __name__ == "__main__":
    csv_path = "./Financial.csv"  # 🔁 여기에 실제 파일 경로를 입력하세요
    df, documents = load_documents(csv_path)
    index = build_inverted_index(documents)

    queries = [
        "car AND electric",
        "asdad OR upbeat",
        # "NOT adasdasd",
        # "(battery AND car) OR (electric AND NOT fire)"
    ]

    for q in queries:
        result = boolean_search(q, index, len(documents))
        sorted_result = sorted(result)

        print(f"\n ============================================= 🔍 Query: {q} =============================================")
        print(f"📄 Matching document IDs: {sorted_result}")

        if sorted_result:
            first_id = sorted_result[0]
            row = df.iloc[first_id]

            print("\n📌 First matched document (formatted):\n")
            print(f"🧾 Title:\n{row['Title']}\n")
            print(f"🏷️ Tag:\n{row['Tag']}\n")
            print(f"📄 Content:\n{row['Content']}\n")

            # print("\n📌 Processed text used for search:")
            # print(documents[first_id])

            print("\n🛠 Debug: term inclusion check")
            print(f"  'car' in doc: {'car' in documents[first_id].split()}")
            print(f"  'electric' in doc: {'electric' in documents[first_id].split()}")
        else:
            print("❌ No results found.")

