# Overall Pipeline Summary

1. **STEP 1: Data Loading & Preprocessing**  
   - `load_documents(csv_path)` reads `Title`, `Tag`, and `Content` from your CSV, concatenates them into one string per document, lower-cases and strips special characters, and returns both the raw DataFrame and a cleaned list of document texts.

2. **STEP 2: Build Inverted Index**  
   - `build_inverted_index(docs)` creates a mapping from each unique word to the set of document IDs in which it appears, enabling O(1) lookups during Boolean search.

3. **STEP 3: Boolean Search Engine**  
   - `boolean_search(query, index, total_docs)`  
     - **Tokenizes** the query (lower-cases and spaces out parentheses).  
     - Uses two stacks—one for operators (AND/OR/NOT with proper precedence) and one for operand sets—to compute set operations.  
     - Returns the set of matching document IDs.

4. **STEP 4: Prepare LSI Model**  
   - `build_lsi(docs, n_components)`  
     - Vectorizes `docs` into a TF–IDF matrix (1–2 grams, English stopwords, document-frequency thresholds).  
     - Applies Truncated SVD to project into a latent semantic space.  
     - Normalizes each document vector to unit length.

5. **STEP 5: Extract Synonyms & Similar Terms**  
   - Load a pretrained Word2Vec model.  
   - `get_synonyms(term)` pulls WordNet synonyms; `get_similar_terms(term)` finds nearest neighbors in Word2Vec space.  
   - `expand_terms(terms)` merges original terms with their synonyms and similar words for query expansion.

6. **STEP 6: LSI Re-ranking**  
   - `lsi_rerank(terms, vectorizer, svd, lsi_norm, normalizer, subset_ids, top_k, threshold)`  
     - Builds a normalized LSI vector for the expanded query.  
     - Computes cosine similarity against all document LSI vectors.  
     - Filters to the Boolean-filtered IDs (`subset_ids`), applies a similarity threshold, and returns the top K document IDs.

7. **STEP 7: Execute & Display Results**  
   1. Load documents & build the inverted index.  
   2. Build the LSI model.  
   3. For each query in `raw_queries`:  
      - Run Boolean search → candidate IDs  
      - Extract & expand terms → display original vs. expanded keywords  
      - Call `lsi_rerank()` → print top document IDs, similarity scores, titles, and content snippets.

---

## 1. IR vs. LSI: Quick Definitions

- **IR (Information Retrieval)**  
  A classic keyword-matching search model.  
  - **Pros:** Simple to implement, very fast.  
  - **Cons:** Cannot handle vocabulary mismatch (e.g. “car” vs. “automobile”).

- **LSI (Latent Semantic Indexing)**  
  Applies SVD to the TF–IDF matrix to create a low-dimensional semantic space and measures **semantic similarity** between documents.  
  - **Pros:** Captures synonyms and related concepts for more flexible retrieval.  
  - **Cons:** Overhead of SVD training and extra computation at query time.

---

## 2. Without vs. With LSI

| Aspect                | Without LSI                              | With LSI                                          |
|-----------------------|------------------------------------------|---------------------------------------------------|
| Matching criterion    | Exact **keyword match**                   | **Semantic similarity**                            |
| Vocabulary mismatch   | ❌ Not addressed                         | ✅ Synonyms & related terms partially captured      |
| Implementation effort | Very easy                                 | Requires TF–IDF → SVD → normalization pipeline      |
| Query performance     | Very fast                                 | Slower due to dimension reduction & similarity compute |
| Diversity & accuracy  | Low                                       | High                                               |

---

## 3. Running the LSI Code

1. **Prepare environment**  
   ```bash
   pip install pandas nltk gensim scikit-learn
   python -m nltk.downloader wordnet omw-1.4

2. **Prepare CSV file**
Ensure Financial.csv contains Title, Tag, and Content columns.

3.	**Run the script**
```bash
python booleanModelLsi.py
```

4. **View results**
The terminal will display Boolean filter counts and the final top LSI-ranked documents with scores.

## 4. Customizing Raw Queries
At the top of the script, define any Boolean queries you like:

```py
raw_queries = [
    "car AND electric",
    "asdad OR upbeat",
    "NOT adasdasd",
    "NOT tesla",
    "(battery AND car) OR (electric AND NOT fire)",
    # add more as needed
]
```
Each query automatically runs through Boolean filtering, term expansion, and LSI re-ranking—outputting the final results.```