"""
NOTE: real deployments use a proper embedding model (sentence-transformers, an
API embedder). Some sandboxed/regulated network environments block downloading
model weights entirely -- this TF-IDF + SVD stand-in proves the exact same
retrieval mechanics (chunk -> vectorize -> store -> nearest-neighbor search)
without needing any network access. Swap the embedder class for a real one
and nothing else in this file changes.
"""
import glob
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class TfidfEmbeddingFunction(EmbeddingFunction):
    def __init__(self, corpus: list[str]):
        self.vectorizer = TfidfVectorizer(max_features=512)
        self.svd = TruncatedSVD(n_components=32, random_state=42)
        tfidf = self.vectorizer.fit_transform(corpus)
        self.svd.fit(tfidf)

    def __call__(self, input: Documents) -> Embeddings:
        return self.svd.transform(self.vectorizer.transform(input)).tolist()

def chunk_text(text, chunk_size=60, overlap=15):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size-overlap) if words[i:i+chunk_size]]

if __name__ == "__main__":
    paths = sorted(glob.glob("rag_docs/*.txt"))
    all_chunks, ids, metadatas = [], [], []
    for path in paths:
        for i, chunk in enumerate(chunk_text(open(path).read())):
            all_chunks.append(chunk); ids.append(f"{path}-{i}"); metadatas.append({"source": path})
    embedder = TfidfEmbeddingFunction(corpus=all_chunks)
    client = chromadb.PersistentClient(path="chroma_db")
    try: client.delete_collection("churn_project_docs")
    except Exception: pass
    collection = client.get_or_create_collection(name="churn_project_docs", embedding_function=embedder)
    collection.add(documents=all_chunks, ids=ids, metadatas=metadatas)
    print(f"Indexed {len(all_chunks)} chunks from {len(paths)} documents")
    for query in ["What accuracy did the champion model get?", "How is drift detected?", "What is the maximum discount?"]:
        results = collection.query(query_texts=[query], n_results=1)
        print(f"\nQuery: {query}")
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print(f"  [{meta['source']}] {doc}")
