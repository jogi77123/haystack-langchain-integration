import logging
import os
import torch
from modules.document_handler import load_hashes, save_hashes, index_documents_recursive
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from modules.indexer import initialize_faiss_store

# Logging beállítása
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# WSL-specifikus beállítások
LIBRARY_PATH = "/mnt/c/GPT AI/Library"
SQL_URL = "sqlite:////mnt/c/GPT AI/Library index/faiss_document_store.db"
EMBEDDING_DIM = 768
BATCH_SIZE = 500
HASHES_FILE = "/mnt/c/GPT AI/Library index/document_hashes.json"
VECTORSTORE_PATH = "/mnt/c/GPT AI/LangChain/vectorstore"

# GPU ellenőrzés
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"GPU használata: {device}")

def update_langchain_index(document_store):
    """LangChain FAISS index frissítése."""
    if not (os.path.exists(f"{VECTORSTORE_PATH}/index.faiss") and os.path.exists(f"{VECTORSTORE_PATH}/index.pkl")):
        logging.info("LangChain FAISS index nem található. Új index létrehozása...")

        # Dokumentumok kinyerése a Haystack dokumentumtárból
        documents = [
            Document(
                page_content=doc.content,
                metadata=doc.meta
            ) for doc in document_store.get_all_documents()
        ]

        embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        vectorstore = FAISS.from_documents(documents, embedding)
        vectorstore.save_local(VECTORSTORE_PATH)

        logging.info("LangChain FAISS index sikeresen létrehozva.")
    else:
        logging.info("LangChain FAISS index már létezik.")

if __name__ == "__main__":
    logging.info("Haystack dokumentumtár inicializálása...")

    # FAISS tároló inicializálása a külön függvénnyel
    try:
        document_store = initialize_faiss_store(
            sql_url=SQL_URL,
            embedding_dim=EMBEDDING_DIM
        )
    except RuntimeError as e:
        logging.error(f"FAISS tároló inicializálási hiba: {e}")
        exit(1)

    # EmbeddingRetriever inicializálása
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_format="sentence_transformers",
        use_gpu=torch.cuda.is_available()
    )

    # Hash-ek betöltése
    document_hashes = load_hashes(HASHES_FILE)

    # Dokumentumok indexelése
    logging.info("Dokumentumok indexelése...")
    index_documents_recursive(LIBRARY_PATH, document_store, document_hashes, retriever, BATCH_SIZE, HASHES_FILE)

    # LangChain FAISS index frissítése
    update_langchain_index(document_store)

    # Hash fájl ellenőrzése
    logging.info("Hash fájl ellenőrzése...")
    if os.path.exists(HASHES_FILE):
        logging.info("Hash fájl sikeresen létrehozva.")
    else:
        logging.error("Hash fájl nem található.")
