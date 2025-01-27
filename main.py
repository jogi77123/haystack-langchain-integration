import logging
import os
import torch
from modules.document_handler import load_hashes, save_hashes, index_documents_recursive
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

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
    index_faiss_path = f"{VECTORSTORE_PATH}/index.faiss"
    index_pkl_path = f"{VECTORSTORE_PATH}/index.pkl"

    # Ellenőrizzük, hogy mindkét fájl létezik-e
    if not (os.path.exists(index_faiss_path) and os.path.exists(index_pkl_path)):
        logging.info("LangChain FAISS index nem található vagy hiányos. Új index létrehozása...")

        # Dokumentumok kinyerése a Haystack dokumentumtárból
        documents = [
            {"page_content": doc.content, "metadata": doc.meta} for doc in document_store.get_all_documents()
        ]

        # LangChain index létrehozása
        embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        vectorstore = FAISS.from_documents(documents, embedding)
        vectorstore.save_local(VECTORSTORE_PATH)

        logging.info("LangChain FAISS index sikeresen létrehozva.")
    else:
        logging.info("LangChain FAISS index már létezik.")

if __name__ == "__main__":
    logging.info("Haystack dokumentumtár inicializálása...")

    try:
        # FAISS dokumentumtár inicializálása
        document_store = FAISSDocumentStore(
            sql_url=SQL_URL,
            faiss_index_factory_str="Flat",
            embedding_dim=EMBEDDING_DIM,
            index="document",
            validate_index_sync=False
        )
    except Exception as e:
        logging.warning(f"FAISS index hiba: {e}")
        logging.info("Új FAISS index generálása...")
        document_store = FAISSDocumentStore(
            sql_url=SQL_URL,
            faiss_index_factory_str="Flat",
            embedding_dim=EMBEDDING_DIM,
            index="document"
        )

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
