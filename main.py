import logging
import os
import torch
from modules.document_handler import load_hashes, save_hashes, index_documents_recursive
from modules.indexer import initialize_faiss_store
from modules.search import search_documents
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, FARMReader
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

# Logging beállítása
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# WSL-specifikus beállítások
LIBRARY_PATH = "/mnt/c/GPT AI/Library"
SQL_URL = "sqlite:////mnt/c/GPT AI/Library index/faiss_document_store.db"
EMBEDDING_DIM = 768
BATCH_SIZE = 500
HASHES_FILE = "/mnt/c/GPT AI/Library index/document_hashes.json"
VECTORSTORE_PATH = "/mnt/c/GPT AI/LangChain/vectorstore"

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"GPU használata: {device}")

def update_langchain_index(document_store):
    """LangChain FAISS index frissítése."""
    if not (os.path.exists(f"{VECTORSTORE_PATH}/index.faiss") and os.path.exists(f"{VECTORSTORE_PATH}/index.pkl")):
        logging.info("LangChain FAISS index nem található. Új index létrehozása...")
        
        documents = [
            Document(
                page_content=doc.content,
                metadata=doc.meta
            ) for doc in document_store.get_all_documents()
        ]
        
        if not documents:
            logging.warning("Nincsenek dokumentumok a FAISS index létrehozásához! Kihagyás...")
            return
        
        embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        vectorstore = FAISS.from_documents(documents, embedding)
        vectorstore.save_local(VECTORSTORE_PATH)
        
        logging.info("LangChain FAISS index sikeresen létrehozva.")
    else:
        logging.info("LangChain FAISS index már létezik.")

if __name__ == "__main__":
    logging.info("Haystack dokumentumtár inicializálása...")
    
    try:
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
    
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_format="sentence_transformers",
        use_gpu=torch.cuda.is_available()
    )
    
    document_hashes = load_hashes(HASHES_FILE)
    new_documents = []
    
    if new_documents:
        index_documents_recursive(document_store, retriever, new_documents)
        save_hashes(document_hashes, HASHES_FILE)
    else:
        logging.info("Nincsenek új dokumentumok az indexeléshez.")
    
    update_langchain_index(document_store)
    
    logging.info("Hash fájl ellenőrzése...")
    if os.path.exists(HASHES_FILE):
        logging.info("Hash fájl sikeresen létrehozva.")
    else:
        logging.error("Hash fájl nem található.")
    
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=torch.cuda.is_available())
    query = "Moving Average"
    results = search_documents(query, retriever, reader, top_k_retriever=10, top_k_reader=5)
    
    if results:
        logging.info(f"Keresési eredmények a kérdésre: '{query}'")
        for result in results:
            answer = result.get('answer', 'Nincs válasz')
            score = result.get('score', 0)
            document_name = result.get('meta', {}).get('name', 'Ismeretlen dokumentum')
            logging.info(f"- Válasz: {answer} | Pontosság: {score:.2f} | Dokumentum: {document_name}")
    else:
        logging.info(f"Nincs találat a kérdésre: '{query}'")
