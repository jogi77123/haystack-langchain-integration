import logging
import os
import torch
import faiss
import time
from modules.document_handler import load_hashes, save_hashes, index_documents_recursive
from modules.indexer import initialize_faiss_store
from modules.search import search_haystack_documents, search_langchain_documents
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document  # Document importálása
from haystack.nodes import FARMReader

# Logging beállítása
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# WSL-specifikus beállítások
LIBRARY_PATH = "/mnt/c/GPT AI/Library"
SQL_URL = "sqlite:////mnt/c/GPT AI/Library index/faiss_document_store.db"
EMBEDDING_DIM = 768
BATCH_SIZE = 500
HASHES_FILE = "/mnt/c/GPT AI/Library index/document_hashes.json"
VECTORSTORE_PATH = "/mnt/c/GPT AI/LangChain/vectorstore"
FAISS_INDEX_PATH = "/mnt/c/GPT AI/Library index/faiss_index"
INDEX_FILE = "/mnt/c/GPT AI/Library index/faiss_index/index.faiss"
JSON_FILE = "/mnt/c/GPT AI/Library index/faiss_index/faiss_index.json"

# GPU ellenőrzés
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

        embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        vectorstore = FAISS.from_documents(documents, embedding)
        vectorstore.save_local(VECTORSTORE_PATH)

        logging.info("LangChain FAISS index sikeresen létrehozva.")
    else:
        logging.info("LangChain FAISS index már létezik.")

if __name__ == "__main__":
    logging.info("Haystack dokumentumtár inicializálása...")

    if os.path.exists(FAISS_INDEX_PATH):
        logging.info("🛠 FAISS index betöltése korábbi mentésből...")
        document_store = FAISSDocumentStore.load(FAISS_INDEX_PATH)
    else:
        logging.info("🆕 Új FAISS index létrehozása...")
        document_store = FAISSDocumentStore(
            sql_url=SQL_URL,
            faiss_index_factory_str="Flat",
            embedding_dim=EMBEDDING_DIM,
            index="document",
            validate_index_sync=False,
            return_embedding=True
        )

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_format="sentence_transformers",
        use_gpu=torch.cuda.is_available()
    )

    docs_count = document_store.get_document_count()
    logging.info(f"📌 FAISS index dokumentumszáma: {docs_count}")

    if docs_count == 0:
        logging.warning("⚠️ Az FAISS index üres! Embeddingek újragenerálása...")
        document_store.update_embeddings(retriever, batch_size=BATCH_SIZE)
        logging.info("✅ Embeddingek sikeresen frissítve.")

    document_hashes = load_hashes(HASHES_FILE)
    logging.info("Dokumentumok indexelése...")
    index_documents_recursive(LIBRARY_PATH, document_store, document_hashes, retriever, BATCH_SIZE, HASHES_FILE)
    update_langchain_index(document_store)

    logging.info("Hash fájl ellenőrzése...")
    if os.path.exists(HASHES_FILE):
        logging.info("Hash fájl sikeresen létrehozva.")
    else:
        logging.error("Hash fájl nem található.")

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=torch.cuda.is_available())    

    query = "Magyarország fővárosa?"
    results = search_haystack_documents(query, retriever, reader, top_k_retriever=10, top_k_reader=1)
    langchain_results = search_langchain_documents(query, top_k=5)

    if results:
        logging.info(f"Keresési eredmények a kérdésre: '{query}'")
        for result in results:
            answer = getattr(result, "answer", "Nincs válasz")
            score = getattr(result, "score", 0)
            document_name = result.meta.get("name", "Ismeretlen dokumentum") if hasattr(result, "meta") else "Ismeretlen dokumentum"
            logging.info(f"- Válasz: {answer} | Pontosság: {score:.2f} | Dokumentum: {document_name}")
    else:
        logging.info(f"Nincs találat a kérdésre: '{query}'")

# 🔴 1. Fájlok teljes törlése (ha léteznek)
def force_delete_faiss_files():
    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    json_file = os.path.join(FAISS_INDEX_PATH, "faiss_index.json")

    for file_path in [index_file, json_file]:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"🗑️ Törölve: {file_path}")
                time.sleep(0.1)  # Várunk, hogy a rendszer teljesen felszabadítsa a fájlt
        except Exception as e:
            logging.error(f"❌ Hiba történt a fájl törlése közben: {e}")

force_delete_faiss_files()

# 🔴 2. FAISS index teljes újraépítése
logging.info("🛠 FAISS index újraépítése...")

try:
    # Ha a dokumentumtár üres vagy hibás, hozzunk létre egy teljesen újat!
    if document_store.get_document_count() == 0:
        logging.warning("⚠️ A FAISS index üres, új index inicializálása...")
        document_store = FAISSDocumentStore(
            sql_url="sqlite:///",
            faiss_index_factory_str="Flat",
            embedding_dim=768,
            index="document",
            return_embedding=True
        )
    
    # 🔴 3. FAISS index mentése (változatlanul hagyva!)
    logging.info("🛠 FAISS index mentése leállítás előtt...")
    document_store.save(FAISS_INDEX_PATH)
    logging.info("✅ FAISS index sikeresen mentve.")

except Exception as e:
    logging.error(f"❌ Hiba a FAISS index mentése során: {e}")




