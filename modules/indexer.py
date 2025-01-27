import os
import logging
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document
from modules.document_handler import read_file, calculate_hash

def initialize_faiss_store(sql_url, embedding_dim):
    """
    Inicializálja a FAISS tárolót a megadott paraméterekkel.

    :param sql_url: Az SQL adatbázis URL-je a dokumentum tároló számára.
    :param embedding_dim: Az embedding dimenziója.
    :return: A FAISS dokumentumtár objektuma.
    """
    logging.info("FAISS tároló inicializálása...")

    try:
        document_store = FAISSDocumentStore(
            sql_url=sql_url,
            faiss_index_factory_str="Flat",
            embedding_dim=embedding_dim,
            index="document",
            validate_index_sync=False
        )
        logging.info("FAISS tároló sikeresen inicializálva.")
        return document_store
    except Exception as e:
        logging.error(f"Hiba a FAISS tároló inicializálásakor: {e}")
        raise RuntimeError("Nem sikerült a FAISS tárolót inicializálni.")

def index_documents_recursive(path, document_store, document_hashes, retriever, batch_size, hash_file):
    """Dokumentumok rekurzív indexelése."""
    files = []
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            _, ext = os.path.splitext(file_name)
            if ext in ALLOWED_FILE_TYPES:
                files.append(file_path)

    logging.info(f"{len(files)} fájl található a mappában. Ellenőrzés kezdése...")
    new_documents = []
    failed_files = []

    for i, file_path in enumerate(files, start=1):
        content = read_file(file_path)
        if not content:
            logging.warning(f"Hibás vagy üres fájl: {file_path}")
            failed_files.append(file_path)
            continue

        doc_hash = calculate_hash(content)
        if doc_hash in document_hashes:
            logging.info(f"Dokumentum már indexelve: {file_path}")
            continue

        # Új dokumentum feldolgozása
        document_hashes.add(doc_hash)
        document = Document(content=content, meta={"name": os.path.basename(file_path)})
        new_documents.append(document)

        # Folyamat kijelzése
        progress = (i / len(files)) * 100
        logging.info(f"Feldolgozás: {progress:.2f}% ({i}/{len(files)})")

    if new_documents:
        logging.info(f"{len(new_documents)} új dokumentum hozzáadása az indexhez.")
        document_store.write_documents(new_documents)
        document_store.update_embeddings(retriever, batch_size=batch_size)
        save_hashes(document_hashes, hash_file)
        logging.info("Új dokumentumok sikeresen hozzáadva az adatbázishoz.")
    else:
        logging.info("Nincsenek új dokumentumok.")

    if failed_files:
        logging.error(f"Hibás fájlok: {len(failed_files)}")
        for file in failed_files:
            logging.error(f"- {file}")
