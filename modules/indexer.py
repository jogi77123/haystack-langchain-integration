import os
import logging
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document
from modules.document_handler import read_file, calculate_hash

def initialize_faiss_store(sql_url, faiss_index_path, embedding_dim, retriever_model, use_gpu):
    """
    FAISS dokumentumtár inicializálása.

    Args:
        sql_url (str): Az SQL adatbázis elérési útvonala.
        faiss_index_path (str): Az FAISS index fájl elérési útvonala.
        embedding_dim (int): Az embedding dimenziója.
        retriever_model (str): A retriever modell neve.
        use_gpu (bool): GPU használata.

    Returns:
        FAISSDocumentStore: Inicializált dokumentumtár.
        EmbeddingRetriever: Inicializált retriever.
    """
    try:
        if os.path.exists(faiss_index_path):
            logging.info("Meglévő FAISS index betöltése...")
            document_store = FAISSDocumentStore.load(faiss_index_path)
        else:
            logging.warning("FAISS index nem található, új index létrehozása...")
            document_store = FAISSDocumentStore(
                sql_url=sql_url,
                faiss_index_factory_str="Flat",
                embedding_dim=embedding_dim,
                index="document"
            )

        # Retriever inicializálása
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model=retriever_model,
            model_format="sentence_transformers",
            use_gpu=use_gpu
        )

        return document_store, retriever

    except Exception as e:
        logging.error(f"Nem sikerült inicializálni az FAISS dokumentumtárat: {e}")
        raise e

def index_documents_recursive(path, document_store, document_hashes):
    """Dokumentumok rekurzív indexelése, csak az új fájlokat hozzáadva."""
    files = []
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            _, ext = os.path.splitext(file_name)
            if ext in ALLOWED_FILE_TYPES:
                files.append(file_path)

    logging.info(f"{len(files)} fájl található a mappában. Ellenőrzés kezdése...")
    new_documents = []

    for file_path in files:
        content = read_file(file_path)
        if not content:
            logging.warning(f"Hibás vagy üres fájl kihagyva: {file_path}")
            continue

        doc_hash = calculate_hash(content)
        if doc_hash in document_hashes:
            logging.info(f"Dokumentum már indexelve: {file_path}")
            continue

        # Új dokumentum feldolgozása
        document_hashes.add(doc_hash)
        document = Document(content=content, meta={"name": os.path.basename(file_path)})
        new_documents.append(document)

    if new_documents:
        logging.info(f"{len(new_documents)} új dokumentum hozzáadása az indexhez...")
        document_store.write_documents(new_documents)
        save_hashes(document_hashes)
        logging.info("Új dokumentumok sikeresen hozzáadva az adatbázishoz.")
    else:
        logging.info("Nincsenek új dokumentumok.")
