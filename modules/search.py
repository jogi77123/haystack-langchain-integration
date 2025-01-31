from haystack.pipelines import ExtractiveQAPipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from haystack.nodes import FARMReader
import logging

# LangChain index elérési útja
VECTORSTORE_PATH = "/mnt/c/GPT AI/LangChain/vectorstore"


def search_haystack_documents(query, retriever, reader, top_k_retriever=10, top_k_reader=5):
    """Keresés Haystack retriever és reader segítségével."""
    try:
        pipe = ExtractiveQAPipeline(reader, retriever)
        prediction = pipe.run(
            query=query,
            params={"Retriever": {"top_k": top_k_retriever}, "Reader": {"top_k": top_k_reader}}
        )
        return prediction["answers"]
    except Exception as e:
        logging.error(f"Hiba a Haystack keresés során: {e}")
        raise RuntimeError(f"Haystack keresési hiba: {e}")


def search_langchain_documents(query, top_k=5):
    """LangChain-alapú keresés FAISS indexszel."""
    try:
        embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings=embedding)

        # Hasonlóság alapú keresés
        results = vectorstore.similarity_search(query, k=top_k)

        # Válaszok formázása
        formatted_results = [
            {"content": result.page_content, "source": result.metadata.get("source", "N/A")}
            for result in results
        ]
        return formatted_results
    except Exception as e:
        logging.error(f"Hiba a LangChain keresés során: {e}")
        raise RuntimeError(f"LangChain keresési hiba: {e}")


# Reader inicializálása
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
