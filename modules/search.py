from haystack.pipelines import ExtractiveQAPipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

# LangChain index elérési útja
VECTORSTORE_PATH = "/mnt/c/GPT AI/LangChain/vectorstore"

def search_documents(query, retriever, reader, top_k_retriever=10, top_k_reader=5):
    """Haystack-alapú dokumentumkeresés."""
    pipe = ExtractiveQAPipeline(reader, retriever)
    prediction = pipe.run(
        query=query,
        params={"Retriever": {"top_k": top_k_retriever}, "Reader": {"top_k": top_k_reader}}
    )
    return prediction["answers"]

def search_langchain(query, top_k=5):
    """LangChain-alapú keresés."""
    # LangChain index betöltése
    try:
        embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings=embedding)

        # Hasonlóság alapú keresés
        results = vectorstore.similarity_search(query, k=top_k)

        # Válasz formázása
        formatted_results = [
            {"content": result.page_content, "source": result.metadata.get("source", "N/A")}
            for result in results
        ]
        return formatted_results
    except Exception as e:
        return {"error": f"Hiba a LangChain index használatakor: {e}"}

