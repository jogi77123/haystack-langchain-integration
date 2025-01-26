from haystack.pipelines import ExtractiveQAPipeline

def search_documents(query, retriever, reader, top_k_retriever=10, top_k_reader=5):
    """Dokumentumok keresése."""
    pipe = ExtractiveQAPipeline(reader, retriever)
    prediction = pipe.run(
        query=query,
        params={"Retriever": {"top_k": top_k_retriever}, "Reader": {"top_k": top_k_reader}}
    )
    return prediction["answers"]
