"""Streamlit UI for chatting with the TrueHorizon AI knowledge base."""

import pickle
import numpy as np
import faiss
import openai
import streamlit as st
from agno import Agent
import os
import whoosh.index as whoosh_index
from whoosh.qparser import MultifieldParser
from whoosh.scoring import BM25F

INDEX_PATH = "faiss_index.pkl"
WHOOSH_INDEX_DIR = "whoosh_indexdir"
DATA_DIR = "data" # Assuming this is where the text files are, to get fnames

# Constants for RAG
K_RRF_CONSTANT = 60
NUM_RESULTS_PER_RETRIEVER = 10
NUM_CONTEXT_RESULTS = 3


def embed_text(text: str) -> list:
    result = openai.Embedding.create(model="text-embedding-ada-002", input=[text])
    return result["data"][0]["embedding"]


def load_indexes():
    # Load FAISS index
    with open(INDEX_PATH, "rb") as f:
        faiss_data = pickle.load(f)
    faiss_idx = faiss_data["index"]
    texts = faiss_data["texts"]

    # Load Whoosh index
    whoosh_ix = whoosh_index.open_dir(WHOOSH_INDEX_DIR)

    # Load fnames
    fnames = sorted([fname for fname in os.listdir(DATA_DIR) if fname.endswith(".txt")])
    if len(fnames) != len(texts):
        temp_fnames = []
        with whoosh_ix.reader() as reader:
            for doc_num in range(reader.doc_count()):
                stored_fields = reader.stored_fields(doc_num)
                if stored_fields and 'path' in stored_fields:
                    temp_fnames.append(stored_fields['path'])
        if len(temp_fnames) == len(texts):
            fnames = temp_fnames
        else:
            raise RuntimeError(
                f"Mismatch in number of text files and loaded texts. "
                f"len(fnames)={len(fnames)}, len(texts)={len(texts)}, len(temp_fnames) from whoosh={len(temp_fnames)}"
            )
    return faiss_idx, texts, fnames, whoosh_ix


class RAGAgent(Agent):
    """Retrieval-augmented agent using FAISS and Whoosh with Reciprocal Rank Fusion."""

    def __init__(self, faiss_idx, texts, fnames, whoosh_ix):
        super().__init__()
        self.faiss_idx = faiss_idx
        self.texts = texts
        self.fnames = fnames
        self.fname_to_idx = {fname: i for i, fname in enumerate(fnames)}
        self.whoosh_ix = whoosh_ix
        self.whoosh_parser = MultifieldParser(["title", "content"], schema=self.whoosh_ix.schema)

    def _faiss_search(self, query: str, k: int) -> list[tuple[int, float]]:
        query_vector = np.array([embed_text(query)], dtype="float32")
        distances, idxs = self.faiss_idx.search(query_vector, k=k)
        results = []
        for i in range(len(idxs[0])):
            doc_idx = idxs[0][i]
            dist = distances[0][i]
            score = 1.0 / (1.0 + dist)
            results.append((doc_idx, score))
        return results

    def _whoosh_search(self, query_str: str, k: int) -> list[tuple[int, float]]:
        results = []
        with self.whoosh_ix.searcher(weighting=BM25F()) as searcher:
            query_parsed = self.whoosh_parser.parse(query_str)
            search_results = searcher.search(query_parsed, limit=k)
            for hit in search_results:
                fname = hit.fields()['path']
                if fname in self.fname_to_idx:
                    doc_idx = self.fname_to_idx[fname]
                    results.append((doc_idx, hit.score))
                else:
                    print(f"Warning: Whoosh result path {fname} not found in fnames mapping.")
        return results

    def _reciprocal_rank_fusion(self, results_list: list[list[tuple[int, float]]], rrf_k: int = K_RRF_CONSTANT) -> list[tuple[int, float]]:
        fused_scores = {}
        for results in results_list:
            for rank, (doc_idx, _) in enumerate(results):
                score = 1.0 / (rrf_k + rank + 1)
                if doc_idx in fused_scores:
                    fused_scores[doc_idx] += score
                else:
                    fused_scores[doc_idx] = score
        reranked_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return reranked_results

    def run(self, query: str) -> str:
        faiss_results = self._faiss_search(query, k=NUM_RESULTS_PER_RETRIEVER)
        whoosh_results = self._whoosh_search(query, k=NUM_RESULTS_PER_RETRIEVER)

        fused_results = self._reciprocal_rank_fusion([faiss_results, whoosh_results])

        context_docs = []
        for doc_idx, score in fused_results[:NUM_CONTEXT_RESULTS]:
            context_docs.append(self.texts[doc_idx])
        
        if not context_docs:
            context = "No relevant information found."
        else:
            context = "\n\n---\n\n".join(context_docs)

        prompt = (
            "Answer the user question using ONLY the following website content. "
            "Do not use any external knowledge. If the answer is not found in the content, say so.\n\n"
            "Content:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}"
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["choices"][0]["message"]["content"].strip()


def main():
    st.set_page_config(layout="wide")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load agent (cached for performance)
    @st.cache_resource
    def load_rag_agent():
        faiss_idx, texts, fnames, whoosh_ix = load_indexes()
        return RAGAgent(faiss_idx, texts, fnames, whoosh_ix)

    agent = load_rag_agent()

    st.title("TrueHorizon AI Chatbot")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    query = st.chat_input("Ask a question about TrueHorizon AI")
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
            
        # Get assistant response
        with st.spinner("Thinking..."):
            answer = agent.run(query)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)

if __name__ == "__main__":
    main()
