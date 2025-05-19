"""Streamlit UI for chatting with the TrueHorizon AI knowledge base."""

import pickle
import numpy as np
import faiss
import openai
import streamlit as st
from agno import Agent

INDEX_PATH = "faiss_index.pkl"


def embed_text(text: str) -> list:
    result = openai.Embedding.create(model="text-embedding-ada-002", input=[text])
    return result["data"][0]["embedding"]


def load_index():
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["texts"]


class RAGAgent(Agent):
    def __init__(self, index, texts):
        super().__init__()
        self.index = index
        self.texts = texts

    def run(self, query: str) -> str:
        vector = np.array([embed_text(query)], dtype="float32")
        _, idxs = self.index.search(vector, k=3)
        context = "\n".join(self.texts[i] for i in idxs[0])
        prompt = (
            "Answer the user question using the following website content:\n" + context
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt + "\n\n" + query}],
        )
        return response["choices"][0]["message"]["content"].strip()


def main():
    index, texts = load_index()
    agent = RAGAgent(index, texts)

    st.title("TrueHorizon AI Chatbot")
    query = st.text_input("Ask a question about TrueHorizon AI")
    if query:
        answer = agent.run(query)
        st.write(answer)


if __name__ == "__main__":
    main()
