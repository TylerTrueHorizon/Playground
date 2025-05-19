"""Command line chat with TrueHorizon AI knowledge base using Agno."""

import pickle
import openai
import numpy as np
import faiss
from agno import Agent

INDEX_PATH = "faiss_index.pkl"


def embed_text(text: str) -> list:
    """Return an embedding vector for the given text."""
    result = openai.Embedding.create(model="text-embedding-ada-002", input=[text])
    return result["data"][0]["embedding"]


def load_index():
    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["texts"]


class RAGAgent(Agent):
    """Simple retrieval-augmented agent."""

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
    print("Ask me anything about TrueHorizon AI (Ctrl+C to exit)")
    while True:
        try:
            question = input("Question: ")
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        answer = agent.run(question)
        print("Answer:", answer)


if __name__ == "__main__":
    main()
