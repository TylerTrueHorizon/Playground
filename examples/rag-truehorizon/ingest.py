"""Crawl TrueHorizon AI pages and build a FAISS index."""

import os
import pickle
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import faiss
import numpy as np
import openai

BASE_URL = "https://truehorizon.ai"

DATA_DIR = "data"
INDEX_PATH = "faiss_index.pkl"


def fetch_page(url: str) -> str:
    """Fetch a single page and return its text."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text()


def crawl_site(start_url: str) -> None:
    """Crawl all pages under the given domain and save them to ``DATA_DIR``."""
    seen = set()
    queue = [start_url]
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            text = fetch_page(url)
        except Exception:
            continue
        path = urlparse(url).path.strip("/") or "index"
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(os.path.join(DATA_DIR, f"{path}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup.find_all("a", href=True):
            link = tag["href"]
            if link.startswith("http"):
                full = link
            else:
                full = urljoin(start_url, link)
            if full.startswith(start_url) and full not in seen:
                queue.append(full)


def embed_text(text: str) -> list:
    """Return an embedding vector for the given text."""
    result = openai.Embedding.create(model="text-embedding-ada-002", input=[text])
    return result["data"][0]["embedding"]


def build_index() -> None:
    """Create a FAISS index from crawled pages."""
    texts = []
    for fname in os.listdir(DATA_DIR):
        with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
            texts.append(f.read())

    vectors = [embed_text(t) for t in texts]
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"index": index, "texts": texts}, f)


if __name__ == "__main__":
    crawl_site(BASE_URL)
    build_index()
    print("Index built at", INDEX_PATH)
