# TrueHorizon AI RAG Example

This folder contains a simple retrieval-augmented generation (RAG) chatbot setup for TrueHorizon AI. The dataset is built from pages scraped from [truehorizon.ai](https://truehorizon.ai). The crawler automatically discovers all pages under the site.

## Setup

1. Install Python packages:

```bash
pip install -r requirements.txt
```

2. Run `ingest.py` to crawl the website and build an index (this only needs to be done when the site content changes):

```bash
python ingest.py
```

3. Start the command line chat application:

```bash
python chat.py
```

4. Alternatively, launch a small web UI:

```bash
streamlit run app.py
```

The chatbot loads the pre-built index and answers questions about TrueHorizon AI without re-ingesting the site on each run.
