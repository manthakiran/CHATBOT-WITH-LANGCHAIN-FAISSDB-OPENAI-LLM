# PDF Question-Answering Chatbot — LangChain, FAISS & LLM

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a PDF document. It loads a PDF, splits it into chunks, stores embeddings in a **FAISS** vector database, retrieves the passages most relevant to your question, and asks an LLM to answer **only from those passages**.

The notebook ships configured for the *TCS Integrated Annual Report 2025-26* (a 360-page PDF), but it works with any text-based PDF.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-0467DF)
![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-FFD21E?logo=huggingface&logoColor=black)

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Large language models can answer questions fluently, but they can also make things up and know nothing about your private or very recent documents. RAG fixes this by giving the model the relevant source text at question time.

This project demonstrates the full RAG loop in one short notebook:

1. **Load** a PDF into page-level documents.
2. **Split** the pages into overlapping chunks.
3. **Embed** each chunk with a Sentence-Transformers model and store the vectors in FAISS.
4. **Retrieve** the 3 chunks most similar to the user's question.
5. **Prompt** an LLM with those chunks and strict instructions to answer only from them and cite the supporting sentence.
6. **Print** the grounded answer.

If the answer isn't in the retrieved text, the prompt instructs the model to reply with a fixed fallback message instead of guessing.

## How It Works

```
tcs.pdf
   │
   ▼
PyPDFLoader ──► 360 page documents (with page metadata)
   │
   ▼
RecursiveCharacterTextSplitter (2500 chars, 500 overlap) ──► 640 chunks
   │
   ▼
HuggingFaceEmbeddings (all-mpnet-base-v2) ──► FAISS vector database
   │
Question ──► similarity search (top 3 chunks)
   │
   ▼
Grounded prompt (context + question + rules)
   │
   ▼
LLM via OpenAI-compatible API ──► Answer
```

The notebook steps:

| Step | Description |
| --- | --- |
| 1. Load PDF | `PyPDFLoader` reads `tcs.pdf` into a list of `Document` objects, one per page, each with metadata such as source and page number |
| 2. Split | `RecursiveCharacterTextSplitter` creates chunks of 2,500 characters with 500 characters of overlap so context isn't lost at boundaries |
| 3. Embed | `HuggingFaceEmbeddings` loads `sentence-transformers/all-mpnet-base-v2` |
| 4. Index | `FAISS.from_documents(chunks, model)` builds the vector database |
| 5. Prompt | Runs a similarity search (`k=3`), joins the retrieved text and builds the prompt |
| 6. Answer | Sends the prompt to the LLM and prints the response |

## Features

- **Grounded answers:** the prompt forbids using prior knowledge and requires the model to cite the supporting sentence
- **Honest fallback:** when the context lacks the answer, the model must reply exactly *"I could not find the answer in the provided context."*
- **Local embeddings:** vector search runs on your machine; only the final prompt is sent to the LLM
- **Page-aware chunks:** each chunk keeps its source file and page metadata, making it easy to extend with page citations
- **Provider-flexible:** uses the `openai` SDK, so any OpenAI-compatible endpoint can be used by changing the base URL, key and model name
- **Minimal and readable:** about a dozen short cells, ideal for learning or as a starting point

## Tech Stack

| Component | Technology |
| --- | --- |
| Language | Python 3 (Jupyter Notebook) |
| PDF loading | LangChain `PyPDFLoader` (backed by `pypdf`) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` via `langchain-huggingface` |
| Vector store | FAISS via `langchain-community` |
| LLM | Gemini through its OpenAI-compatible endpoint, called with the `openai` SDK |

## Project Structure

```
rag-chatbot/
├── Untitled.ipynb      # The notebook (consider renaming, e.g. rag_chatbot.ipynb)
├── tcs.pdf             # Source document (you provide this)
└── README.md
```

The notebook currently keeps the FAISS index in memory only, so no index files are written to disk.

## Installation

### Prerequisites

- **Python 3.10 or newer** (developed on Python 3.13)
- **Jupyter Notebook** or JupyterLab (or VS Code with the Jupyter extension)
- An **API key** for an OpenAI-compatible LLM endpoint (the notebook uses Google's Gemini endpoint)
- A **text-based PDF** to query (scanned PDFs need OCR first)

### 1. Get the code

```bash
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```bash
pip install langchain-community langchain-text-splitters langchain-huggingface \
            sentence-transformers faiss-cpu pypdf openai jupyter
```

The first run downloads the embedding model (roughly 400 MB) from the Hugging Face Hub. Setting an `HF_TOKEN` environment variable avoids rate-limit warnings but is optional.

### 4. Add your PDF

Place your PDF in the project folder and name it `tcs.pdf`, or change the filename in the loader cell:

```python
loader = PyPDFLoader('your_document.pdf')
```

## Configuration

| Setting | Where | Default | Description |
| --- | --- | --- | --- |
| PDF path | Load PDF cell | `tcs.pdf` | Document to index |
| `chunk_size` | Splitting cell | `2500` | Characters per chunk |
| `chunk_overlap` | Splitting cell | `500` | Characters shared between neighbouring chunks |
| Embedding model | Embedding cell | `sentence-transformers/all-mpnet-base-v2` | Any Sentence-Transformers model |
| `k` | Prompt cell | `3` | Number of chunks retrieved per question |
| `base_url` | LLM client cell | Gemini OpenAI-compatible URL | Change to use another provider |
| `model` | LLM call cell | as set in the notebook | Any model your endpoint supports |

### API key

**Do not paste a real API key into the notebook.** Load it from an environment variable instead:

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key-here"

# Windows (PowerShell)
$env:GEMINI_API_KEY = "your-key-here"
```

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
```

## Usage

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```
   Open the notebook and make sure your API key is available (see [Configuration](#api-key)).
2. **Run the cells in order** (`Kernel → Restart & Run All` works too). Building the index over a 360-page report takes a little while on CPU because 640 chunks must be embedded.
3. **Ask a question** by editing this cell and re-running everything from there:
   ```python
   question = "What was the net profit reported?"
   ```
4. **Read the answer** printed by the final cell. It will either answer using the retrieved text with a supporting sentence, or return the fallback message if the PDF excerpts don't contain the answer.

You only need to rebuild the index when you change the PDF or chunking settings. To ask a new question, re-run just the cells from the `question` cell onward.

### Asking several questions

To avoid re-running cells by hand, wrap the retrieval and answer steps in a function:

```python
def ask(question, k=3):
    docs = vector_db.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"""
    Answer only from the <context> below. Do not use any prior knowledge.
    If the answer is absent, reply exactly: 'I could not find the answer in the provided context.'
    Do not guess. Cite the supporting sentence.

    Context: {context}.
    Question: {question}.
    """
    response = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

print(ask("What was the net profit reported?"))
print(ask("How many employees does the company have?"))
```

## Dependencies

### Python packages

| Package | Purpose |
| --- | --- |
| `langchain-community` | `PyPDFLoader` and the FAISS vector store |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| `langchain-huggingface` | `HuggingFaceEmbeddings` wrapper |
| `sentence-transformers` | Embedding model backend |
| `faiss-cpu` | Vector similarity search |
| `pypdf` | PDF parsing used by `PyPDFLoader` |
| `openai` | Client for the OpenAI-compatible LLM endpoint |
| `jupyter` | Run the notebook |

### System requirements

- Internet access for the first-time model download and for LLM API calls
- No GPU required (embeddings run on CPU)
- Enough RAM to hold the embeddings and model (a few GB)

> `langchain-community` has announced it is being sunset in favour of standalone integration packages. The notebook still works, but you may want to migrate the PDF loader and FAISS store to their dedicated packages in future.

## Limitations and Future Improvements

- **Not yet interactive:** despite the "chatbot" name, the notebook answers one question at a time and has no conversation memory or follow-up handling.
- **Index not saved:** the FAISS index is rebuilt on every session. Use `vector_db.save_local("faiss_index")` and `FAISS.load_local(...)` to persist it.
- **Text extraction quality:** designed PDFs such as annual reports can produce odd extraction artifacts (for example stray glyph names like `/A.calt` in place of ligatures). These can slightly reduce retrieval quality.
- **No page citations in output:** page numbers are stored in metadata but not yet shown alongside answers.
- **Fixed retrieval:** only the top 3 chunks are used, so questions spanning many sections may be answered incompletely. Raising `k` or using MMR search can help.
- **Tables and figures:** numeric tables may lose their structure when extracted as plain text, which matters for financial questions. Always verify important figures against the source document.

Ideas for future work:

- [ ] Add a `requirements.txt` and `.gitignore`
- [ ] Persist and reload the FAISS index
- [ ] Return page numbers with each answer
- [ ] Add conversation memory for follow-up questions
- [ ] Build a chat UI with Streamlit or Gradio
- [ ] Support multiple PDFs and other file types
- [ ] Add a reranker and an evaluation set of question/answer pairs

## Contributing

Contributions are welcome, including bug fixes, new features, documentation and ideas.

### Getting started

1. **Fork** the repository and clone your fork.
2. Follow the [Installation](#installation) steps.
3. Create a branch for your work:
   ```bash
   git checkout -b feature/short-description
   ```
   Use prefixes such as `feature/`, `fix/`, `docs/` or `refactor/`.

### Guidelines

- **Never commit secrets.** API keys, tokens and `.env` files must stay out of the repository. Load credentials from environment variables.
- **Mind licensing of documents.** Don't commit PDFs you don't have the right to redistribute; add them to `.gitignore` and document where to download them instead.
- **Clear notebook outputs before committing** to keep diffs small and avoid leaking local paths or data. [`nbstripout`](https://github.com/kynan/nbstripout) can do this automatically:
  ```bash
  pip install nbstripout
  nbstripout --install
  ```
- **Keep the notebook runnable top to bottom** (`Kernel → Restart & Run All`) before opening a pull request.
- **Style:** follow [PEP 8](https://peps.python.org/pep-0008/), use descriptive names, and add short comments to new code.
- **Keep changes focused:** one feature or fix per pull request.
- **Document behaviour changes** in this README, especially new settings or dependencies.

### Commit messages

Use short, imperative messages, ideally following [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: persist FAISS index to disk
fix: handle empty retrieval results
docs: add example questions
```

### Submitting a pull request

1. Push your branch and open a pull request against `main`.
2. Describe **what** changed and **why**, and how you tested it.
3. Include example questions and answers where relevant.
4. Respond to review feedback; a maintainer will merge once approved.

### Reporting bugs and requesting features

Open a GitHub issue with the steps to reproduce, the full error message, your Python version and operating system, and what you expected to happen.

## License

No license has been specified yet. Add a `LICENSE` file (for example MIT or Apache-2.0) to let others know how they may use and contribute to this project, then update this section.

---

<p align="center">Built with LangChain, FAISS and Sentence-Transformers</p>
