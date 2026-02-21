# Ask-Your-Docs

A **Retrieval-Augmented Generation (RAG)** application built with **LangChain** and **OpenAI**. Turn your documents into a Q&A chatbot or use your data as the source for interactive AI applications.

---

## What it does

- Ask questions over a large set of documents and get answers grounded in your content.
- Build chatbots (e.g. customer support) that follow instructions and cite sources.

The app chunks your text, embeds it with OpenAI, stores it in [Chroma](https://www.trychroma.com/), then uses similarity search + an LLM to answer queries with source references.

---

## How it works

1. **Chunk** — Split documents into small text segments (e.g. 500–1000 characters with overlap). This keeps retrieval precise and reduces cost by sending only relevant parts to the LLM.
2. **Embed & store** — Turn chunks into vector embeddings and store them in a Chroma database. Embeddings capture meaning and similarity so we can find relevant passages for each query.
3. **Retrieve** — For each user question, run a similarity search and fetch the most relevant chunks.
4. **Generate** — The OpenAI model answers using only those chunks as context and returns the answer plus the source documents.

---

## Example output

After running a query, the script prints the prompt, the model’s answer, and the source files used.

---

## Setup

### 1. Get the project

Clone or download this repository, then open a terminal in the project directory.

### 2. OpenAI API key

- Create an API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
- In the project root, create a `.env` file and add:

  ```
  OPENAI_API_KEY="your_generated_secret_key"
  ```

- Ensure `.env` is in your `.gitignore` so the key is not committed.

### 3. (Optional) Use your own documents

- Put your `.md` files in the `data` directory (or another folder of your choice).
- In `create_database.py`, set `DATA_PATH` to that directory, e.g. `DATA_PATH = "data"` or `DATA_PATH = "data/your_folder"`.

---

## Run the application

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Build the vector database (run once, or when you change documents):**

```bash
python create_database.py
```

**Ask questions:**

```bash
python query_data.py "What operating systems does EC2 support"
```

The script prints the prompt, the model’s answer, and the source files used.
