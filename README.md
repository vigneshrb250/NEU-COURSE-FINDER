# NEU Course Finder

NEU Course Finder is an end-to-end **Retrieval-Augmented Generation (RAG)** application that allows students to ask natural language questions about Northeastern University (NEU) courses. It scrapes course data, embeds it into a vector database, and uses a Large Language Model (LLM) to provide contextual answers with source references.

---

## Features

- Natural language queries about NEU courses
- Embeddings via `BAAI/bge-small-en-v1.5`
- Vector store using `ChromaDB`
- LLM-powered responses via `HuggingFaceH4/zephyr-7b-beta`
- Full-stack architecture using `Flask` + `llama-index`
- Render deployment with zipped vector DB auto-downloaded from Hugging Face

---

## RAG Pipeline Overview

The NEU Course Finder follows a classic **Retrieval-Augmented Generation (RAG)** architecture:

### Components:

- **Web Scraping**: Extract NEU course titles and descriptions from the university website.
- **Embedding**: Convert course descriptions into vector representations using `BAAI/bge-small-en-v1.5`.
- **Vector Store**: Store these embeddings in a persistent Chroma vector database.
- **Context Retrieval**: Given a user query, retrieve top-k similar vectors (course chunks) using semantic similarity.
- **LLM Response Generation**: Pass the retrieved context to a Hugging Face-hosted `zephyr-7b-beta` model to generate a relevant, concise answer.

This architecture ensures factual responses grounded in the scraped course data while leveraging LLM fluency.


---

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vigneshrb250/NEU-COURSE-FINDER.git
cd NEU-COURSE-FINDER/course_finder_app
```

### 2. Set Environment Variables

Create a .env file inside course_finder_app/:
HUGGINGFACE_API_TOKEN=your_huggingface_token
FLASK_SECRET_KEY=your_flask_secret_key

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

### 5. Example Queries

- "Which course teaches data structures?"
- "What topics are covered in CS5800?"
- "Tell me about Machine Learning electives."

Each response includes relevant source chunks from the course database to ensure factuality.






