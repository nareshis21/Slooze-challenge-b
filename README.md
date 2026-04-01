# Challenge B: Naresh AI DocuPulse - Submission

DocuPulse is an advanced, modular Retrieval-Augmented Generation (RAG) system for strategic document analysis, summarization, and query processing. This implementation uses high-fidelity PDF parsing via Docling and high-performance inference through the Groq Llama-3-70b or Llama-4-Scout models.

## Core Features and Enhancements

*   **Persistent Document Library**: Documents are indexed once and permanently cached on disk in the .llama_cache directory. Users can switch between previously analyzed PDFs instantly without re-ingesting content.
*   **Structured Parsing (Docling)**: Leverages the Docling framework for capturing complex document hierarchies, page metadata, and reconstructing high-precision tables from technical reports.
*   **Hybrid Retrieval Pipeline**: 
    *   Semantic search via FAISS-based vector indices.
    *   Keyword search via BM25 retrieval for specific term lookups.
    *   LLM-based re-ranking to isolate the top three most relevant contexts.
*   **Strategic Deep Intelligence**: Specialized mode that uses multi-stage recursive retrieval to perform SWOT analyses and extract strategic KPIs from financial or executive documents.
*   **API Resilience (Exact Backoff)**: Handles Groq API rate limits with an integrated persistence layer, including real-time UI countdowns to ensure request fulfillment.
*   **Modular Architecture**: The codebase is organized into domain-specific packages (agent, processor, ingestion, scripts) to ensure maintainability and professional standards.

## Prerequisites

1.  Python 3.10 or higher.
2.  Groq API Key: Available at console.groq.com.
3.  Environment Variables: GROQ_API_KEY and GROQ_MODEL.

## Installation and Setup

### 1. Clone the Project
Navigate to the DocuPulse directory:
```bash
cd challenge-b/DocuPulse
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a .env file in the DocuPulse folder:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

## How to Use the Agent

### Main Application (Streamlit)
To launch the primary interface for PDF uploads, library management, and document chat:
```bash
streamlit run app.py
```

### Testing and Diagnostics
For automated verification of the RAG pipeline and citation accuracy:
```bash
python scripts/test_agent.py
python scripts/verify_cite.py
```

## Modular Structure Overview

*   **app.py**: Primary Streamlit entry point.
*   **agent/**: Core RAG engine and API clients.
*   **processor/**: Document extraction and table cleaning.
*   **ingestion/**: FAISS indexing and persistent caching logic.
*   **scripts/**: Command-line utilities for end-to-end testing.

## Deployment via Docker

The project includes a Dockerfile optimized for high-performance PDF processing:
1.  Build the image: `docker build -t docupulse-agent .`
2.  Run the container on port 7860: `docker run -p 7860:7860 --env-file .env docupulse-agent`

Author: Naresh Kumar Lahajal
Version: 2.2.0 (Modular Edition)
