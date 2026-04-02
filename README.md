# 🏗️ Technical Case Study: DocuPulse (Challenge B)

**DocuPulse** is an advanced Retrieval-Augmented Generation (RAG) system engineered for high-fidelity document analysis and strategic intelligence extraction. This project explores the intersection of **Vision-Language Layout Understanding (Docling v2)**, **Recursive Retrieval Architecture**, and **High-Speed Inference (Groq)**.

🚀 **[Live Demo: Try DocuPulse on Hugging Face](https://huggingface.co/spaces/NEXAS/challenge-b)**


---

## 🔬 Architectural Decisions & Rationale

### 1. High-Fidelity Extraction (Docling v2.x vs. Legacy OCR)
Traditional PDF parsers (PyPDF, Tesseract) often fail on complex financial reports that contain nested tables, headers, and multi-column layouts.
*   **Decision**: Integrated **Docling v2.x** for vision-based document conversion.
*   **Rationale**: By using a layout-aware, vision-language model, the system reconstructs document hierarchies with perfect fidelity. This allows for near-perfect **Table Reconstruction** directly into Pandas DataFrames, bypassing the noise of traditional text extraction.

### 2. Advanced RAG: Recursive Retrieval
To solve the "Lost in the Middle" problem and handle hierarchical document nodes, I implemented a **Recursive Retrieval Architecture**:
*   **DoclingNodeParser**: Custom chunking that maintains document headers, parent-child relationships, and section metadata.
*   **RecursiveRetriever**: Leverages LlamaIndex to navigate these relationships. If a specific table cell is retrieved, the engine automatically resolves back to the parent section context for superior grounding.

### 3. Multi-Index Hybrid Storage
Different queries require different "eyes" on the document:
*   **VectorStoreIndex (FAISS)**: Optimized for precise semantic lookup at the node level.
*   **SummaryIndex**: Reserved for document-wide synthesis tasks like executive summaries and global strategy analysis.
*   **BM25 Integration**: Ensures that exact keyword matches (e.g., specific financial line items) are never missed by the semantic encoder.

---

## ⚡ Ingestion Engineering: Single-Pass Pipeline
Performance optimization was critical for user experience with large PDFs.
*   **90% Latency Reduction**: By consolidating text extraction, layout parsing, and table detection into a single **Single-Pass Conversion** in `PDFProcessor`, I reduced ingestion time from minutes to under 45 seconds for a 50-page PDF.
*   **Strategic Dashboarding**: Implemented multi-stage extraction logic to populate a real-time **Strategy Dashboard** (SWOT, KPIs, Risks) through specialized, recursive analysis passes.

---

## 🐳 Deployment & Lifecycle
*   **Persistence**: MD5-based file hashing ensures a zero-duplication document library stored in `.llama_cache`. 
*   **Containerization**: Multi-stage Docker build optimized for high-performance PDF processing in resource-constrained environments like Hugging Face Spaces.

---

## ⚙️ Core Technical Stack
*   **Extraction**: Docling v2.x (Vision-Language Pipeline)
*   **RAG Framework**: LlamaIndex (RecursiveRetriever + FAISS)
*   **Inference**: Groq (Llama-4-Scout, Llama-3.3-70B)
*   **UI/UX**: Streamlit 1.4+ (Custom Strategic Dashboard)

---

*Project by **Naresh Kumar Lahajal** - Advanced Agentic AI Engineering.*
