import os
import hashlib
import json
import faiss
from pathlib import Path
import re
import time
from typing import List, Dict, Any

from llama_index.core import (
    VectorStoreIndex, 
    SummaryIndex, 
    StorageContext, 
    Document, 
    Settings,
    QueryBundle,
    load_index_from_storage
)
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
import shutil


# NEW: Import the refactored PDFProcessor
from processor.pdf_processor import PDFProcessor

class AgentRateLimitError(Exception):
    """Custom exception containing the wait time extracted from an API rate limit error."""
    def __init__(self, wait_time: float, message: str):
        self.wait_time = wait_time
        super().__init__(message)

class LlamaPDFAgent:

    def __init__(self, api_key: str = None, model: str = None):
        # 1. Initialize Settings with Groq and FastEmbed
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

        
        Settings.llm = Groq(
            model=self.model, 
            api_key=self.api_key,
            streaming=True # Global streaming support
        )
        Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

        
        # 2. Use the specialized PDFProcessor
        self.pdf_processor = PDFProcessor()
        
        self.vector_index = None
        self.summary_index = None
        self.recursive_query_engine = None
        self.is_loaded = False
        self.cache_dir = "./.llama_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.tables = [] # Store extracted DataFrames
        self.registry_path = os.path.join(self.cache_dir, "registry.json")
        self._init_registry()



    def ingest_pdf(self, pdf_file):
        """
        Ingests a PDF using Persistence: Loads from disk if already indexed.
        """
        file_hash = self.pdf_processor.get_pdf_hash(pdf_file)
        self.current_hash = file_hash
        cache_path = Path(self.cache_dir) / file_hash
        doc_cache_path = os.path.join(self.cache_dir, file_hash)
        
        # 1. Check if already indexed
        if os.path.exists(os.path.join(doc_cache_path, "default_vector_store.json")):
            storage_context = StorageContext.from_defaults(
                persist_dir=doc_cache_path,
                vector_store=FaissVectorStore.from_persist_dir(doc_cache_path)
            )
            self.vector_index = load_index_from_storage(storage_context)
            
            # NEW: Check for persistent JSON tables and Markdown (Eliminates redundant heavy parsing)
            tables_cache = cache_path / "tables.json"
            
            if tables_cache.exists():
                try:
                    with open(tables_cache, "r", encoding="utf-8") as f:
                        raw_tables = json.load(f)
                    self.tables = []
                    for rt in raw_tables:
                        self.tables.append({
                            "id": rt["id"],
                            "label": rt["label"],
                            "df": pd.DataFrame(rt["data"])
                        })
                except Exception as e:
                    self.tables = []
            
            # Re-load metadata (Docling)
            result = self.pdf_processor.load_docling_documents(pdf_file, cache_path=cache_path)
            documents = result["documents"]
            if not self.tables: self.tables = result["tables"]
            self.summary_index = SummaryIndex.from_documents(documents)
            
            # Rebuild Retriever/Engine
            nodes = list(self.vector_index.docstore.docs.values())
            self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
            vector_retriever = self.vector_index.as_retriever(similarity_top_k=5)
            self.recursive_retriever = RecursiveRetriever(
                "vector",
                retriever_dict={"vector": vector_retriever},
                node_dict={node.node_id: node for node in nodes}
            )
            self.recursive_query_engine = RetrieverQueryEngine.from_args(
                self.recursive_retriever,
                node_postprocessors=[LLMRerank(top_n=3)],
                streaming=True
            )
            
            self.is_loaded = True
            self._save_to_registry(file_hash, pdf_file.name)
            return f"Loaded '{pdf_file.name}' from library storage."

        # 2. Fresh Ingest (Load and parse)
        
        # 1. Load Documents with rich metadata via Docling JSON
        result = self.pdf_processor.load_docling_documents(pdf_file, cache_path=cache_path)
        documents = result["documents"]
        self.tables = result["tables"]
        
        # 2. Advanced Node Parsing (Captures page numbers and layout)
        node_parser = DoclingNodeParser()
        nodes = node_parser.get_nodes_from_documents(documents)

        
        # 3. Vector Index with FAISS
        d = 384 # BGE-small-en-v1.5 dimension
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context.docstore.add_documents(nodes)
        
        self.vector_index = VectorStoreIndex(
            nodes, 
            storage_context=storage_context
        )
        
        # Persist to disk
        self.vector_index.storage_context.persist(persist_dir=doc_cache_path)
        
        # 4. BM25 Retriever for Hybrid Search
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, 
            similarity_top_k=5
        )
        
        # 5. Recursive Retriever for Context Depth
        vector_retriever = self.vector_index.as_retriever(similarity_top_k=5)
        self.recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            node_dict={node.node_id: node for node in list(nodes)},
            verbose=True
        )
        
        # 6. Summary Index for global overview
        self.summary_index = SummaryIndex.from_documents(documents)
        
        # Setup the main recursive query engine
        self.recursive_query_engine = RetrieverQueryEngine.from_args(
            self.recursive_retriever,
            node_postprocessors=[LLMRerank(top_n=3)],
            streaming=True # Enable at engine level
        )
        
        self.is_loaded = True
        self._save_to_registry(file_hash, pdf_file.name)
        return f"Successfully indexed '{pdf_file.name}' and saved to library."


    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Returns answer and source citations including page numbers.
        """
        if not self.is_loaded: return {"answer": "No document loaded.", "sources": []}
        
        try:
            response = self.recursive_query_engine.query(question)
        except Exception as e:
            # Check for RateLimit (429) message: "Please try again in X.XXXs"
            error_str = str(e)
            match = re.search(r"Please try again in (\d+\.\d+)s", error_str)
            if match:
                wait_time = float(match.group(1))
                raise AgentRateLimitError(wait_time, error_str)
            raise e
        
        sources = []
        for node in response.source_nodes:
            # metadata contains 'doc_items' which has 'prov' with 'page_no'
            page_no = node.metadata.get("page_label") or node.metadata.get("page_no")
            
            if not page_no and "doc_items" in node.metadata:
                try:
                    doc_items = node.metadata["doc_items"]
                    if doc_items and "prov" in doc_items[0] and doc_items[0]["prov"]:
                        page_no = doc_items[0]["prov"][0].get("page_no")
                except (KeyError, IndexError, TypeError):
                    pass
            
            sources.append({
                "text": node.get_content()[:250] + "...", # Snippet for UI
                "page": page_no
            })

            
        return {
            "answer": str(response), # Full text for batch processing (SWOT, Insights)
            "answer_gen": response.response_gen, # Generator for streaming (Chat)
            "sources": sources
        }



    def get_kpi_viz_data(self):
        """
        Processes existing KPI text and extracts numerical pairs for charting.
        """
        kpi_text = self.get_deep_insights().get("key_metrics", "")
        if not kpi_text:
            return None
            
        prompt = f"""
        Extract key numerical metrics from the following text for visualization.
        Format as a JSON list of objects with 'label' and 'value'.
        Include only numerical values. If a value is a percentage, convert 10% to 10.
        
        Text: {kpi_text}
        """
        
        try:
            response = self.llm.complete(prompt)
            raw_json = str(response)
            if "```json" in raw_json:
                raw_json = raw_json.split("```json")[1].split("```")[0].strip()
            return json.loads(raw_json)
        except Exception:
            return None


    def summarize_document(self):
        if not self.is_loaded: return "No document loaded."
        query_engine = self.summary_index.as_query_engine(
            response_mode="tree_summarize",
            streaming=True
        )
        response = query_engine.query("Provide a comprehensive executive summary of this document.")
        return response


    def get_deep_insights(self) -> Dict[str, str]:
        """
        Performs a multi-stage analysis to extract strategic depth.
        """
        if not self.is_loaded: return {}
        
        prompts = {
            "strategic_vision": "What is the primary strategic vision or long-term objective described in this document?",
            "key_metrics": "Extract the top 5 most critical numerical KPIs or financial metrics mentioned. Format as a list.",
            "risks_and_challenges": "Identify the most significant risks, headwinds, or challenges mentioned for the business.",
            "swot_analysis": "Based on the content, provide a concise SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) in valid JSON format with keys 'S', 'W', 'O', 'T'."
        }
        
        insights = {}
        for key, query in prompts.items():
            result = self.answer_question(query)
            insights[key] = result.get("answer_text") or result.get("answer", "")
            
        return insights


    def _init_registry(self):
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, "w") as f:
                json.dump({}, f)

    def _get_registry(self) -> Dict[str, str]:
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_to_registry(self, file_hash: str, filename: str):
        registry = self._get_registry()
        registry[file_hash] = filename
        with open(self.registry_path, "w") as f:
            json.dump(registry, f)

    def get_library(self) -> List[Dict[str, str]]:
        registry = self._get_registry()
        return [{"hash": h, "filename": f} for h, f in registry.items()]

    def delete_document(self, file_hash: str):
        registry = self._get_registry()
        if file_hash in registry:
            doc_path = os.path.join(self.cache_dir, file_hash)
            if os.path.exists(doc_path):
                shutil.rmtree(doc_path)
            del registry[file_hash]
            with open(self.registry_path, "w") as f:
                json.dump(registry, f)
            if self.is_loaded and getattr(self, "current_hash", None) == file_hash:
                self.is_loaded = False
            return True
        return False



