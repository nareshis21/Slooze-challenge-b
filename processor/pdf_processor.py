import hashlib
import json
import tempfile
import os
import io
from pathlib import Path
from typing import List, Dict
from llama_index.core import Document
import pandas as pd
# Advanced Docling Imports for Table Extraction
from llama_index.readers.docling import DoclingReader
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions, TableFormerMode
from docling.document_converter import DocumentConverter
from docling.document_converter import PdfFormatOption

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize advanced pipeline options for accurate table discovery
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        # NOTE: OCR is disabled by default for 10x speed boost on text-based PDFs.
        pipeline_options.do_ocr = False 
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def get_pdf_hash(self, pdf_file) -> str:
        """
        Generates an MD5 hash for the PDF file object to serve as a cache key.
        """
        pos = pdf_file.tell()
        pdf_file.seek(0)
        file_hash = hashlib.md5(pdf_file.read()).hexdigest()
        pdf_file.seek(pos)
        return file_hash

    def load_docling_documents(self, pdf_file, cache_path: Path = None) -> Dict:
        """
        Uses Docling for unified RAG and Table Extraction in a single pass.
        Returns a dict with 'documents' (LlamaIndex) and 'tables' (List of DataFrames).
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=os.getcwd()) as tmp:
            pdf_file.seek(0)
            tmp.write(pdf_file.read())
            tmp_path = Path(tmp.name)
            
        try:
            # 1. Single-Pass Conversion (Core Optimization - Truly One Pass)
            result = self.doc_converter.convert(tmp_path)
            doc = result.document # This is a DoclingDocument (v2)
            
            # 2. Extract LlamaIndex Documents from the result manually
            # This replaces DoclingReader and avoids double-conversion
            json_content = result.document.model_dump_json() 
            
            # We create a single LlamaIndex Document with the full JSON content.
            # Use our existing hash generator for consistency.
            pdf_file.seek(0)
            file_hash = self.get_pdf_hash(pdf_file)
            
            documents = [Document(
                text=json_content,
                metadata={
                    "filename": pdf_file.name,
                    "dl_doc_hash": file_hash
                }
            )]

            # 3. Extract structured tables (Uses Docling v2 high-speed export)
            tables = []
            for i, table in enumerate(doc.tables):
                try:
                    df = table.export_to_dataframe(doc=doc)
                    if not df.empty:
                        # Find page number for labeling
                        page_no = "?"
                        if table.prov and len(table.prov) > 0:
                            page_no = table.prov[0].page_no
                            
                        tables.append({
                            "id": i + 1,
                            "label": f"Table {i+1} (Page {page_no})",
                            "df": df
                        })
                except Exception as e:
                    print(f"Table Extraction Error [Table {i+1}]: {e}")

            # PERSIST: Save Markdown and Tables
            if cache_path:
                try:
                    cache_path.mkdir(parents=True, exist_ok=True)
                    
                    # Markdown Export (Instant from existing result)
                    md_content = result.document.export_to_markdown()
                    with open(cache_path / "content.md", "w", encoding="utf-8") as f:
                        f.write(md_content)
                    
                    # Store tables as JSON for persistence
                    if tables:
                        serialized_tables = [{
                            "id": t["id"],
                            "label": t["label"],
                            "data": t["df"].to_dict(orient="records")
                        } for t in tables]
                        with open(cache_path / "tables.json", "w", encoding="utf-8") as f:
                            json.dump(serialized_tables, f, indent=2)
                            
                except Exception as e:
                    print(f"Persistence Error: {e}")

            return {
                "documents": documents,
                "tables": tables
            }
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

if __name__ == "__main__":
    # Test
    pass

