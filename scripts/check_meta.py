from llama_index.readers.docling import DoclingReader
import os
from pathlib import Path

def check_metadata():
    pdf_path = "nvidia_q4_fy24.pdf"
    if not os.path.exists(pdf_path):
        print("PDF not found.")
        return
        
    reader = DoclingReader()
    documents = reader.load_data(file_path=Path(pdf_path))
    
    print(f"Loaded {len(documents)} documents.")
    for i, doc in enumerate(documents[:2]): # Just check first two
        print(f"Doc {i} Metadata: {doc.metadata}")
        # print(f"Doc {i} Text Preview: {doc.text[:200]}...")

if __name__ == "__main__":
    check_metadata()
