from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
import os
from pathlib import Path

def inspect_nodes():
    pdf_path = "nvidia_q4_fy24.pdf"
    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    documents = reader.load_data(file_path=Path(pdf_path))
    
    parser = DoclingNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    
    if nodes:
        print(f"Node 0 Metadata: {nodes[0].metadata.keys()}")
        print(f"Node 0 Metadata Content: {nodes[0].metadata}")
    else:
        print("No nodes created.")

if __name__ == "__main__":
    inspect_nodes()
