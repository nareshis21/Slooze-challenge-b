from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
import os
from pathlib import Path
import json

def inspect_nodes():
    pdf_path = "nvidia_q4_fy24.pdf"
    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    documents = reader.load_data(file_path=Path(pdf_path))
    
    parser = DoclingNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    
    if nodes:
        # Find a node that is likely to have a page number (not just a title)
        for node in nodes[5:15]: 
            metadata = node.metadata
            print("--- METADATA START ---")
            print(json.dumps(metadata, indent=2))
            print("--- METADATA END ---")
    else:
        print("No nodes created.")

if __name__ == "__main__":
    inspect_nodes()
