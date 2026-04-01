import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from dotenv import load_dotenv
from agent.agent import LlamaPDFAgent

load_dotenv()

def verify_citations():
    agent = LlamaPDFAgent()
    # Updated path to root
    pdf_path = os.path.join(root_dir, "nvidia_q4_fy24.pdf")
    
    with open(pdf_path, "rb") as f:
        class MockFile:
            def __init__(self, file, name):
                self.file = file
                self.name = name
            def read(self): return self.file.read()
            def seek(self, pos): self.file.seek(pos)
            def tell(self): return self.file.tell()
        
        mock_file = MockFile(f, pdf_path)
        agent.ingest_pdf(mock_file)
        
        q = "What was the revenue for Data Center in Q4?"
        result = agent.answer_question(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print("\nSOURCES:")
        for s in result['sources']:
            print(f"- Page {s['page']}: {s['text'][:50]}...")

if __name__ == "__main__":
    verify_citations()
