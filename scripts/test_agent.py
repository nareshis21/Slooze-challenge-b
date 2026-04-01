import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from dotenv import load_dotenv
from agent.agent import LlamaPDFAgent
import io

load_dotenv()

def test_agent():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found in environment.")
        return

    agent = LlamaPDFAgent(api_key=api_key)
    
    # Use the downloaded NVIDIA PDF - updated path
    pdf_path = os.path.join(root_dir, "nvidia_q4_fy24.pdf")
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return

    with open(pdf_path, "rb") as f:
        # Mocking a streamlit-like upload object
        class MockFile:
            def __init__(self, file, name):
                self.file = file
                self.name = name
            def read(self):
                return self.file.read()
            def seek(self, pos):
                self.file.seek(pos)
            def tell(self):
                return self.file.tell()
        
        mock_file = MockFile(f, pdf_path)
        print("Ingesting PDF...")
        msg = agent.ingest_pdf(mock_file)
        print(msg)

        print("\n--- Testing Q&A ---")
        q = "What was the total revenue for FY24?"
        result = agent.answer_question(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print("\nSources:")
        for src in result['sources']:
            print(f"- [Page {src['page']}] {src['text'][:100]}...")

        print("\n--- Testing Deep Insights ---")
        insights = agent.get_deep_insights()

        for key, value in insights.items():
            print(f"\n[{key.upper()}]")
            print(value)

if __name__ == "__main__":
    test_agent()
