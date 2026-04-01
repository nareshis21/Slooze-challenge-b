import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GroqClient:
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        
        if not self.api_key:
            raise ValueError("Groq API Key not found. Please set GROQ_API_KEY in your .env file.")
        
        self.client = Groq(api_key=self.api_key)

    def get_completion(self, prompt: str, system_message: str = "You are a helpful AI assistant."):
        """
        Calls the Groq API to get a completion for the given prompt.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error calling Groq API: {e}"

    def get_json_completion(self, prompt: str, system_message: str = "You are a helpful AI assistant."):
        """
        Calls the Groq API with JSON mode enabled.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
                response_format={"type": "json_object"},
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"{{\"error\": \"{e}\"}}"

    def list_models(self):
        """
        Lists available models from Groq.
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

if __name__ == "__main__":
    # Test LLM client (requires API key)
    try:
        client = GroqClient()
        response = client.get_completion("Hello, how are you?")
        print(f"Groq Response: {response}")
    except ValueError as e:
        print(e)
