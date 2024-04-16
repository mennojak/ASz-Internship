import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata

class ChatBot_Harcoded():
    load_dotenv()

    pdf_path = './laserbehandeling-bij-nastaar.pdf'
    with open(pdf_path, 'rb') as f:
        pdf_reader = PdfReader(f)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Normalize whitespace
    text = " ".join(text.replace(u"\xa0", " ").strip().split())

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, 
        max_new_tokens=512,
        top_k=5,
        top_p=0.95,
        huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )

    template = """
    Jij bent een assistent van het Albert Schweitzer ziekenhuis. Patiënten zullen vragen aan jou stellen over hun medische situaties. Geef antwoord aan de hand van de gegeven context.
    Als het antwoord niet in de context staat, geef dan aan dat je het niet weet en biedt de patiënt aan om contact op te nemen met het ziekenhuis in dat geval. 
    Geef antwoord in een toegankelijk formaat van taalniveau B1.

    Context: {context}
    Vraag: {question}
    Antwoord: 

    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def ask_question(self, question):
        # Gebruik de Mistral LLM om een antwoord te genereren op basis van de tekst van de documenten
        # als context.
        context = self.text # Gebruik de volledige tekst van het document als context
        prompt_input = self.prompt.format(context=context, question=question)
        result = self.llm.invoke(prompt_input)
        answer_start = result.rfind("Antwoord:") + len("Antwoord:")
        answer = result[answer_start:].strip()
        return answer

# Code below is to test the chatbot without streamlit
# bot = ChatBot_Harcoded()
# input = input("Stel me een vraag: ")
# result = bot.ask_question(input)
# print(result)
