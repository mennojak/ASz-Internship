import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata

class ChromaWrapper:
    def __init__(self, chroma, embedding_model):
        self.chroma = chroma
        self.embedding_model = embedding_model

    def __call__(self, input_text):
        vector = self.embedding_model.encode(input_text).tolist()
        return vector

class ChatBot():
    load_dotenv()

    pdf_path = './laserbehandeling-bij-nastaar.pdf'
    with open(pdf_path, 'rb') as f:
        pdf_reader = PdfReader(f)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Normalize whitespace
    text = " ".join(text.replace(u"\xa0", " ").strip().split())

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=4)
    docs = text_splitter.split_text(text)

    documents = [Document(doc) for doc in docs]

    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma("langchain_store", embeddings)
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    vectorstore.add_documents(documents)

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

    vectorstore_wrapper = ChromaWrapper(vectorstore, embedding_model)

    rag_chain = (
        {"context": vectorstore_wrapper, "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser() 
    )

# Code below is to test the chatbot without streamlit
# bot = ChatBot()
# input = input("Stel me een vraag: ")
# result = bot.rag_chain.invoke(input)
# print(result)