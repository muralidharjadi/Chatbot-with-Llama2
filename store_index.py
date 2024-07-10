from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
import os

load_dotenv()

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#db = Chroma.from_documents(text_chunks, embeddings)

persist_directory = "data_base/"
vectordb = Chroma.from_documents(
    documents=text_chunks, embedding=embeddings, persist_directory=persist_directory
)
