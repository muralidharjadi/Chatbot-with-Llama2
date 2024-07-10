from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from dotenv import load_dotenv
from src.prompt import *
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

embeddings = download_hugging_face_embeddings()
persist_directory = "data_base/"

db3 = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
query = "Key Sectors of the Indian Economy?"
docs = db3.similarity_search(query)
#print(docs[0].page_content)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm = OpenAI(temperature=0.6)
response = llm.invoke("What is Capital of India?")
print(response)
