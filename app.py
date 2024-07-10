from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
import json
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          pipeline)

app = Flask(__name__)

embeddings = download_hugging_face_embeddings()
persist_directory = "data_base/"

db3 = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
query = "Key Sectors of the Indian Economy?"
docs = db3.similarity_search(query)
#print(docs[0].page_content)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}
config_data = json.load(open("config.json"))
HF_TOKEN = config_data["HF_TOKEN"]
model_name = "meta-llama/Meta-Llama-3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          token=HF_TOKEN)

tokenizer.pad_token = tokenizer.eos_token
llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    token=HF_TOKEN
)

