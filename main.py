from langchain_community.document_loaders import PyPDFLoader
import os
import openai
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

character_splitter = CharacterTextSplitter(chunk_size=200,
                                           chunk_overlap=0,
                                           separator='')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                               chunk_overlap=150)

# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=1000,
#     chunk_overlap=150,
#     length_function=len
# )

sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai.api_key = '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'

loaders = [PyPDFLoader("pdf_files/MachineLearning-Lecture01.pdf"),
           PyPDFLoader("pdf_files/MachineLearning-Lecture01.pdf")]

documents = []

for loader in loaders:
    documents.extend(loader.load())

# pages = documents
# print(len(pages))

# page = pages[0]
# content = page.page_content
# print(content)

test = text_splitter.split_documents(documents)
print(len(test))

# Add Embeddings