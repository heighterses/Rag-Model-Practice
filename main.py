import rm
from chromadb.cli.cli import docs  

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

openai_api_key = os.environ.get("OPENAI_API_KEY")

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

splitter = text_splitter.split_documents(documents)
print(len(splitter))

# Add Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

s1 = "I love dogs"
s2 = "I love cats"
s3 = " My brother has two best friends"
s4 = "Horse is running faster than aunt"

embed1 = embedding.embed_query(s1)
embed2 = embedding.embed_query(s2)
embed3 = embedding.embed_query(s3)
embed4 = embedding.embed_query(s4)

comparison1 = np.dot(embed1, embed4)
comparison2 = np.dot(embed1, embed2)
comparison3 = np.dot(embed4, embed1)

print(comparison1, comparison3, comparison2)

# Add Vectors

from langchain.vectorstores import Chroma

persist_directory = 'docs/chroma/'
# !rm -rf ./docs/Chroma  # remove old database files if any

vector_db = Chroma.from_documents(
    documents=splitter,
    embedding=embedding,
    persist_directory=persist_directory

)

print(vector_db._collection.count())
