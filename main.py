from langchain_community.document_loaders import PyPDFLoader
import os
import openai
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai.api_key = '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'

loader = PyPDFLoader("pdf_files/MachineLearning-Lecture01.pdf")
pages = loader.load()
print(len(pages))

page = pages[3]
content = page.page_content[:500]
print(content)
