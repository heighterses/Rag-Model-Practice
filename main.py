from langchain_community.document_loaders import PyPDFLoader
import os
import openai
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

character_splitter = CharacterTextSplitter(chunk_size=200,
                                           chunk_overlap=0,
                                           separator='')


recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                                    chunk_overlap=0,
                                                    separators=["\n\n", "\n", " ", ""])

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

loader = PyPDFLoader("pdf_files/MachineLearning-Lecture01.pdf")
pages = loader.load()
print(len(pages))

page = pages[0]
content = page.page_content
print(content)

test = recursive_splitter.split_text(content)
print(test)