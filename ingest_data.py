from pathlib import Path
from tqdm import tqdm

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from pdf2image import convert_from_path
import pytesseract

import faiss
import pickle

# TODO: Maybe split the text while processing the pages so we can hold the page number in the metadata
def read_pdf(pdf) -> str:
    '''Read a PDF file using OCR.'''
    pages = convert_from_path(pdf)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text
    

ps = list(Path("./docs").glob("*.pdf"))
data = []
sources = []
for p in tqdm(ps):
    data.append(read_pdf(p))
    sources.append(p)

# Split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))


# Create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, HuggingFaceEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
