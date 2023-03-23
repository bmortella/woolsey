from pathlib import Path
from tqdm import tqdm

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from pdf2image import convert_from_path
import pytesseract

import faiss
import pickle


data = list()
sources = list()
docs = list(Path("./docs").glob("*.pdf"))

# Iterate over all PDFs in the docs folder.
for pdf in tqdm(docs):
    # Convert the PDF to images.
    pages = convert_from_path(pdf)
    with open(f"./out/{pdf.stem}.txt", "w", encoding="UTF-8") as f:
        for i, page in enumerate(pages):
            # Read the text from the page using OCR.
            text = pytesseract.image_to_string(page)
            f.write(text)
            
            # Split text into smaller chunks. Needed due to the context limits of the LLMs.
            text_splitter = CharacterTextSplitter(chunk_size=800, separator="\n")
            splits = text_splitter.split_text(text)
            
            data.extend(splits)
            sources.extend([{"source": f"{pdf.stem} (p.{i+1})"}] * len(splits))

# Create a vector store from the documents and save it to disk.
store = FAISS.from_texts(data, HuggingFaceEmbeddings(), metadatas=sources)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
