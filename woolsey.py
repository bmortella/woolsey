"""Ask a question to the PDFs."""
import faiss
from langchain.llms import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import argparse

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='Ask a question to the documents.')
parser.add_argument('question', type=str, help='The question to ask the documents.')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0,
                                                       max_tokens=512), vectorstore=store)
result = chain({"question": args.question})

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
