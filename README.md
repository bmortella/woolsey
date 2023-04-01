# woolsey

Experimenting with langchain document QA with a set of PDFs.

Code adapted from https://github.com/hwchase17/notion-qa

## Usage

First install the requirements with `pip install -r requeriments.txt`. Also set your `OPENAI_API_KEY` environment variable or add `openai_api_key='your API key here'` to the code just after `temperature=0`.

1. Place your PDFs in the docs folder.
2. Run `python ingest_data.py`.
3. Run `python wolsey.py "Your question here"`

## Notes

- To minimize expenses, this code employs an embedding model from HuggingFace defined in the langchain code instead of OpenAI's ada embedding. Although this particular model is only compatible with English, the model available at [this link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) should support multiple languages.
- I also decreased the chunk size to 1000 from 1500 in the original repo as it made sense for my use case.
