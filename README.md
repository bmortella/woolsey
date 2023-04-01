# woolsey

Experimenting with langchain document QA with a set of PDFs.

Code adapted from https://github.com/hwchase17/notion-qa

## Usage

First install the requirements with `pip install -r requeriments.txt`

1. Place your PDFs in the docs folder.
2. Run `python ingest_data.py`.
3. Run `python wolsey.py "Your question here"`

## Notes

In order to save costs this code does not use OpenAI's ada embedding. I also decreased the chunk size to 1000 from 1500 in the original repo as it made sence for my use case.
