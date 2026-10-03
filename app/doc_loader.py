import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def doc_loader():
    all_docs = []

    for path in glob.glob('docs/**/*.pdf', recursive=True):
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    for path in glob.glob('docs/**/*.txt', recursive=True):
        loader = TextLoader(path, encoding ='utf-8')
        docs = loader.load()
        all_docs.extend(docs)

    print(f'Loaded {len(all_docs)} documents')
    return all_docs