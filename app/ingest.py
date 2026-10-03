import os
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.doc_loader import doc_loader
from app.vector_db import get_vector_store
from app.parent_retriever import build_parent_retriever
from app.chunker import parent_splitter, child_splitter
from app.local_store import PickleDocStore

def clear_old_stores():
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db", ignore_errors=True)
        print("chroma_db cleaned")
    if os.path.exists("./docstore.pkl"):
        os.remove("./docstore.pkl")
        print("docstore.pkl cleaned")

def run_ingestion():
    clear_old_stores()

    vectorstore = get_vector_store()

    docstore = PickleDocStore('./docstore.pkl')

    retriever = build_parent_retriever(vectorstore, docstore, parent_splitter, child_splitter)

    docs = doc_loader()
    if not docs:
        print('No documents found')
        return

    print(f'{len(docs)} documents loading...')
    retriever.add_documents(docs, ids=None)
    print('Ingestion_successful!')

if __name__ == '__main__':
    run_ingestion()
    os._exit(0)