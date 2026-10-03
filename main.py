from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.retriever import get_retriever
from app.llm import get_llm
from app.reranker import rerank

load_dotenv()


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def process_text(text):
    print("Searching...")

    retriever = get_retriever()
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant which answers questions based on context given below. Answer in concise manner.
    If you don't know the answer or if the relevant information is not in context, answer with "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:""")

    rag_chain = prompt | llm | StrOutputParser()

    docs = retriever.invoke(text)
    docs = rerank(text, docs, top_k=5)
    context = format_docs(docs)

    answer = rag_chain.invoke({
        "context": context,
        "question": text
    })

    return {"response": answer}


if __name__ == "__main__":
    questions = [
    "What are the core hours during which employees must be reachable?",
    "How many weeks of fully paid leave does the birth parent receive?",
    "How much is the one-time home office budget, and within how many days must receipts be submitted?",
    "What is the annual learning budget per employee?",
    "What is the maximum hotel reimbursement per night in London?",
    "What is the minimum password length and how often must passwords be changed?",
    "How much is the referral bonus and how long must the referred candidate stay?",
    # out-of-scope
    "Why was Eddard Stark executed in Game of Thrones?",
    "What is the capital of France?",
    "What is the CEO's annual salary at Nordvale Robotics?"
    ]

    for q in questions:
        result = process_text(q)
        print(f"Q: {q}")
        print(f"A: {result['response']}")
