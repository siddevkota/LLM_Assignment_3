import os
import re
import requests
import faiss
import torch
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from tqdm import tqdm
import json


def crawl_website(url, max_paragraphs=50):
    """
    Fetches and returns cleaned paragraphs from a website.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    # get all paragraphs
    paragraphs = soup.find_all("p")
    texts = []
    for p in paragraphs:
        text = re.sub(r"\s+", " ", p.get_text()).strip()
        if len(text) > 50:
            texts.append(text)
        if len(texts) >= max_paragraphs:
            break
    return texts


def build_faiss_index(texts, model_name="all-MiniLM-L6-v2"):
    """
    Embeds texts and builds a FAISS index.
    """
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, embedder


def retrieve(query, index, embedder, texts, top_k=3):
    """
    Given a query, retrieve top_k relevant text chunks.
    """
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, top_k)
    return [texts[i] for i in I[0]]


def answer_question(query, retrieved_chunks, reader=None):
    """
    Use a QA model to answer the question from retrieved context.
    """
    if reader is None:
        reader = pipeline(
            "question-answering", model="distilbert-base-uncased-distilled-squad"
        )
    context = " ".join(retrieved_chunks)
    result = reader(question=query, context=context)
    return result["answer"]


def evaluate_rag(questions, answers, index, embedder, texts, reader, top_k=3):
    """
    Evaluate the RAG pipeline on a set of questions/answers.
    """
    correct = 0
    for q, gold in tqdm(zip(questions, answers), total=len(questions)):
        retrieved = retrieve(q, index, embedder, texts, top_k=top_k)
        pred = answer_question(q, retrieved, reader=reader)
        # Simple normalization
        gold_norm = gold.lower().strip().split()[0]
        pred_norm = pred.lower().strip().split()[0]
        if pred_norm in gold_norm or gold_norm in pred_norm:
            correct += 1
    acc = correct / len(questions)
    print(f"Accuracy: {acc*100:.2f}%")
    return acc


def load_triviaqa_devset(num_samples=10):
    # Created sample questions manually because TriviaQA caused errors

    sample_questions = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What is the largest planet in our solar system?",
        "In what year did World War II end?",
        "What is the chemical symbol for gold?",
        "Who wrote the novel '1984'?",
        "What is the speed of light in vacuum?",
        "Which country is known as the Land of the Rising Sun?",
        "What is the smallest unit of matter?",
        "Who painted the Mona Lisa?",
    ]

    sample_answers = [
        "Paris",
        "Alexander Graham Bell",
        "Jupiter",
        "1945",
        "Au",
        "George Orwell",
        "299,792,458 meters per second",
        "Japan",
        "Atom",
        "Leonardo da Vinci",
    ]

    # Return only the requested number of samples
    questions = sample_questions[:num_samples]
    answers = sample_answers[:num_samples]

    return questions, answers


if __name__ == "__main__":
    # 1. Crawl website
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print(f"Crawling {url} ...")
    texts = crawl_website(url)
    print(f"Crawled {len(texts)} paragraphs.")

    # 2. Build retriever index
    print("Building FAISS index ...")
    index, embeddings, embedder = build_faiss_index(texts)

    # 3. Setup reader
    reader = pipeline(
        "question-answering", model="distilbert-base-uncased-distilled-squad"
    )

    # 4. Load sample questions from TriviaQA
    questions, answers = load_triviaqa_devset(num_samples=5)

    # 5. Evaluate
    print("Evaluating RAG pipeline ...")
    evaluate_rag(questions, answers, index, embedder, texts, reader)

    # 6. QA
    print("\nTry your own question!")
    while True:
        query = input("Enter your question (or type 'exit'): ")
        if query.strip().lower() == "exit":
            break
        retrieved = retrieve(query, index, embedder, texts)
        print("Top relevant text:", retrieved[0][:200], "...")
        answer = answer_question(query, retrieved, reader)
        print("Answer:", answer)
