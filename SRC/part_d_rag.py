
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
POLICY_DIR = BASE_DIR / "data" / "policies"
OUTPUT_DIR = BASE_DIR / "outputs"
RAW_DIR = BASE_DIR / "data" / "raw"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
import mock_llm


# --------------------------------------------------
# 2. LOAD POLICY DOCUMENTS
# --------------------------------------------------

def load_policies():

    documents = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):

        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "text": text
        })

    if len(documents) != 5:
        raise ValueError(
            f"Expected 5 policy documents, found {len(documents)}"
        )

    return documents


# --------------------------------------------------
# 3. CREATE CHUNKS
# --------------------------------------------------

def create_chunks(documents, chunk_size=40, overlap=10):

    chunks = []

    for document in documents:

        words = document["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(words[start:end])

            chunks.append({
                "source": document["source"],
                "text": chunk_text
            })

            if end >= len(words):
                break

            start = end - overlap

    return chunks


# --------------------------------------------------
# 4. BUILD TF-IDF VECTOR DATABASE
# --------------------------------------------------

documents = load_policies()

chunks = create_chunks(
    documents,
    chunk_size=40,
    overlap=10
)

chunk_texts = [
    chunk["text"] for chunk in chunks
]

vectorizer = TfidfVectorizer()

chunk_vectors = vectorizer.fit_transform(chunk_texts)


# --------------------------------------------------
# 5. RETRIEVE TOP POLICY CHUNKS
# --------------------------------------------------

def retrieve(query, k=3):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        chunk_vectors
    ).flatten()

    top_indices = similarities.argsort()[::-1][:k]

    results = []

    for index in top_indices:

        results.append({
            "source": chunks[index]["source"],
            "score": float(similarities[index]),
            "text": chunks[index]["text"]
        })

    return results


# Backward-compatible name for existing code.
def search_policy(question, top_k=3):
    return retrieve(question, k=top_k)


# --------------------------------------------------
# 6. ANSWER QUESTIONS USING POLICY CONTEXT
# --------------------------------------------------

FALLBACK_ANSWER = (
    "I could not find that information in the policy documents."
)


def answer(question):

    results = retrieve(question, k=3)

    context_parts = []

    for result in results:

        context_parts.append(
            f"[{result['source']}]\n{result['text']}"
        )

    context = "\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a POLICY ASSISTANT. "
                "Answer only using the supplied policy context. "
                "Do not invent or assume policy information. "
                "If the answer is not available in the context, "
                "say exactly: "
                + FALLBACK_ANSWER
            )
        },
        {
            "role": "user",
            "content": (
                "CONTEXT:\n"
                + context
                + "\nQUESTION:\n"
                + question
            )
        }
    ]

    try:

        response = mock_llm.chat(messages)

        model_answer = response.choices[0].message.content

        return model_answer, results

    except mock_llm.TransientLLMError:

        return FALLBACK_ANSWER, results


# --------------------------------------------------
# 7. ZERO-OVERLAP RETRIEVAL
# --------------------------------------------------

def retrieve_zero_overlap(query, k=3):

    zero_chunks = create_chunks(
        documents,
        chunk_size=40,
        overlap=0
    )

    zero_texts = [
        chunk["text"] for chunk in zero_chunks
    ]

    zero_vectorizer = TfidfVectorizer()

    zero_vectors = zero_vectorizer.fit_transform(
        zero_texts
    )

    query_vector = zero_vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        zero_vectors
    ).flatten()

    top_indices = similarities.argsort()[::-1][:k]

    results = []

    for index in top_indices:

        results.append({
            "source": zero_chunks[index]["source"],
            "score": float(similarities[index]),
            "text": zero_chunks[index]["text"]
        })

    return results


# --------------------------------------------------
# 8. POLICY EVALUATION
# --------------------------------------------------

def find_column(data, possible_names):

    for name in possible_names:

        if name in data.columns:
            return name

    return None


def evaluate_policies():

    evaluation_path = RAW_DIR / "policy_eval.csv"

    if not evaluation_path.exists():

        print("\npolicy_eval.csv not found.")
        print("Retrieval tests will still run.")

        return

    evaluation = pd.read_csv(evaluation_path)

    question_col = find_column(
        evaluation,
        ["question", "query"]
    )

    expected_doc_col = find_column(
        evaluation,
        ["expected_doc", "expected_source"]
    )

    expected_fact_col = find_column(
        evaluation,
        ["expected_fact", "expected_answer"]
    )

    if not question_col or not expected_doc_col or not expected_fact_col:

        print("\nCould not identify evaluation CSV columns.")
        print("Available columns:", list(evaluation.columns))

        return

    overlap_doc_hits = 0
    overlap_fact_hits = 0
    overlap_fallbacks = 0

    zero_doc_hits = 0
    zero_fact_hits = 0

    failures = []

    for _, row in evaluation.iterrows():

        question = str(row[question_col])

        expected_doc = str(row[expected_doc_col])

        expected_fact = str(row[expected_fact_col])

        answerable = (
            expected_doc.lower() not in
            ["none", "nan", "", "not covered"]
        )

        # Overlap evaluation
        results = retrieve(question, k=3)

        model_answer, sources = answer(question)

        source_names = [
            result["source"] for result in results
        ]

        if answerable:

            if expected_doc in source_names:
                overlap_doc_hits += 1

            if expected_fact in model_answer:
                overlap_fact_hits += 1

        elif model_answer.startswith("I could not find"):
            overlap_fallbacks += 1

        # Zero-overlap evaluation
        zero_results = retrieve_zero_overlap(question, k=3)

        zero_source_names = [
            result["source"] for result in zero_results
        ]

        if answerable:

            if expected_doc in zero_source_names:
                zero_doc_hits += 1

            combined_zero_context = "\n".join(
                result["text"] for result in zero_results
            )

            if expected_fact in combined_zero_context:
                zero_fact_hits += 1

            if (
                expected_doc not in zero_source_names
                or expected_fact not in combined_zero_context
            ):
                failures.append(question)

    print("\nOVERLAP EVALUATION")
    print("Expected document retrieved:", overlap_doc_hits)
    print("Expected facts in answers:", overlap_fact_hits)
    print("Unanswerable fallback answers:", overlap_fallbacks)

    print("\nZERO-OVERLAP EVALUATION")
    print("Expected document retrieved:", zero_doc_hits)
    print("Expected facts in retrieved chunks:", zero_fact_hits)

    print("\nQUESTIONS THAT MAY BREAK WITHOUT OVERLAP:")

    for question in failures:
        print("-", question)


# --------------------------------------------------
# 9. TEST RETRIEVAL AND ANSWERS
# --------------------------------------------------

if __name__ == "__main__":

    print("POLICY DOCUMENTS LOADED:", len(documents))

    print("TOTAL CHUNKS CREATED:", len(chunks))

    print("\nCHUNK COUNTS BY DOCUMENT:")

    for document in documents:

        count = sum(
            1 for chunk in chunks
            if chunk["source"] == document["source"]
        )

        print(document["source"], ":", count)

    questions = [
        "refund policy late delivery",
        "What should happen when a delivery is delayed?",
        "How are payment issues handled?",
        "How is customer privacy protected?",
        "When should a support ticket be escalated?"
    ]

    for question in questions:

        print("\n" + "=" * 60)

        print("QUESTION:", question)

        results = retrieve(question, k=3)

        print("\nTOP THREE POLICY CHUNKS:")

        for result in results:

            print(
                result["source"],
                "| Score:",
                round(result["score"], 3)
            )

        model_answer, sources = answer(question)

        print("\nANSWER:")
        print(model_answer)

    # Run evaluation if the evaluation file exists.
    evaluate_policies()

    print("\nPART D RAG COMPLETED SUCCESSFULLY!")
