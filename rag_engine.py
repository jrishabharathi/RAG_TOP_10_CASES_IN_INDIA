import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path


# ---------------- LOAD PDF ----------------

def load_pdf(pdf_path):

    reader = PdfReader(str(pdf_path))

    mypages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    full_text = "\n".join(mypages).strip()  #join every pages and removes the blank spaces 

    if not full_text:
        raise ValueError(
            f"No readable text found in {pdf_path.name}"
        )

    return full_text


# ---------------- CHUNKING ----------------

def create_chunks(full_text, chunksize=500, overlap=100):

    if chunksize <= 0 or overlap < 0 or overlap >= chunksize:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    start = 0

    while start < len(full_text):

        end = start + chunksize

        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


# ---------------- EMBEDDINGS ----------------

def build_vectors(
    chunks,
    model_name="all-MiniLM-L6-v2"
):

    if not chunks:
        raise ValueError("Cannot create embeddings without chunks")

    embed_model = SentenceTransformer(model_name)

    embed_vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embed_vectors = np.asarray(
        embed_vectors,
        dtype="float32"
    )

    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(
        embed_vectors.shape[1]
    )

    index.add(embed_vectors)

    return index, embed_model


# ---------------- LOAD MULTIPLE PDFs ----------------

def load_base(pdf_path="datafiles_docs"):

    chunks = []

    folder = Path(pdf_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {pdf_path}"
        )

    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {pdf_path}"
        )

    for pdf in pdf_files:

        full_text = load_pdf(pdf)

        pdf_chunks = create_chunks(full_text)

        chunks.extend(pdf_chunks)

    if not chunks:
        raise ValueError("No text chunks were created")

    index, embed_model = build_vectors(chunks)

    return index, embed_model, chunks


# ---------------- RETRIEVAL ----------------

def retrieve_content(
    question,
    embed_model,
    chunks,
    index,
    num_chunks=3
):

    if not question.strip():
        raise ValueError("Question cannot be blank")

    if num_chunks < 1:
        raise ValueError("Request at least one chunk")

    query_embed = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embed = np.asarray(
        query_embed,
        dtype="float32"
    )

    faiss.normalize_L2(query_embed)

    sim_scores, vec_indices = index.search(
        query_embed,
        min(num_chunks, len(chunks))
    )

    background = ""

    for i in vec_indices[0]:

        if i != -1:
            background += chunks[i] + "\n\n"

    return background


# ---------------- ASK RAG ----------------

def ask_rag(
    question,
    embed_model,
    chunks,
    index,
    api_key,
    num_chunks=3,
    model_name="openai/gpt-oss-120b"
):

    if not api_key:
        raise ValueError("API key missing")

    background = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks
    )

    prompt = f"""
    You are a professional AI assistant.

    Answer the user's question using ONLY the
    document background provided below.

    Rules:
    1. Do not invent or assume facts.
    2. If information is unavailable, say
       "Information Missing".
    3. Clearly distinguish allegations from
       established facts and court findings.
    4. Give answers in simple and clear language.
    5. Do not provide legal advice.

    DOCUMENT BACKGROUND:

    {background}

    USER QUESTION:

    {question}

    ANSWER:
    """

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
