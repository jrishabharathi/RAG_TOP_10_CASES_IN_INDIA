import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from groq import Groq

from rag_engine import load_base, ask_rag


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Criminal Cases RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PDF_FOLDER = Path("datafiles_docs")


# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #eff8ff;
}

[data-testid="stSidebar"] {
    background-color: #dceeff;
}

/* Keep text readable when the user's Streamlit theme is dark. */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] label,
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label {
    color: #17324d;
}

/* Match Streamlit's fixed chat input area to the light page. */
[data-testid="stBottom"] {
    background-color: #eff8ff;
}

[data-testid="stSidebar"] {
    min-width: 260px;
    max-width: 260px;
}

.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: #2563eb;
}

.sub-title {
    text-align: center;
    color: gray;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. SESSION STATE
# ==================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==================================================
# 4. API KEY
# ==================================================

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""


# ==================================================
# 5. READ PDF DOCUMENTS
# ==================================================

def get_pdf_files():

    if not PDF_FOLDER.exists():
        return []

    return sorted(PDF_FOLDER.glob("*.pdf"))


def extract_pdf_details(pdf_path):

    reader = PdfReader(str(pdf_path))

    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")

    full_text = "\n".join(pages).strip()

    return {
        "name": pdf_path.name,
        "pages": len(reader.pages),
        "size": round(pdf_path.stat().st_size / 1024, 2),
        "text": full_text
    }


# ==================================================
# 6. GENERATE PDF DESCRIPTION USING GROQ
# ==================================================

def generate_file_description(file_name, pdf_text, api_key):

    if not pdf_text.strip():
        return "No readable text found. This PDF may require OCR."

    client = Groq(api_key=api_key)

    prompt = f"""
    You are an assistant explaining PDF documents
    used in an educational criminal case RAG chatbot.

    Read the following document content and provide:

    1. Document title or subject.
    2. Main topic of the document.
    3. Important information covered.
    4. Purpose of this document in the RAG chatbot.

    Write a short and clear description in simple English.
    Use neutral, factual language.
    Distinguish allegations from established court findings.
    Do not invent information or include graphic details.

    File name: {file_name}

    Document content:
    {pdf_text[:6000]}
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# ==================================================
# 7. LOAD RAG DATABASE
# ==================================================

@st.cache_resource(show_spinner=False)
def initialize_rag():

    return load_base(str(PDF_FOLDER))


# ==================================================
# 8. SIDEBAR
# ==================================================

st.sidebar.title("⚖️ CASE CHATBOT")

st.sidebar.caption("TOP 10 CRIMINAL CASES IN INDIA")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Chatbot", "RAG Pipeline", "Info"],
    key="main_navigation"
)

info_page = None

if page == "Info":

    info_page = st.sidebar.radio(
        "Information",
        ["Description", "Technologies Used"],
        key="info_navigation"
    )

st.sidebar.divider()

st.sidebar.subheader("👥 Team Members")
team_members = [
    "Chandana H L",
    "Dumbitha",
    "Lahari",
    "Brundashree",
    "Deekshith M",
    "J Rishab Bharathi",
]

for member in team_members:
    st.sidebar.markdown(f"• {member}")

st.sidebar.divider()
st.sidebar.caption("Retrieval Augmented Generation System")


# ==================================================
# 9. CHATBOT PAGE
# ==================================================

if page == "Chatbot":

    st.markdown(
        '<p class="main-title">TOP 10 CRIMINAL CASES IN INDIA</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sub-title">AI Powered RAG Chatbot</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("💬 Ask Your Question")

    st.write(
        "Ask questions about the criminal case documents "
        "available in the datafiles_docs folder."
    )

    api_key = get_api_key()

    if not api_key:

        st.error(
            "Groq API key is missing. "
            "Please configure .streamlit/secrets.toml."
        )

    else:

        try:

            with st.spinner("Loading PDF documents and RAG database..."):

                index, embed_model, chunks = initialize_rag()

            st.success("RAG database loaded successfully.")

            # Display previous messages
            for message in st.session_state.chat_history:

                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # User input
            question = st.chat_input("Ask your question here...")

            if question:

                with st.chat_message("user"):
                    st.markdown(question)

                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question
                })

                with st.chat_message("assistant"):

                    with st.spinner("Searching documents..."):

                        try:

                            answer = ask_rag(
                                question=question,
                                embed_model=embed_model,
                                chunks=chunks,
                                index=index,
                                api_key=api_key,
                                num_chunks=3
                            )

                            st.markdown(answer)

                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer
                            })

                        except Exception as e:

                            st.error(f"Answer generation error: {e}")

        except Exception as e:

            st.error(f"RAG loading error: {e}")

    if st.button("🗑️ Clear Chat"):

        st.session_state.chat_history = []
        st.rerun()


# ==================================================
# 10. RAG PIPELINE PAGE
# ==================================================

elif page == "RAG Pipeline":

    st.title("⚙️ RAG PIPELINE")

    st.write(
        "This diagram explains how the application "
        "processes PDF documents and generates answers."
    )

    st.divider()

    pipeline = """
    digraph RAG {

        rankdir=TB;

        node [
            shape=box,
            style="rounded,filled",
            fillcolor="#DBEAFE",
            color="#2563EB",
            fontname="Arial"
        ];

        A [label="PDF Documents"];
        B [label="Text Extraction"];
        C [label="Text Chunking"];
        D [label="Embedding Generation"];
        E [label="FAISS Vector Database"];

        F [label="User Question"];
        G [label="Question Embedding"];
        H [label="Similarity Search"];
        I [label="Relevant Chunks"];
        J [label="Groq LLM"];
        K [label="Generated Answer"];

        A -> B -> C -> D -> E;

        F -> G -> H;
        E -> H;
        H -> I -> J -> K;
    }
    """

    st.graphviz_chart(pipeline, use_container_width=True)

    st.divider()

    st.subheader("Pipeline Explanation")

    pipeline_steps = [
        ("1. PDF Loading",
         "Loads all PDF documents from datafiles_docs."),

        ("2. Text Extraction",
         "Extracts readable text from PDF pages."),

        ("3. Text Chunking",
         "Divides the extracted text into smaller chunks."),

        ("4. Embedding Generation",
         "Converts text chunks into numerical vectors."),

        ("5. FAISS Indexing",
         "Stores vectors for similarity searching."),

        ("6. User Query",
         "Accepts a question from the user."),

        ("7. Similarity Search",
         "Finds the most relevant document chunks."),

        ("8. Context Retrieval",
         "Provides retrieved text to the language model."),

        ("9. Answer Generation",
         "Groq generates an answer using the retrieved context.")
    ]

    for title, description in pipeline_steps:

        st.markdown(f"**{title}**")
        st.write(description)


# ==================================================
# 11. INFO PAGE
# ==================================================

elif page == "Info":

    # ----------------------------------------------
    # DESCRIPTION
    # ----------------------------------------------

    if info_page == "Description":

        st.title("📘 PROJECT DESCRIPTION")

        st.divider()

        st.subheader("About the Project")

        st.write("""
        The Top 10 Criminal Cases in India RAG Chatbot
        is an AI-powered question-answering application.

        It uses Retrieval Augmented Generation (RAG)
        to retrieve relevant information from PDF documents
        and generate answers to user questions.

        The system uses Sentence Transformers for embeddings,
        FAISS for similarity search, and Groq for
        language model-based answer generation.
        """)

        st.divider()

        st.subheader("📂 DOCUMENTS IN DATAFILES_DOCS")

        st.write(
            "The following section automatically displays "
            "the PDF files stored in the datafiles_docs folder."
        )

        pdf_files = get_pdf_files()

        if not pdf_files:

            st.warning(
                "No PDF files found. Please check the datafiles_docs folder."
            )

        else:

            st.success(
                f"Total PDF documents found: {len(pdf_files)}"
            )

            api_key = get_api_key()

            for number, pdf_file in enumerate(pdf_files, start=1):

                with st.expander(
                    f"{number}. {pdf_file.name}",
                    expanded=False
                ):

                    try:

                        details = extract_pdf_details(pdf_file)

                        st.write(f"**File Name:** {details['name']}")

                        st.write(
                            f"**Number of Pages:** {details['pages']}"
                        )

                        st.write(
                            f"**File Size:** {details['size']} KB"
                        )

                        st.divider()

                        st.markdown("**Document Description**")

                        if st.button(
                            "Generate Description",
                            key=f"description_{number}"
                        ):

                            if not api_key:

                                st.error(
                                    "Groq API key is missing."
                                )

                            elif not details["text"].strip():

                                st.warning(
                                    "No readable text found. "
                                    "OCR may be required for this PDF."
                                )

                            else:

                                with st.spinner(
                                    "Generating document description..."
                                ):

                                    try:

                                        description = generate_file_description(
                                            details["name"],
                                            details["text"],
                                            api_key
                                        )

                                        st.session_state[
                                            f"pdf_description_{number}"
                                        ] = description

                                    except Exception as e:

                                        st.error(
                                            f"Description error: {e}"
                                        )

                        saved_description = st.session_state.get(
                            f"pdf_description_{number}"
                        )

                        if saved_description:
                            st.write(saved_description)

                        with st.expander("View extracted text preview"):

                            if details["text"]:
                                st.text(details["text"][:1500])
                            else:
                                st.warning("No readable text available.")

                    except Exception as e:

                        st.error(
                            f"Unable to read {pdf_file.name}: {e}"
                        )

        st.divider()

        st.subheader("Main Features")

        st.markdown("""
        - PDF document loading
        - Automatic document description
        - Text extraction and chunking
        - Embedding generation
        - FAISS similarity search
        - Relevant information retrieval
        - AI-powered answer generation
        - Interactive chatbot
        - Chat history management
        """)

        st.info(
            "This application is for educational and "
            "informational purposes only, not legal advice."
        )


    # ----------------------------------------------
    # TECHNOLOGIES USED
    # ----------------------------------------------

    elif info_page == "Technologies Used":

        st.title("💻 TECHNOLOGIES USED")

        st.divider()

        technologies = [
            (
                "Python",
                "Main programming language used to develop the application."
            ),
            (
                "Streamlit",
                "Creates the interactive web application interface."
            ),
            (
                "PyPDF",
                "Extracts text and page information from PDF documents."
            ),
            (
                "Sentence Transformers",
                "Converts document text into numerical embeddings."
            ),
            (
                "FAISS",
                "Stores and searches vector embeddings efficiently."
            ),
            (
                "NumPy",
                "Performs numerical operations on embedding vectors."
            ),
            (
                "Groq API",
                "Provides language model-based answer generation and document descriptions."
            ),
            (
                "RAG",
                "Combines document retrieval with language model generation."
            )
        ]

        for name, description in technologies:

            st.subheader(name)
            st.write(description)
            st.divider()
