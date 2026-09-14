import streamlit as st
import tempfile
import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
api_key = st.secrets["OPEN_API_KEY"]
load_dotenv()

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer")
st.markdown("Upload your resume and get AI-powered feedback.")

api_key_open = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("OPENROUTER_API_KEY not found in .env")
    st.stop()

llm = ChatOpenRouter(
    model="openai/gpt-oss-20b",
    api_key=api_key,
    temperature=0.3
)

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

if uploaded_file:

    with st.spinner("Reading Resume..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        resume_text = "\n".join(
            [page.page_content for page in pages]
        )

    st.success("Resume Loaded Successfully")

    with st.expander("Resume Preview"):
        st.write(resume_text[:3000])

    analyze_btn = st.button("Analyze Resume")

    if analyze_btn:

        prompt = ChatPromptTemplate.from_template(
        """
        You are an expert ATS Resume Reviewer.

        Analyze the resume below.

        Resume:
        {resume}

        Return:

        1. ATS Score (0-100)
        2. Technical Skills Found
        3. Missing Skills
        4. Strengths
        5. Weaknesses
        6. Resume Improvement Suggestions
        7. 10 Interview Questions

        Format clearly using markdown.
        """
        )

        chain = prompt | llm

        with st.spinner("Analyzing Resume..."):

            response = chain.invoke(
                {
                    "resume": resume_text
                }
            )

        st.subheader("📊 Analysis Report")
        st.markdown(response.content)
