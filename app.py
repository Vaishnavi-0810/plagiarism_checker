import streamlit as st
import pandas as pd
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Title
# -----------------------------
st.set_page_config(page_title="AI Plagiarism Checker", page_icon="📄")

st.title("📄 AI Plagiarism Checker")
st.write("Upload multiple text (.txt) files to detect plagiarism using TF-IDF and Cosine Similarity.")

# -----------------------------
# Text Preprocessing Function
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# -----------------------------
# File Uploader
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload Text Files",
    type=["txt"],
    accept_multiple_files=True
)

# -----------------------------
# If files are uploaded
# -----------------------------
if uploaded_files:

    documents = []
    filenames = []

    for file in uploaded_files:
        text = file.read().decode("utf-8")
        text = preprocess(text)

        documents.append(text)
        filenames.append(file.name)

    # -----------------------------
    # TF-IDF Vectorization
    # -----------------------------
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    # -----------------------------
    # Cosine Similarity
    # -----------------------------
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # -----------------------------
    # Show Similarity Matrix
    # -----------------------------
    st.subheader("Similarity Matrix")

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=filenames,
        columns=filenames
    )

    st.dataframe(similarity_df)

    # -----------------------------
    # Generate Report
    # -----------------------------
    threshold = 0.70

    results = []

    n = len(documents)

    for i in range(n):
        for j in range(i + 1, n):

            similarity = similarity_matrix[i][j]

            if similarity >= threshold:

                results.append([
                    filenames[i],
                    filenames[j],
                    round(similarity * 100, 2)
                ])

    # -----------------------------
    # Display Report
    # -----------------------------
    st.subheader("Plagiarism Report")

    if len(results) == 0:
        st.success("✅ No plagiarism detected.")
    else:

        report = pd.DataFrame(
            results,
            columns=[
                "Document 1",
                "Document 2",
                "Similarity (%)"
            ]
        )

        st.dataframe(report)

        csv = report.to_csv(index=False)

        st.download_button(
            label="📥 Download CSV Report",
            data=csv,
            file_name="plagiarism_report.csv",
            mime="text/csv"
        )