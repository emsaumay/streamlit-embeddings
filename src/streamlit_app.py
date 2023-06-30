import streamlit as st
import pandas as pd
from src.question_answering import get_answer

# Load embeddings
embeddings_csv_path = "embeddings.csv"
embeddings = pd.read_csv(embeddings_csv_path)

def main():
    st.title("Research Paper Q&A")
    st.write("Ask a question about the research paper:")

    # Text input for question
    question = st.text_input("Your question")

    # Button to submit question
    if st.button("Submit"):
        # Use embeddings to get answer
        answer = get_answer(question, embeddings)
        st.write("Answer:", answer)

if __name__ == "__main__":
    main()