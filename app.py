import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")

st.title("🤖 FAQ Chatbot")
st.write("Ask me a question about Python, AI, or programming.")

# FAQ collection
faqs = [
    {
        "question": "What is Python?",
        "answer": "Python is a high-level programming language known for its simple syntax."
    },
    {
        "question": "What is artificial intelligence?",
        "answer": "Artificial Intelligence enables computers to perform tasks that normally require human intelligence."
    },
    {
        "question": "What is machine learning?",
        "answer": "Machine learning enables computers to learn patterns from data."
    },
    {
        "question": "What is NLP?",
        "answer": "Natural Language Processing helps computers understand human language."
    },
    {
        "question": "What is a Python variable?",
        "answer": "A variable stores a value in a program, such as age = 20."
    },
    {
        "question": "What is a Python function?",
        "answer": "A function is a reusable block of code defined using the def keyword."
    },
    {
        "question": "What is a Python list?",
        "answer": "A list is an ordered collection of items that can be changed."
    },
    {
        "question": "What is a loop?",
        "answer": "A loop repeats a block of code. Python has for loops and while loops."
    },
    {
        "question": "What is a chatbot?",
        "answer": "A chatbot is a program that communicates with users through text or voice."
    },
    {
        "question": "What is data science?",
        "answer": "Data science uses programming, statistics, and analysis to extract useful insights from data."
    }
]

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Prepare the FAQ vectors
faq_questions = [
    preprocess_text(faq["question"])
    for faq in faqs
]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_questions)


# Define the answer function BEFORE calling it
def get_answer(user_question):
    cleaned_question = preprocess_text(user_question)

    if not cleaned_question:
        return "Please enter a question."

    user_vector = vectorizer.transform([cleaned_question])

    scores = cosine_similarity(
        user_vector,
        faq_vectors
    )[0]

    best_index = scores.argmax()

    if scores[best_index] < 0.15:
        return (
            "Sorry, I couldn't find a matching answer. "
            "Try asking about Python, AI, machine learning, or NLP."
        )

    return faqs[best_index]["answer"]


# Question input
user_question = st.text_input("Ask your question here:")

if st.button("Get Answer"):
    if user_question.strip():
        answer = get_answer(user_question)
        st.success(answer)
    else:
        st.warning("Please enter a question.")


# Sidebar
with st.sidebar:
    st.header("About This Chatbot")
    st.write(
        "This chatbot uses text preprocessing, TF-IDF, "
        "and cosine similarity to match questions with answers."
    )

    st.subheader("Try these questions")
    st.write("- What is Python?")
    st.write("- What is machine learning?")
    st.write("- What is NLP?")
    st.write("- What is a Python function?")

st.divider()
st.caption("CodeAlpha AI Internship | FAQ Chatbot")
