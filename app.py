import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import joblib
import torch
from transformers import BertTokenizer, BertModel

# Load job listings and pre-trained models
job_listings = pd.read_csv('new_job_details.csv')
job_embeddings = np.load('job_embeddings.npy')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
job_vectors = joblib.load('job_vectors.joblib')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Convert all string columns to lowercase and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return preprocess_text(text)

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    # Check if 'last_hidden_state' is present in the outputs
    if 'last_hidden_state' in outputs:
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    else:
        raise ValueError("BERT model output does not contain 'last_hidden_state'")


def get_recommendations(resume_text, top_n=5):
    if not resume_text:
        raise ValueError("Resume text cannot be empty")
    # TF-IDF similarity
    user_vector = vectorizer.transform([resume_text])
    tfidf_similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()

    # BERT similarity
    user_embedding = get_bert_embedding(resume_text)
    bert_similarity_scores = cosine_similarity([user_embedding], job_embeddings).flatten()

    # Combine similarity scores (weighted average)
    combined_scores = 0.5 * tfidf_similarity_scores + 0.5 * bert_similarity_scores

    similarity_scores = list(enumerate(combined_scores))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_jobs = [job_listings.iloc[i[0]] for i in similarity_scores[:top_n]]
    return top_jobs

def format_job_output(job):
    output = f"""
    ### Company Name:
    {job['Company']}
    
    Job Title:
    {job['Job Role']}
    
    Job Description:
    {job['Job_Description']}
    """
    return output


# Streamlit app
st.title("Job Recommendation System")

input_type = st.radio("Choose input type", ("Text Input", "Upload PDF"))

if input_type == "Text Input":
    user_input = st.text_area("Enter your job role, skills, and career goals")
    if st.button("Get Recommendations"):
        if user_input:
            resume_text = preprocess_text(user_input)
            recommended_jobs = get_recommendations(resume_text, top_n=5)
            st.write("## Recommended Jobs")
            for job in recommended_jobs:
                st.markdown(format_job_output(job) , unsafe_allow_html=True)
        else:
            st.warning("Please enter your job role, skills, and career goals")

elif input_type == "Upload PDF":
    pdf_file = st.file_uploader("Upload your resume in PDF format", type=["pdf"])
    if st.button("Get Recommendations"):
        if pdf_file is not None:
            resume_text = extract_text_from_pdf(pdf_file)
            recommended_jobs = get_recommendations(resume_text, top_n=5)
            st.write("## Recommended Jobs")
            for job in recommended_jobs:
                st.markdown(format_job_output(job) , unsafe_allow_html=True)
        else:
            st.warning("Please upload a PDF file")

