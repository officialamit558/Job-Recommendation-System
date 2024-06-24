# Job Recommendation System
This project implements a Job Recommendation System using natural language processing techniques and deep learning models. It provides recommendations based on the similarity between a user-provided resume (either as text input or uploaded PDF) and a database of job listings.

Features
Text Preprocessing: Converts text to lowercase and removes non-alphanumeric characters for consistency.
TF-IDF Vectorization: Computes similarity scores using TF-IDF vectors of job descriptions.
BERT Embeddings: Utilizes BERT (Bidirectional Encoder Representations from Transformers) for deep contextualized embeddings of job descriptions.
Cosine Similarity: Measures similarity between user input and job listings using cosine similarity metric.
Streamlit Web Application: Provides an interactive UI for users to input their job preferences and receive recommended job listings.
Requirements
Python 3.7+
Libraries:
streamlit
pandas
numpy
scikit-learn
PyPDF2
joblib
torch
transformers
Install the required packages using:

Copy code
pip install -r requirements.txt
Usage
To run the application locally:

arduino
Copy code
streamlit run app.py
How it Works
Input Types:

Text Input: Users can directly enter their job role, skills, and career goals.
Upload PDF: Users can upload their resume in PDF format, and the system extracts text for analysis.
Processing:

Text Preprocessing: Cleans and prepares input text for analysis.
TF-IDF Similarity: Computes similarity scores using TF-IDF vectors of job descriptions.
BERT Embeddings: Calculates deep embeddings of job descriptions using BERT.
Recommendation:

Combines TF-IDF and BERT similarity scores to generate top job recommendations.
Displays recommended jobs along with company name, job title, description, required skills, and location.
Folder Structure
graphql
Copy code
├── app.py                # Streamlit application script
├── new_job_details.csv   # Sample job details dataset
├── job_embeddings.npy    # Pre-computed job embeddings
├── tfidf_vectorizer.joblib  # Serialized TF-IDF vectorizer
├── job_vectors.joblib    # Serialized job TF-IDF vectors
├── README.md             # Project readme (you are here)
└── requirements.txt      # Python dependencies
Contributors
@YourGitHubHandle - Developer
License
This project is licensed under the MIT License - see the LICENSE file for details.
