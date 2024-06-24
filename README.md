# Job Recommendation System

This project implements a Job Recommendation System using natural language processing techniques and deep learning models. It provides recommendations based on the similarity between a user-provided resume (either as text input or uploaded PDF) and a database of job listings.

## Features

- **Text Preprocessing**: Converts text to lowercase and removes non-alphanumeric characters for consistency.
- **TF-IDF Vectorization**: Computes similarity scores using TF-IDF vectors of job descriptions.
- **BERT Embeddings**: Utilizes BERT (Bidirectional Encoder Representations from Transformers) for deep contextualized embeddings of job descriptions.
- **Cosine Similarity**: Measures similarity between user input and job listings using cosine similarity metric.
- **Streamlit Web Application**: Provides an interactive UI for users to input their job preferences and receive recommended job listings.

## Requirements

- Python 3.7+
- Libraries:
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - PyPDF2
  - joblib
  - torch
  - transformers

Install the required packages using:

```bash
pip install -r requirements.txt
