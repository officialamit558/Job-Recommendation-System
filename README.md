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
```


## Usage
- To run the application locally:

```bash
streamlit run app.py
```
## How it Works
### Input Types
- Text Input: Users can directly enter their job role, skills, and career goals.
- Upload PDF: Users can upload their resume in PDF format, and the system extracts text for analysis.
### Processing
- Text Preprocessing: Cleans and prepares input text for analysis.
- TF-IDF Similarity: Computes similarity scores using TF-IDF vectors of job descriptions.
- BERT Embeddings: Calculates deep embeddings of job descriptions using BERT.
## Recommendation
- Combines TF-IDF and BERT similarity scores to generate top job recommendations.
- Displays recommended jobs along with company name, job title, description, required skills, and location.

## Folder Structure

### Explanation

- **`app.py`**: This file contains the script for the Streamlit application, which is used to run the job recommendation system.
  
- **`new_job_details.csv`**: A sample dataset containing job details such as company, job title, description, skills, and location.

- **`job_embeddings.npy`**: This file stores pre-computed embeddings of job descriptions, likely generated using a machine learning model or algorithm.

- **`tfidf_vectorizer.joblib`**: Serialized TF-IDF vectorizer used to transform job descriptions into numerical vectors for similarity computation.

- **`job_vectors.joblib`**: Serialized TF-IDF vectors of job descriptions, likely used for efficient retrieval and computation.

- **`README.md`**: The markdown file you are currently viewing, serving as the documentation and overview of the project.

- **`requirements.txt`**: A file listing all Python dependencies required to run the project, typically used with `pip install -r requirements.txt` to install dependencies.

### Notes

- Ensure file and directory names are accurate and reflect the actual structure of your project.
  
- Use comments (`#`) to provide brief descriptions or explanations next to each file or directory entry.
  
- Use Markdown formatting to maintain readability and clarity in your `README.md` file.

This structured approach helps users and contributors understand the organization of your project files quickly and locate specific components they may need to work with or understand further. Adjust the comments and descriptions as per your project's specific details and needs.


