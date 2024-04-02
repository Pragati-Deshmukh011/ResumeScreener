import streamlit as st
import sqlite3
import docx2txt
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import pandas as pd

# Download NLTK resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Download spaCy model for named entity recognition (NER)
nlp = spacy.load('en_core_web_sm')

# Connect to SQLite database
conn = sqlite3.connect('resume_screener.db')
cursor = conn.cursor()

# Create tables if they don't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS candidate_info
                  (id INTEGER PRIMARY KEY, candidate_name TEXT, candidate_email TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS candidate_skills
                  (id INTEGER PRIMARY KEY, candidate_name TEXT, skill TEXT)''')
conn.commit()

# Page layout
st.title('Automatic Resume Screener')
col1, col2, col3 = st.columns([1, 1, 2])

# Insert Resume Section (multiple files)
with col1:
    st.sidebar.header('Insert Resumes')
    uploaded_files = st.sidebar.file_uploader('Upload Resumes', type=['pdf', 'docx'], accept_multiple_files=True)

# Job Posts Module (multiple inputs)
with col2:
    st.sidebar.header('Job Posts')
    job_postings_input = st.sidebar.text_area('Enter Job Postings (One per line)')

def extract_name_from_text(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return ent.text
    return 'Unknown'

def extract_candidate_name(uploaded_file):
    if uploaded_file.type == 'application/pdf':
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return extract_name_from_text(text)
    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        text = docx2txt.process(uploaded_file)
        return extract_name_from_text(text)
# Counter for number of resumes processed
resumes_processed = 0
# Save data to SQLite database and process resumes for each candidate
if st.sidebar.button('Screen Resumes'):
    if uploaded_files:
        job_postings_list = job_postings_input.split('\n')
        st.success('Resumes uploaded successfully!')

        results_list = []
        vectorizer = TfidfVectorizer()

        for uploaded_file in uploaded_files:
            st.sidebar.subheader(f'Processing for {uploaded_file.name}')

            candidate_name = extract_candidate_name(uploaded_file)
            st.sidebar.write(f'Candidate Name: {candidate_name}')

            cursor.execute('INSERT INTO candidate_info (candidate_name) VALUES (?)', (candidate_name,))
            conn.commit()

            file_content = None
            if uploaded_file.type == 'application/pdf':
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ''
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                file_content = text
            elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                file_content = docx2txt.process(uploaded_file)
            if file_content:
                # Increment counter for each resume processed
                resumes_processed += 1
            if file_content:
                tokens = word_tokenize(file_content)
                tokens = [word.lower() for word in tokens if word.isalpha()]
                stop_words = set(stopwords.words('english'))
                tokens = [word for word in tokens if word not in stop_words]

                tagged_tokens = pos_tag(tokens)
                skills = [word for word, pos in tagged_tokens if pos.startswith('NN')]

                cursor.execute('INSERT INTO candidate_skills (candidate_name, skill) VALUES (?, ?)',
                               (candidate_name, ', '.join(skills)))
                conn.commit()

                job_postings_vectorized = vectorizer.fit_transform(job_postings_list)
                candidate_skills_vectorized = vectorizer.transform([', '.join(skills)])
                similarity_scores = cosine_similarity(candidate_skills_vectorized, job_postings_vectorized)

                # Find the most appropriate job posting based on similarity
                max_similarity_index = similarity_scores.argmax()
                max_similarity_score = similarity_scores[0][max_similarity_index]
                most_appropriate_job_posting = job_postings_list[max_similarity_index]

                results_list.append({'Candidate Name': candidate_name,
                                     'Job Posting': most_appropriate_job_posting,
                                     'Similarity Score': max_similarity_score})

        results_df = pd.DataFrame(results_list)
        st.success('Resumes processed successfully!')

        st.subheader('Results:')
        st.dataframe(results_df[['Candidate Name', 'Job Posting', 'Similarity Score']])
# Display number of resumes processed
st.subheader('Number of Resumes Processed')
st.write(resumes_processed)
# Delete Data Button
if st.button('Delete Data'):
    cursor.execute('DELETE FROM candidate_info')
    cursor.execute('DELETE FROM candidate_skills')
    conn.commit()
    st.success('Data deleted successfully!')

# Footer
st.sidebar.markdown('Developed by Prerna Gaikwad')

# Close SQLite connection
conn.close()
