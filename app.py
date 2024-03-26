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
st.title('Resume Screener')
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


# Save data to SQLite database and process resumes for each candidate
if st.sidebar.button('Screen Resumes'):
    if uploaded_files:
        job_postings_list = job_postings_input.split('\n')
        st.success('Resumes uploaded successfully!')

        for uploaded_file in uploaded_files:
            st.sidebar.subheader(f'Processing for {uploaded_file.name}')

            # Extract candidate's name from the resume
            candidate_name = extract_candidate_name(uploaded_file)
            st.sidebar.write(f'Candidate Name: {candidate_name}')

            # Save candidate information to database
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
                # Tokenize text and extract skills
                tokens = word_tokenize(file_content)
                tokens = [word.lower() for word in tokens if word.isalpha()]
                stop_words = set(stopwords.words('english'))
                tokens = [word for word in tokens if word not in stop_words]

                # Part-of-speech tagging
                tagged_tokens = pos_tag(tokens)

                # Extracting nouns and proper nouns (potential skills)
                skills = [word for word, pos in tagged_tokens if pos.startswith('NN')]

                # Save extracted skills to database
                cursor.execute('INSERT INTO candidate_skills (candidate_name, skill) VALUES (?, ?)',
                               (candidate_name, ', '.join(skills)))
                conn.commit()

        st.success('Resumes processed successfully!')

# Output Module
st.header('Screening Output')

# Retrieve candidate names from the database
cursor.execute('SELECT DISTINCT candidate_name FROM candidate_info')
candidate_names = [row[0] for row in cursor.fetchall()]

# Process resumes and display output for each candidate
for candidate_name in candidate_names:
    st.subheader(f'{candidate_name}\'s Skills:')

    # Retrieve skills from the database for the current candidate
    cursor.execute('SELECT skill FROM candidate_skills WHERE candidate_name = ?', (candidate_name,))
    candidate_skills = ', '.join([row[0] for row in cursor.fetchall()])
    st.write(candidate_skills)

    if job_postings_input:
        st.subheader('Job Postings')
        st.write(job_postings_list)

        # Vectorize job postings and candidate skills
        vectorizer = TfidfVectorizer()
        job_postings_vectorized = vectorizer.fit_transform(job_postings_list)
        candidate_skills_vectorized = vectorizer.transform([candidate_skills])

        # Calculate cosine similarity between candidate skills and job postings
        similarity_scores = cosine_similarity(candidate_skills_vectorized, job_postings_vectorized)

        # Find the most appropriate job posting based on similarity
        most_appropriate_job_index = similarity_scores.argmax()
        most_appropriate_job_posting = job_postings_list[most_appropriate_job_index]

        st.subheader('Most Appropriate Job Posting:')
        st.write(most_appropriate_job_posting)

if st.button('Delete Data'):
    # Delete data from candidate_info table
    cursor.execute('DELETE FROM candidate_info')
    # Delete data from candidate_skills table
    cursor.execute('DELETE FROM candidate_skills')
    conn.commit()
    st.success('Data deleted successfully!')

# Footer
st.sidebar.markdown('Developed by Prerna Gaikwad')

# Close SQLite connection
conn.close()


