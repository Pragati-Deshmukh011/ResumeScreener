To run the provided code successfully, you'll need to install several dependencies. Here's a list of the dependencies along with the commands to install them:

Streamlit:

bash
Copy code
pip install streamlit
SQLite (should be included with Python installation):

docx2txt:

bash
Copy code
pip install docx2txt
PyPDF2:

bash
Copy code
pip install PyPDF2
NLTK (including required resources):

bash
Copy code
pip install nltk
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader averaged_perceptron_tagger
scikit-learn (for TfidfVectorizer and cosine_similarity):

bash
Copy code
pip install scikit-learn
spaCy (including the 'en_core_web_sm' model):

bash
Copy code
pip install spacy
python -m spacy download en_core_web_sm
Once you've installed these dependencies, you should be able to run the provided code without issues.
