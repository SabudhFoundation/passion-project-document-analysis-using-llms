import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import re
from wordcloud import WordCloud
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import chromadb
from chromadb.config import Settings
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import matplotlib.pyplot as plt

# Download necessary NLTK datasets (run this once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Stopwords list
STOPWORDS = set(stopwords.words('english'))

# POS tags to exclude (prepositions, conjunctions, articles, auxiliary verbs, etc.)
EXCLUDED_POS_TAGS = {
    'IN',  # Prepositions
    'CC',  # Conjunctions
    'DT',  # Determiners/Articles
    'PRP', 'PRP$',  # Pronouns
    'MD',  # Modal verbs
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs (all tenses)
    'CD'  # Cardinal numbers
}

# Setup ChromaDB client
CHROMA_DB_DIR = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
COLLECTION_NAME = "10k_amazon"

# Directory to store images of graphs
GRAPH_DIR = "./graphs"
if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

def process_and_store_pdf(pdf_path):
    """Process the PDF, extract text, store in ChromaDB, and create visualizations."""
    # Load the PDF and extract its text
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Create or get the collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Store the chunks in ChromaDB
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.page_content],
            metadatas=[{"page": chunk.metadata.get("page", "N/A")}],
            ids=[f"chunk_{i}"]
        )

    print(f"Processed and stored {len(chunks)} chunks in ChromaDB.")

    # Combine all chunks to create a complete text for visualization purposes
    all_text = ' '.join([chunk.page_content for chunk in chunks])

    # Generate visualizations
    generate_wordcloud(all_text)
    plot_most_common_words(all_text)
    plot_risk_factors(all_text)
    plot_trends_over_time(all_text)
    plot_comparative_analysis(all_text)
    plot_top_keywords_by_section(chunks)

def filter_words(text):
    """
    Tokenize the text, remove stopwords, and exclude words based on POS tags.
    """
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    filtered_tokens = []

    for word, tag in pos_tag(tokens):  # POS tagging
        if (
            word not in STOPWORDS and  # Remove stopwords
            word.isalpha() and  # Exclude non-alphabetic tokens
            tag not in EXCLUDED_POS_TAGS  # Exclude specific POS tags
        ):
            filtered_tokens.append(word)

    return filtered_tokens

def plot_most_common_words(text, num_words=10):
    """Plot the most common words in the text and save as an image."""
    words = filter_words(text)  # Filter the text
    word_counts = Counter(words).most_common(num_words)

    words, counts = zip(*word_counts)
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color='skyblue')
    plt.title(f'Top {num_words} Most Common Words in the 10-K Report')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    most_common_words_path = os.path.join(GRAPH_DIR, "most_common_words.png")
    plt.savefig(most_common_words_path)
    plt.close()

def generate_wordcloud(text):
    """Generate a word cloud from the filtered text and save as an image."""
    words = filter_words(text)  # Filter the text
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    wordcloud_path = os.path.join(GRAPH_DIR, "wordcloud.png")
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Amazon 10-K Report')
    plt.savefig(wordcloud_path)
    plt.close()

def plot_risk_factors(text):
    """Plot a bar chart of risk factors mentioned in the text and save as an image."""
    risk_factors = ["Competition", "Regulatory Changes", "Cybersecurity", "Supply Chain Disruptions", "Economic Conditions"]

    risk_counts = {risk: text.count(risk) for risk in risk_factors}

    plt.figure(figsize=(10, 6))
    plt.bar(risk_counts.keys(), risk_counts.values(), color='lightgreen')
    plt.title('Frequency of Risk Factors Mentioned in the 10-K Report')
    plt.xlabel('Risk Factors')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    risk_factors_path = os.path.join(GRAPH_DIR, "risk_factors.png")
    plt.savefig(risk_factors_path)
    plt.close()

def plot_trends_over_time(text):
    """Plot trends over time (e.g., mentions of specific years or quarters)."""
    years = [str(year) for year in range(2010, 2024)]  # Adjust range as needed
    year_counts = {year: text.count(year) for year in years}

    plt.figure(figsize=(12, 6))
    plt.plot(years, year_counts.values(), marker='o')
    plt.title('Trends Over Time in the 10-K Report')
    plt.xlabel('Year')
    plt.ylabel('Mentions')
    plt.grid(True)
    trends_over_time_path = os.path.join(GRAPH_DIR, "trends_over_time.png")
    plt.savefig(trends_over_time_path)
    plt.close()

def plot_comparative_analysis(text):
    """Plot a comparative analysis of financial metrics over time."""
    financial_data = {
        'Year': [2018, 2019, 2020, 2021, 2022],
        'Revenue (Billion $)': [232.89, 280.52, 386.06, 469.82, 513.98],
        'Operating Income (Billion $)': [7.30, 14.54, 22.90, 24.89, 33.36],
        'Net Income (Billion $)': [10.07, 11.59, 21.33, 33.36, 33.36]
    }
    df = pd.DataFrame(financial_data)

    plt.figure(figsize=(14, 7))
    df.set_index('Year').plot(kind='bar', ax=plt.gca())
    plt.title('Comparative Analysis of Financial Metrics Over the Years')
    plt.ylabel('Amount (Billion $)')
    plt.xlabel('Year')
    plt.grid(axis='y')
    comparative_analysis_path = os.path.join(GRAPH_DIR, "comparative_analysis.png")
    plt.savefig(comparative_analysis_path)
    plt.close()

def plot_top_keywords_by_section(chunks):
    """Plot top keywords by section in the report and save as an image."""
    sections = [ "Financial Statements", "Risk Factors"]
    keywords_by_section = {}

    for section in sections:
        # Combine text for the section
        section_text = ' '.join([chunk.page_content for chunk in chunks if section.lower() in chunk.page_content.lower()])

        # Filter words using the filtering function
        filtered_keywords = filter_words(section_text)

        # Count the most common words in the filtered keywords
        keywords_count = Counter(filtered_keywords).most_common(10)
        keywords_by_section[section] = keywords_count

    # Plotting the results
    fig, axes = plt.subplots(len(sections), 1, figsize=(12, 18))
    for ax, section in zip(axes, sections):
        if keywords_by_section[section]:
            keywords, counts = zip(*keywords_by_section[section])
            ax.bar(keywords, counts, color='lightblue')
            ax.set_title(f'Top Keywords in {section}')
            ax.set_xlabel('Keywords')
            ax.set_ylabel('Frequency')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No Keywords Found', fontsize=12, ha='center', va='center')
            ax.set_title(f'Top Keywords in {section}')
            ax.set_axis_off()

    section_keywords_path = os.path.join(GRAPH_DIR, "top_keywords_by_section.png")
    plt.tight_layout()
    plt.savefig(section_keywords_path)
    plt.close()



if __name__ == "__main__":
    # Path to Amazon 10-K PDF
    AMAZON_10K_PATH = "amazon.pdf"  # Ensure this path is correct

    if not os.path.exists(AMAZON_10K_PATH):
        print(f"Error: {AMAZON_10K_PATH} not found!")
    else:
        process_and_store_pdf(AMAZON_10K_PATH)
