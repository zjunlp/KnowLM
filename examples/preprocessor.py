import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

def clean_text(text):
    # Remove special characters, numbers, and extra whitespaces
    cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text.lower()

def tokenize_text(text):
    # Tokenize text using NLTK
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    # Remove stopwords from the tokenized text
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def format_data(text):
    # Example function for formatting data (replace with your logic)
    formatted_data = " ".join(text)  # Join tokens into a single string
    return formatted_data

if __name__ == "__main__":
    # Example usage
    sample_text = "This is an example sentence. It contains some numbers like 12345."
    cleaned_text = clean_text(sample_text)
    tokens = tokenize_text(cleaned_text)
    filtered_tokens = remove_stopwords(tokens)
    formatted_data = format_data(filtered_tokens)

    print("Cleaned Text:", cleaned_text)
    print("Tokenized Text:", tokens)
    print("Filtered Tokens:", filtered_tokens)
    print("Formatted Data:", formatted_data)

    # Your custom logic can be added here
    # For example, you can perform additional processing on 'formatted_data'
    # or save it to a file, send it to an API, etc.
