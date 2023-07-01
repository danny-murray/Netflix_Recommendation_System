import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Loading SpaCy and pre-trained model
nlp = spacy.load('en_core_web_md')

# Preprocessing dataset
def preprocess_dataset(dataset):
    dataset['combined_text'] = dataset['title'] + ' ' + dataset['director'] + ' ' + dataset['cast'] + ' ' + dataset['description']
    dataset['combined_text'] = dataset['combined_text'].fillna('')
    dataset['cast'] = dataset['cast'].fillna('')
    return dataset

# Extracting relevant entities from user preferences
def extract_entities(user_preferences):
    doc = nlp(user_preferences)
    entities = {}
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            entities[ent.text.lower()] = ent.text.lower()
    return entities

# Tokenising, lemmatising and removing stopwords
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    doc = nlp(text)

    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.lemma_.lower() not in stop_words
        and token.text not in string.punctuation
    ]

    return tokens

# Finding similar programmes based on user preferences
def find_similar_shows(user_preferences, dataset):
    tfidf_vectoriser = TfidfVectorizer(tokenizer=preprocess_text)

    # Vectorising combined text
    tfidf_matrix = tfidf_vectoriser.fit_transform(dataset['combined_text'])

    # Vectorising user preferences
    user_preferences_vector = tfidf_vectoriser.transform([user_preferences])

    # Calculating cosine similarity between user preferences and dataset
    similarities = cosine_similarity(user_preferences_vector, tfidf_matrix)[0]

    # Indices of programmes sorted by similarity score (descending order)
    sorted_indices = similarities.argsort()[::-1]

    # Collect all similar programmes that mention any specified actors
    similar_shows = []
    actor_names = extract_entities(user_preferences).keys()
    for idx in sorted_indices:
        show_text = ' '.join(str(val) for val in dataset.loc[idx].values)
        show_text = show_text.lower()
        if any(actor.lower() in show_text for actor in actor_names):
            similar_shows.append((dataset.loc[idx, 'title'], dataset.loc[idx, 'type'], round(similarities[idx] * 100, 2)))
        if len(similar_shows) == 5:
            break

    return similar_shows


# Handling user input, display recommendations
def handle_user_input():
    user_preferences = input_entry.get()
    similar_shows = find_similar_shows(user_preferences, dataset)

    # Displaying recommended programmes
    if similar_shows:
        messagebox.showinfo("Recommendations", "Recommended TV Shows or Movies:\n\n" + "\n".join(
            f"- Title: {title}\n  Type: {show_type}\n  Similarity: {similarity}%" for title, show_type, similarity in
            similar_shows))
    else:
        messagebox.showinfo("Recommendations", "No recommendations found.")

# Loading Netflix dataset
dataset = pd.read_csv('netflix_titles.csv')
dataset = preprocess_dataset(dataset)

# Create GUI window
window = tk.Tk()
window.title("Viewing Recommendations")
window.geometry("400x200")
window.configure(bg="#363636")

# Mainframe Style
frame = ttk.Frame(window)
frame.configure(style="Gray.TFrame")
frame.pack(pady=20)

# Input label and entry style
input_label = ttk.Label(frame, text="Enter your preferences for TV shows or movies:", foreground="white", background="#363636", font=("Helvetica", 12, "bold"))
input_label.pack(pady=10)
input_entry = ttk.Entry(frame, width=50)
input_entry.configure(style="LightGray.TEntry")
input_entry.pack(pady=10)

# Submit button style
submit_button = ttk.Button(frame, text="Submit", command=handle_user_input, style="Gray.TButton")
submit_button.pack(pady=10)

# Window style
window.style = ttk.Style()
window.style.configure("Gray.TFrame",
                        background="#363636")
window.style.configure("Gray.TButton",
                        background="#404040",
                        font=("Helvetica", 10, "bold"),
                        width=15)
window.style.configure("LightGray.TEntry",
                        background="#C0C0C0")

# Run the GUI window
window.mainloop()
