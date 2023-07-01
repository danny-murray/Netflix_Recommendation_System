# Film and TV programme Recommendation System
This project implements a recommendation system based on user preferences to sugget TV shows and films using a Netflix dataset from Kaggle. 

## Features
- Utilises natural language processing (NLP) techniques for text processing and analysis
- Extracts relevant entities from user preferences
- Tokenises, lemmatises and removes stopwords from text data
- Calculates cosine similarity between user preferences and dataset
- Displays top 5 most similar TV shows or films that match specified actors

## Dependencies
- pandas
- spacy
- scikit-learn
- nltk
- tkinter

## Installation
1. Clone the repository:

```bash
git clone https://github.com/danny-murray/Netflix-Recommendation-System.git
```

2. Install the required dependencies:
```bash
pip install pandas spacy scikit-learn nltk
```

## Usage
1. Make sure you have the `netflix_titles.csv` file in the project directory.

2. Run the application:

```bash
python app.py
```

3. Enter your preferences for TV shows or movies in the GUI window.

4. Click the "Submit" button to get recommendations based on your preferences.

## Licence
This project is licensed under the [MIT License](LICENSE).
