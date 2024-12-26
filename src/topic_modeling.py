import os
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel


def load_preprocessed_data():
    """Load the preprocessed dataset."""
    # Dynamically resolve the base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'preprocessed_data.xlsx')

    # Check if the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The preprocessed data file was not found at {data_path}. Please run preprocessing first.")

    # Load the dataset
    df = pd.read_excel(data_path)
    return df['Processed_Title'] + " " + df['Processed_Abstract']


def prepare_corpus(texts):
    """Prepare dictionary and corpus for LDA model."""
    # Tokenize the text
    tokenized_texts = [text.split() for text in texts]

    # Create a dictionary
    dictionary = Dictionary(tokenized_texts)

    # Filter out extreme cases (rare and common words)
    dictionary.filter_extremes(no_below=2, no_above=0.5)

    # Create a corpus (bag of words representation)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    return dictionary, corpus


def build_lda_model(corpus, dictionary, num_topics=10):
    """Build and train the LDA model."""
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    return lda_model


def summarize_topics(lda_model, num_words=5):
    """Generate summaries for each topic based on the top keywords."""
    topic_summaries = {}
    for idx, topic in lda_model.show_topics(formatted=False, num_words=num_words):
        keywords = [word for word, _ in topic]
        topic_summaries[f"Topic {idx}"] = " / ".join(keywords)
    return topic_summaries


def main():
    # Dynamically resolve the base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(BASE_DIR, 'data')

    # Load the preprocessed data
    print("Loading preprocessed data...")
    texts = load_preprocessed_data()

    # Prepare the dictionary and corpus
    print("Preparing dictionary and corpus...")
    dictionary, corpus = prepare_corpus(texts)

    # Build the LDA model
    print("Building the LDA model...")
    lda_model = build_lda_model(corpus, dictionary)

    # Generate topic summaries
    print("\nSummarizing Topics...")
    topic_summaries = summarize_topics(lda_model)

    for topic, summary in topic_summaries.items():
        print(f"{topic}: {summary}")

    # Save the model and dictionary
    model_path = os.path.join(output_dir, 'lda_model')
    dictionary_path = os.path.join(output_dir, 'dictionary')
    print(f"\nSaving the LDA model and dictionary...")
    lda_model.save(model_path)
    dictionary.save(dictionary_path)
    print(f"LDA model and dictionary saved to {model_path} and {dictionary_path}")

if __name__ == "__main__":
    main()