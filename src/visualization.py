import os
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary


def load_model_and_dictionary():
    """Load the LDA model and dictionary."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'data', 'lda_model')
    dictionary_path = os.path.join(BASE_DIR, 'data', 'dictionary')

    lda_model = LdaModel.load(model_path)
    dictionary = Dictionary.load(dictionary_path)

    return lda_model, dictionary


def prepare_visualization(lda_model, dictionary):
    """Prepare data for PyLDAvis."""
    corpus = [dictionary.doc2bow(text.split()) for text in dictionary.values()]
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    return vis_data


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
    data_dir = os.path.join(BASE_DIR, 'data')

    # Load the LDA model and dictionary
    print("Loading the LDA model and dictionary...")
    lda_model, dictionary = load_model_and_dictionary()

    # Generate topic summaries
    print("Generating topic summaries...")
    topic_summaries = summarize_topics(lda_model)

    # Display topic summaries
    print("\nTopic Summaries:")
    for topic, summary in topic_summaries.items():
        print(f"{topic}: {summary}")

    # Load the corpus
    preprocessed_data_path = os.path.join(data_dir, 'preprocessed_data.xlsx')
    df = pd.read_excel(preprocessed_data_path)
    tokenized_text = [text.split() for text in df['Processed_Title'] + ' ' + df['Processed_Abstract']]
    corpus = [dictionary.doc2bow(text) for text in tokenized_text]

    # Generate the visualization
    print("Generating visualization...")
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

    # Save the visualization
    vis_path = os.path.join(data_dir, 'lda_visualization_2.html')
    pyLDAvis.save_html(vis, vis_path)

    print(f"\nVisualization saved to {vis_path}")
    print("Open the HTML file in a browser to explore the topics interactively.")

if __name__ == "__main__":
    main()