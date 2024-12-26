# Trend Analysis in Machining

This project performs topic modeling on research papers to analyze trends in machining. Using natural language processing (NLP) techniques, the project extracts insights from titles, abstracts, and keywords provided in the dataset.

---

## Getting Started

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/m-nawale/trend_analysis.git
2. Navigate to the project directory if not already in it:
    ```bash
    cd trend_analysis
3. Create and activate a virtual environment:
    ```bash
    python -m venv trend_env
    trend_env\Scripts\activate  # On Windows
    source trend_env/bin/activate  # On macOS/Linux
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

## Dataset
    The dataset consists of research papers related to machining. It includes the following columns:

    Title: The title of the paper.
    Authors: The authors of the paper.
    Abstract: The abstract of the paper.
    DOI: The DOI of the paper.
    Ensure the dataset file 19th_CIRP_Conference_Papers.xlsx is placed in the data/ folder before running the scripts.

## Project Workflow
    1.  Preprocessing
        The preprocessing.py script cleans and prepares the dataset for topic modeling. The following steps are performed:

        Convert all text to lowercase.
        Remove special characters, numbers, and punctuation.
        Tokenize the text into individual words.
        Remove stopwords (e.g., "the," "and," "is").
        Lemmatize words to their base form (e.g., "running" → "run").
        Output:

        A preprocessed dataset saved as preprocessed_data.xlsx in the data/ folder, with two new columns:
        Processed_Title
        Processed_Abstract
    2.  Topic Modeling
        The topic_modeling.py script uses the Latent Dirichlet Allocation (LDA) algorithm to extract topics from the preprocessed dataset.

        Steps:

        Combine the Processed_Title and Processed_Abstract columns into a single text field.
        Create a dictionary and corpus for the text data.
        Apply the LDA algorithm to extract topics.
        Output:

        Topics and their associated keywords displayed in the terminal.
        The trained LDA model and dictionary saved in the data/ folder.
    3.  Visualization
        The visualization.py script uses PyLDAvis to create an interactive visualization of the topics.

        Steps:

        Load the LDA model and dictionary.
        Generate a PyLDAvis visualization.
        Save the visualization as lda_visualization.html in the data/ folder.
        Output:

        An interactive HTML file (lda_visualization.html) that allows exploration of the topics.

## Project Structure
    trend_analysis/
    │
    ├── data/                      # Contains the input dataset and output files
    │   ├── 19th_CIRP_Conference_Papers.xlsx
    │   ├── preprocessed_data.xlsx
    │   ├── lda_model
    │   ├── dictionary
    │   ├── lda_visualization.html
    ├── notebooks/                 # Jupyter notebooks for exploratory analysis
    ├── src/                       # Source code for the project
    │   ├── preprocessing.py       # Text preprocessing
    │   ├── topic_modeling.py      # Topic modeling
    │   ├── visualization.py       # Visualization scripts
    ├── tests/                     # Unit tests for the scripts
    ├── README.md                  # Project documentation
    ├── requirements.txt           # List of dependencies
    └── main.py                    # Main script to run the project

## How to Run
    python main.py

## Where to see results
    File generated inside data directory, open this file in your browser.
    lda_visualization.html 

## Contact
    For questions or feedback, reach out at:

    Email: manoj.nawale.work@example.com
    GitLab Profile: mnawale
    GitHub Profile: m-nawale
