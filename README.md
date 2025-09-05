# Fake News Classifier 📰

A deep learning project to classify news articles as real or fake based on their headlines. This repository contains the code and resources for building and evaluating several Recurrent Neural Network (RNN) models, including LSTM, Bidirectional LSTM, and GRU, for text classification.

#🎯 Project Overview

The spread of misinformation is a significant challenge in the digital age. This project aims to tackle this issue by creating a robust classifier that can distinguish between legitimate and fabricated news headlines. The models are trained on the "Fake and Real News Dataset" from Kaggle and achieve an accuracy of approximately 96% in identifying fake news.

#💾 Dataset

The project utilizes the Fake and Real News Dataset sourced from Kaggle.

###Source:
Kaggle Fake and Real News Dataset

###Content:
The dataset consists of two CSV files:

###True.csv:
Contains over 21,000 real news articles.

###Fake.csv:
Contains over 23,000 fake news articles.

###Features:
Each article includes a title, text, subject, and date. For this project, only the title is used for classification.

#⚙️ Project Workflow

The project follows a standard machine learning pipeline:

Data Loading & Integration: The Fake.csv and True.csv files are loaded into Pandas DataFrames. A label column is added (0 for real news, 1 for fake news), and the datasets are merged.

##Data Cleaning:

The combined dataset is shuffled to ensure random distribution.

Duplicate entries are identified and removed to maintain data quality.

Text Preprocessing: The news titles undergo a series of natural language processing (NLP) steps:

Removal of non-alphabetic characters.

Conversion to lowercase.

Tokenization (splitting text into individual words).

Removal of common English stopwords (e.g., "the", "a", "is").

Stemming using NLTK's PorterStemmer to reduce words to their root form (e.g., "committing" -> "commit").

##Vectorization:

Tokenization: The preprocessed text corpus is converted into sequences of integers using tf.keras.preprocessing.text.Tokenizer.

Padding: All sequences are padded to a uniform length to be used as input for the models.

Model Building & Training: Three different RNN architectures were built, trained, and evaluated:

A standard LSTM network.

A Bidirectional LSTM network.

A Bidirectional GRU network.

Evaluation: The models are evaluated on a held-out test set using accuracy, confusion matrix, and a detailed classification report (precision, recall, F1-score).

#📊 Models and Results

Three deep learning models were implemented to compare their performance on the classification task. All models demonstrated high accuracy.

#Model	Accuracy

LSTM	95.77%
Bidirectional LSTM	95.83%
Bidirectional GRU	95.82%


#🛠️ Technologies Used

###Programming Language:
Python 3

###Libraries:
TensorFlow & Keras, Scikit-learn, Pandas, NumPy, NLTK (Natural Language Toolkit), Matplotlib (for visualization, if added)

#🚀 How to Run this Project

To replicate this project on your local machine, follow these steps:

###Clone the repository:

Bash

git clone https://github.com/your-username/FakeNewsClassifier.git
cd FakeNewsClassifier
Set up a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required libraries:


pip install -r requirements.txt
(Note: You can create a requirements.txt file by running pip freeze > requirements.txt in your project's environment.)

###Set up Kaggle API Credentials:

Download your kaggle.json API token from your Kaggle account page.

Place the kaggle.json file in the root directory of this project. The notebook is configured to move it to the correct location (~/.kaggle/).

###Run the Jupyter Notebook:

jupyter notebook FakeNewsClassifier.ipynb
Execute the cells in order to download the data, preprocess it, and train the models.
