# Text Classification with SimpleRNN
Authored by Saeed Asle

# Description
  This project implements a text classification model using a SimpleRNN neural network in Python.
  It utilizes the tensorflow and keras libraries for model building and training, matplotlib for visualization,
  and nltk for natural language processing tasks.

The model is trained on the BBC News Dataset (bbc-text.csv) to classify news articles into five categories:
sport, business, politics, tech, and entertainment.
The dataset is preprocessed to remove stopwords and tokenized using the Tokenizer class from keras.preprocessing.text.

# Features
  * Data preprocessing: Removes stopwords and tokenizes text data.
  * Text tokenization: Uses Tokenizer to convert text data into sequences.
  * Model architecture: Utilizes a SimpleRNN neural network for text classification.
  * Training and evaluation: Trains the model on the dataset and evaluates its performance using accuracy metrics.
  * Prediction: Makes predictions on new text data to classify articles into categories.
# How to Use
Ensure you have the necessary libraries installed. You can install them using pip:

    pip install tensorflow numpy matplotlib nltk
Download NLTK stopwords if needed:

    import nltk
    nltk.download('stopwords')


