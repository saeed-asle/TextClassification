import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN, Dropout
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if needed
# nltk.download('stopwords')

# Define constants and hyperparameters
STOPWORDS = set(stopwords.words('english'))
vocab_size = 5000
embedding_dim = 64
max_length = 200
oov_tok = '<OOV>'  # Out of Vocabulary
training_portion = 0.8

# Initialize data containers
articles = []
labels = []

# Load data from the CSV file
with open('bbc-text.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        article = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        articles.append(article)

# Split data into training and validation sets
train_size = int(len(articles) * training_portion)
train_articles = articles[:train_size]
train_labels = labels[:train_size]
validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

# Tokenize and preprocess text data
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_articles)
train_padded = pad_sequences(train_sequences, maxlen=max_length)

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length)

# Tokenize labels
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

# Build the model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Dropout(0.2))
model.add(SimpleRNN(units=embedding_dim))
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))
model.summary()

# Compile the model
opt = tf.keras.optimizers.legacy.Adam(lr=0.001, decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
num_epochs = 20
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

# Visualize training results
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Inference
txt1 = ["blair prepares to name poll date tony blair is likely to name 5 may as election day when parliament returns from its easter break  the bbc s political editor has learned.  andrew marr says mr blair will ask the queen on 4 or 5 april to dissolve parliament at the end of that week. mr blair has so far resisted calls for him to name the day but all parties have stepped up campaigning recently. downing street would not be drawn on the claim  saying election timing was a matter for the prime minister.  a number 10 spokeswoman would only say:  he will announce an election when he wants to announce an election.  the move will signal a frantic week at westminster as the government is likely to try to get key legislation through parliament. the government needs its finance bill  covering the budget plans  to be passed before the commons closes for business at the end of the session on 7 april.  but it will also seek to push through its serious and organised crime bill and id cards bill. mr marr said on wednesday s today programme:  there s almost nobody at a senior level inside the government or in parliament itself who doesn t expect the election to be called on 4 or 5 april.  as soon as the commons is back after the short easter recess  tony blair whips up to the palace  asks the queen to dissolve parliament ... and we re going.  the labour government officially has until june 2006 to hold general election  but in recent years governments have favoured four-year terms."]
txt2 = ["call to save manufacturing jobs the trades union congress (tuc) is calling on the government to stem job losses in manufacturing firms by reviewing the help it gives companies.  the tuc said in its submission before the budget that action is needed because of 105,000 jobs lost from the sector over the last year. it calls for better pensions, child care provision and decent wages. the 36-page submission also urges the government to examine support other European countries provide to industry. tuc general secretary brendan barber called for a commitment to policies that will make a real difference to the lives of working people. greater investment in childcare strategies and the people delivering that childcare will increase the options available to working parents. a commitment to our public services and manufacturing sector ensures that we can continue to compete on a global level and deliver the frontline services that this country needs. he also called for practical measures to help pensioners, especially women who he said are most likely to retire in poverty. the submission also calls for decent wages and training for people working in the manufacturing sector."]

# Tokenize and pad the inference text
seq1 = tokenizer.texts_to_sequences(txt1)
padded1 = pad_sequences(seq1, maxlen=max_length)
seq2 = tokenizer.texts_to_sequences(txt2)
padded2 = pad_sequences(seq2, maxlen=max_length)

# Make predictions
pred1 = model.predict(padded1)
pred2 = model.predict(padded2)

labels = ['sport', 'business', 'politics', 'tech', 'entertainment']

# Print predictions for the first article
print("Predictions for the first article:")
print(pred1)
print("Predicted category:", labels[np.argmax(pred1) - 1])

# Print predictions for the second article
print("Predictions for the second article:")
print(pred2)
print("Predicted category:", labels[np.argmax(pred2) - 1])
