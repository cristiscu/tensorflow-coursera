# Classifier on BBC News Archive
# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W2/assignment/C3W2_Assignment.ipynb

import io
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

NUM_WORDS = 1000
EMBEDDING_DIM = 16
MAXLEN = 120
PADDING = 'post'
OOV_TOKEN = "<OOV>"
TRAINING_SPLIT = .8


def remove_stopwords(sentence):
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                 "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's",
                 "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only",
                 "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
                 "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
                 "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
                 "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
                 "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
                 "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"]
    sentence = sentence.lower()
    return ' '.join([w for w in sentence.split() if w not in stopwords])


def parse_data_from_file(filename):
    sentences = []
    labels = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[0])
            sentences.append(remove_stopwords(row[1]))
    return sentences, labels


sentences, labels = parse_data_from_file("../data/bbc-text.csv")

# split training/validation sets
train_size = int(len(sentences) * TRAINING_SPLIT)

train_sentences = sentences[:train_size]
train_labels = labels[:train_size]

val_sentences = sentences[train_size:]
val_labels = labels[train_size:]


# generate tokens, padded sequences, tokenized labels
def fit_tokenizer(sentences, num_words, oov_token):
    tokenizer = Tokenizer(num_words, oov_token)
    tokenizer.fit_on_texts(sentences)
    return tokenizer


tokenizer = fit_tokenizer(train_sentences, num_words=NUM_WORDS, oov_token=OOV_TOKEN)
word_index = tokenizer.word_index


def seq_and_pad(sentences, tokenizer, maxlen, padding):
    sequences = tokenizer.texts_to_sequences(sentences)
    return pad_sequences(sequences, maxlen=maxlen, padding=padding)


train_padded_seq = seq_and_pad(train_sentences, tokenizer, maxlen=MAXLEN, padding=PADDING)
val_padded_seq = seq_and_pad(val_sentences, tokenizer, maxlen=MAXLEN, padding=PADDING)


def tokenize_labels(all_labels, split_labels):
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(all_labels)
    label_seq = label_tokenizer.texts_to_sequences(split_labels)
    label_seq_np = np.array(label_seq) - 1
    return label_seq_np


train_label_seq = tokenize_labels(labels, train_labels)
val_label_seq = tokenize_labels(labels, val_labels)


# build+train model
tf.random.set_seed(123)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAXLEN),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_padded_seq, train_label_seq,
                    epochs=30, validation_data=(val_padded_seq, val_label_seq))


# check training/validation accuracy
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


# visualize 3D vectors, w/ generated files for embedding visualization
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
weights = model.layers[0].get_weights()[0]

out_v = io.open('../data/vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('../data/meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, NUM_WORDS):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

# save model in H5 format, to submit for exam
model.save('../saved_models/C3W2_Assignment.h5')
