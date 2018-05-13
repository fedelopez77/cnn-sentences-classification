"""
Preprocessing done with
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
"""


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers.core import Reshape, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers import Input
from keras import regularizers
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.preprocessing import LabelBinarizer

import numpy as np
np.random.seed(1234)


MAX_DOC_LENGTH = 0
VOCABULARY_SIZE = 0
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1


def load_data(file_name):
    messages = []
    labels = []
    with open(file_name, 'r') as f:
        separator = ":"
        for line in f:
            sep_index = line.find(separator)
            user_name = line[:sep_index]
            message = line.rstrip()[sep_index + 1:]

            messages.append(message)
            labels.append(user_name)

    return messages, labels


def preprocess_data(messages, labels):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(messages)

    global MAX_DOC_LENGTH, VOCABULARY_SIZE
    MAX_DOC_LENGTH = max([len(message.split()) for message in messages])
    VOCABULARY_SIZE = len(tokenizer.word_index) + 1

    data = pad_sequences(tokenizer.texts_to_sequences(messages))

    encoder = LabelBinarizer()
    cat_labels = encoder.fit_transform(labels)

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', cat_labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    cat_labels = cat_labels[indices]
    test_proportion = int((1.0 - TEST_SPLIT) * data.shape[0])

    x_train = data[:test_proportion]
    y_train = cat_labels[:test_proportion]
    x_test = data[test_proportion:]
    y_test = cat_labels[test_proportion:]

    return x_train, y_train, x_test, y_test


def get_model(embedding_dim=128, filter_sizes=(3, 4, 5), num_filters=128, dropout=0.2, l1_reg=0.01, l2_reg=0.01,
              classes=3):

    # Embedding layer
    embedding = Embedding(input_dim=VOCABULARY_SIZE,
                          output_dim=embedding_dim,
                          input_length=MAX_DOC_LENGTH,
                          name="embedding")

    input_sentence = Input(shape=(MAX_DOC_LENGTH,), name="input_sentence")

    sentence_vector = embedding(input_sentence)
    # expected sentence_vector.shape = (batch_size, max_doc_length, embedding_dim)

    # This is necessary because Conv2D expects a 4-D tensor (counting the batch_size)
    sentence_vector = Reshape((1, MAX_DOC_LENGTH, embedding_dim))(sentence_vector)

    # 3 Conv2D layers, with num_filters (128) of filters size = (filter_len=[3,4,5], output_dim)
    # each filter produces an output of expected shape (max_doc_len - filter_len + 1)
    # the input of each Conv2D layer is the same sentence_vector
    pool_outputs = []

    for filter_len in filter_sizes:

        conv_name = "Conv2D_{}".format(filter_len)
        conv = Conv2D(filters=num_filters, kernel_size=(filter_len, embedding_dim), strides=(1, 1),
                      activation='relu', data_format='channels_first', padding='valid',
                      kernel_regularizer=regularizers.l2(l2_reg), activity_regularizer=regularizers.l1(l1_reg),
                      name=conv_name)
        # expected output shape = (samples?, num_filters, new_rows=max_doc_len - filter_len + 1, new_cols=1)

        conv_output = conv(sentence_vector)

        max_pool_name = "MaxPool_{}".format(filter_len)
        pooling = MaxPooling2D(pool_size=(MAX_DOC_LENGTH - filter_len + 1, 1), data_format='channels_first',
                               name=max_pool_name)
        # expected output (batch_size, num_filters, pooled_rows=1, pooled_cols=1)

        pool_output = pooling(conv_output)
        pool_outputs.append(pool_output)

    # Concatenate the len(filter_sizes) outputs in only one
    concatenated = Concatenate(axis=1)(pool_outputs)
    # expected concatenated.shape = (batch_size, num_filters * len(filter_sizes), 1, 1)

    feature_vector = Reshape((num_filters * len(filter_sizes),))(concatenated)
    # expected feature_vector.shape = (batch_size, num_filters * len(filter_sizes))

    feature_vector = Dropout(dropout, seed=123)(feature_vector)

    final_output = Dense(classes, activation='softmax',
                         kernel_regularizer=regularizers.l2(l2_reg), activity_regularizer=regularizers.l1(l1_reg),
                         name="fully_connected")(feature_vector)  # 2 because it can be positive or negative
    # expected final_output.shape = (batch_size, classes)

    model = Model(inputs=input_sentence, outputs=final_output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def test_model():
    file_name = "datasets/chat.txt"
    messages, labels = load_data(file_name)

    x_train, y_train, x_test, y_test = preprocess_data(messages, labels)

    embedding_dim = 128
    num_filters = 150
    dropout = 0.1
    l1 = l2 = 0.0001

    model = get_model(embedding_dim=embedding_dim, num_filters=num_filters, dropout=dropout,
                      l1_reg=l1, l2_reg=l2, classes=y_train.shape[1])

    # Training
    model.fit(x_train, y_train, batch_size=1024, epochs=50, verbose=2, validation_split=0.1)
              # callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1)])

    # Evaluation
    score = model.evaluate(x_test, y_test, verbose=1)
    loss, accuracy = score[0], score[1]

    print("-- RESULT: Accuracy: {}, Loss: {}".format(accuracy, loss))
    print("Parameters: embed: {}, num_filters: {}, dropout: {}, l1: {}, l2: {}".format(
        embedding_dim, num_filters, dropout, l1, l2))


test_model()


# def grid_search():
#     best_accuracy = 0.0
#     best_loss = 0.0
#
#     best_embed = 0
#     best_num_filters = 0
#     best_dropout = 0
#     best_l1 = 0
#     best_l2 = 0
#
#     embedding_dims = [300, 400, 500]
#     nums_of_filters = [300, 400, 500]
#     dropouts = [0.1, 0.2, 0.3, 0.4]
#     l1s = [0.0001]
#     l2s = [0.0001]
#     for embedding_dim in embedding_dims:
#         for num_filters in nums_of_filters:
#             for dropout in dropouts:
#                 for l1 in l1s:
#                     for l2 in l2s:
#                         model = get_model(embedding_dim=embedding_dim, num_filters=num_filters, dropout=dropout,
#                                           l1_reg=l1, l2_reg=l2)
#
#                         # Training
#                         model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=2, validation_split=0.1,
#                                   callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1)])
#
#                         # Evaluation
#                         score = model.evaluate(x_test, y_test, verbose=1)
#                         loss, accuracy = score[0], score[1]
#
#                         print("-- Partial Result: Accuracy: {}, Loss: {}".format(accuracy, loss))
#                         print("Parameters: embed: {}, num_filters: {}, dropout: {}, l1: {}, l2: {}".format(
#                             embedding_dim, num_filters, dropout, l1, l2))
#
#                         if accuracy > best_accuracy:
#                             print("-------- NEW BEST RESULT --------")
#                             print("previous acc: {}, new accuracy: {}, Loss: {}".format(best_accuracy, accuracy, loss))
#                             best_accuracy = accuracy
#                             best_loss = loss
#                             best_embed = embedding_dim
#                             best_num_filters = num_filters
#                             best_dropout = dropout
#                             best_l1 = l1
#                             best_l2 = l2
#
#     print("FINAL RESULTS: Best accuracy: {}, best loss: {}".format(best_accuracy, best_loss))
#     print("Best Embedding: {}\nBest num filter: {}\nBest dropout: {}\nBest L1: {}\nBest L2: {}".format(
#             best_embed, best_num_filters, best_dropout, best_l1, best_l2))