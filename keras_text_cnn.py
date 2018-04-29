from keras.layers.embeddings import Embedding
from keras.layers.core import Reshape, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers import Input
from keras import regularizers
from keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.contrib import learn
import numpy as np

import load_data


# Load data
print("Loading data...")
x_text, y = load_data.load_data_and_labels("datasets/rt-polarity.pos", "datasets/rt-polarity.neg")


# Taken from http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(len(y) * 0.1)  # Uses 10% as test (dev)
x_train, x_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

del x, y, x_shuffled, y_shuffled

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/test split: {:d}/{:d}".format(len(y_train), len(y_test)))


def get_model(embedding_dim=128, filter_sizes=(3, 4, 5), num_filters=128, dropout=0.2, l1_reg=0.01, l2_reg=0.01):

    # Embedding layer
    embedding = Embedding(input_dim=len(vocab_processor.vocabulary_),
                          output_dim=embedding_dim,
                          input_length=max_document_length,
                          name="embedding")

    input_sentence = Input(shape=(max_document_length,), name="input_sentence")

    sentence_vector = embedding(input_sentence)
    # expected sentence_vector.shape = (batch_size, max_doc_length, embedding_dim)

    # This is necessary because Conv2D expects a 4-D tensor (counting the batch_size)
    sentence_vector = Reshape((1, max_document_length, embedding_dim))(sentence_vector)


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
        pooling = MaxPooling2D(pool_size=(max_document_length - filter_len + 1, 1), data_format='channels_first',
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

    classes = 2     # positive o negative
    final_output = Dense(classes, activation='softmax',
                         kernel_regularizer=regularizers.l2(l2_reg), activity_regularizer=regularizers.l1(l1_reg),
                         name="fully_connected")(feature_vector)  # 2 because it can be positive or negative
    # expected final_output.shape = (batch_size, 2)

    model = Model(inputs=input_sentence, outputs=final_output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def grid_search():
    best_accuracy = 0.0
    best_loss = 0.0

    best_embed = 0
    best_num_filters = 0
    best_dropout = 0
    best_l1 = 0
    best_l2 = 0

    embedding_dims = [300, 400, 500]
    nums_of_filters = [300, 400, 500]
    dropouts = [0.1, 0.2, 0.3, 0.4]
    l1s = [0.0001]
    l2s = [0.0001]
    for embedding_dim in embedding_dims:
        for num_filters in nums_of_filters:
            for dropout in dropouts:
                for l1 in l1s:
                    for l2 in l2s:
                        model = get_model(embedding_dim=embedding_dim, num_filters=num_filters, dropout=dropout,
                                          l1_reg=l1, l2_reg=l2)

                        # Training
                        model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=2, validation_split=0.1,
                                  callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1)])

                        # Evaluation
                        score = model.evaluate(x_test, y_test, verbose=1)
                        loss, accuracy = score[0], score[1]

                        print("-- Partial Result: Accuracy: {}, Loss: {}".format(accuracy, loss))
                        print("Parameters: embed: {}, num_filters: {}, dropout: {}, l1: {}, l2: {}".format(
                            embedding_dim, num_filters, dropout, l1, l2))

                        if accuracy > best_accuracy:
                            print("-------- NEW BEST RESULT --------")
                            print("previous acc: {}, new accuracy: {}, Loss: {}".format(best_accuracy, accuracy, loss))
                            best_accuracy = accuracy
                            best_loss = loss
                            best_embed = embedding_dim
                            best_num_filters = num_filters
                            best_dropout = dropout
                            best_l1 = l1
                            best_l2 = l2

    print("FINAL RESULTS: Best accuracy: {}, best loss: {}".format(best_accuracy, best_loss))
    print("Best Embedding: {}\nBest num filter: {}\nBest dropout: {}\nBest L1: {}\nBest L2: {}".format(
            best_embed, best_num_filters, best_dropout, best_l1, best_l2))


grid_search()
