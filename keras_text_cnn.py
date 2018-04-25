from keras.layers.embeddings import Embedding
from keras.layers.core import Reshape, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers import Input
from tensorflow.contrib import learn
import numpy as np

import load_data

embedding_dim = 128
filter_sizes = [3, 4, 5]
num_filters = 128
dropout_keep_prob = 0.5

num_epochs = 200


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
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

del x, y, x_shuffled, y_shuffled

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


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
    conv = Conv2D(filters=num_filters, kernel_size=(filter_len, embedding_dim), strides=(1, 1),
                  activation='relu', data_format='channels_first', padding='valid')
    # expected output shape = (samples?, num_filters, new_rows=max_doc_len - filter_len + 1, new_cols=1)
    conv_output = conv(sentence_vector)

    pooling = MaxPooling2D(pool_size=(max_document_length - filter_len + 1, 1), data_format='channels_first')
    # expected output (batch_size, num_filters, pooled_rows=1, pooled_cols=1)
    pool_output = pooling(conv_output)
    pool_outputs.append(pool_output)

# Concatenate the len(filter_sizes) outputs in only one
concatenated = Concatenate(axis=1)(pool_outputs)
# expected concatenated.shape = (batch_size, num_filters * len(filter_sizes), 1, 1)

feature_vector = Reshape((num_filters * len(filter_sizes),))(concatenated)
# expected feature_vector.shape = (batch_size, num_filters * len(filter_sizes))

classes = 2     # whether is a positive or negative review
final_output = Dense(classes, activation='softmax')(feature_vector)
# expected final_output.shape = (batch_size, classes)


model = Model(inputs=input_sentence, outputs=final_output)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)


# Evaluation
score = model.evaluate(x_dev, y_dev, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])