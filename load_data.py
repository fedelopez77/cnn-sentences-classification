
import re
import numpy as np
import io

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)  # replace everything that is not a number, letter or a few chars for a white space
    string = re.sub(r"\'s", " \'s", string)     # adds a space for "'s". Example: that's -> that 's
    string = re.sub(r"\'ve", " \'ve", string)   # adds a space for "'ve". Example: you've -> you 've
    string = re.sub(r"n\'t", " n\'t", string)   # adds a space for "n't". Example: can't -> ca n't
    string = re.sub(r"\'re", " \'re", string)   # adds a space for "'re". Example: you're -> you 're
    string = re.sub(r"\'d", " \'d", string)     # adds a space for "'d". Example: you'd -> you 'd
    string = re.sub(r"\'ll", " \'ll", string)   # adds a space for "'ll". Example: you'll -> you 'll
    string = re.sub(r",", " , ", string)        # adds a space for ",". Example: you, me -> you , me
    string = re.sub(r"!", " ! ", string)        # adds a space for "!". Example: not! -> not !
    string = re.sub(r"\(", " \( ", string)      # adds a slash and space for "(". Example: and) -> and \)
    string = re.sub(r"\)", " \) ", string)      # adds a slash and space for ")". Example: (and -> \( and
    string = re.sub(r"\?", " \? ", string)      # adds a slash and space for "?". Example: and? -> and \?
    string = re.sub(r"\s{2,}", " ", string)     # Replace 2 or more whitespaces for only one
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    with io.open(positive_data_file, encoding='latin-1') as positive_file:
        positive_examples = [str(s).strip() for s in positive_file.readlines()]

    with io.open(negative_data_file, encoding='latin-1') as negative_file:
        negative_examples = [str(s).strip() for s in negative_file.readlines()]

    # Split by words
    x_text = [clean_str(sent) for sent in positive_examples + negative_examples]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
