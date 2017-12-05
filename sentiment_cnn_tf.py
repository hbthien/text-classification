# based on ideas from https://github.com/dennybritz/cnn-text-classification-tf
# and https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py


import numpy as np
import re
import itertools
from collections import Counter
from urllib.request import urlopen

########################## Preprocessing #######################

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(pos_fn, neg_fn):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
#     # Pull sentences with positive sentiment
    pos_file = urlopen(pos_fn)
 
    # Pull sentences with negative sentiment
    neg_file = urlopen(neg_fn)

    # Load data from files
    positive_examples = list(pos_file.readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(neg_file.readlines())
    negative_examples = [s.strip() for s in negative_examples]
    
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent.decode('latin1')) for sent in x_text] # or:     x_text = [clean_str(str(sent)) for sent in x_text]

    x_text = [s.split(" ") for s in x_text]
    
    # Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


############## Evaluation ###############################################

import tensorflow as tf
from tensorflow.contrib import learn
import os, csv


#data parameters
tf.flags.DEFINE_string("pos_file", "https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.pos", "pos sentiment file")
tf.flags.DEFINE_string("neg_file", "https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.neg", "neg sentiment file")


# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(attr.upper(), value)


if FLAGS.eval_train:
    X_train, y_train = load_data_and_labels(FLAGS.pos_file, FLAGS.neg_file)
    print(y_train.shape)
else:
    X_train, y_train = ["a masterpiece four years in the making", "everything is off."]
    y_train = [1, 0]


# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(X_train)))

print("\nEvaluating...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir) #Find the checkpoints
# graph = tf.
print(tf.get_default_graph())


