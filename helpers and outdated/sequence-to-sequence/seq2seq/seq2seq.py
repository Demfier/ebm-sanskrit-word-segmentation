import time
start_time = time.time()
input_lines = open('input_refined.txt').readlines()
output_lines = open('output_refined.txt').readlines()


#=================DATA CLEANING AND LOADING PART: START========================#
print "\n\n\nStarting Data cleaning and loading...........\n\n"

# Just see the top 5 input and output lines
print '_______________First 5 input lines_____________'
for _ in input_lines[:5]:
    print _

print '_______________First 5 output lines_______________'
for _ in output_lines[:5]:
    print _

print 'Number of input and out put lines resp____________', len(input_lines), len(output_lines)

# refine the lines a bit
import re
output_lines = [x for x, y in zip(output_lines, input_lines)
                if re.match("^[A-Za-z0-9_-]*$", y.replace(" ", "").replace("'", ""))]
input_lines = [x for x in input_lines
               if re.match("^[A-Za-z0-9_-]*$", x.replace(" ", "").replace("'", ""))]
print 'After filtering once____________', len(input_lines), len(output_lines)

input_lines = [x.replace('\n', '') for x, y in zip(input_lines, output_lines)
               if re.match("^[A-Za-z0-9_-]*$", y.replace(" ", "").replace("'", ""))]
output_lines = [x.replace('\n', '') for x in output_lines
                if re.match("^[A-Za-z0-9_-]*$", x.replace(" ", "").replace("'", ""))]
print 'After filtering twice____________', len(input_lines), len(output_lines)

# We get dictionaries of input characters
input_chars = set()
for line in input_lines:
    for letter in line.replace('\n', ''):
        input_chars.add(letter)
# print "input_chars_________________________"
# print list(input_chars)

output_chars = set()
for line in output_lines:
    for letter in line:
        output_chars.add(letter)
# print "output_chars_________________________"
# print list(output_chars)

import string

char_set = string.lowercase + string.uppercase + string.digits + ' ' + "'" + '_'

index_to_letter = dict(enumerate(char_set))
letter_to_index = dict((v, k) for k, v in index_to_letter.items())
# print "index to letter dictionary_________________________\n", index_to_letter
# print "letter to index dictionary_________________________\n", letter_to_index

# Biggest word in dictionary

# print len(input_lines)
# print len([x for x in input_lines if len(x) < 50])
# print len([x for x in input_lines if len(x) < 100])
# print len(output_lines)
# print len([x for x in output_lines if len(x) < 50])
# print len([x for x in output_lines if len(x) < 100])
# print len([x for x, y in zip(input_lines, output_lines) if (len(x) <= 50 and len(y) <= 50)])
# print len([x for x, y in zip(output_lines, input_lines) if (len(x) <= 50 and len(y) <= 50)])


input_lines_temp = [x for x, y in zip(input_lines, output_lines)
                    if (len(x) <= 50 and len(y) <= 50)]
output_lines_temp = [x for x, y in zip(output_lines, input_lines)
                     if (len(x) <= 50 and len(y) <= 50)]

input_lines = input_lines_temp
output_lines = output_lines_temp

print "Number of input & output lines with sentences <=50 characters_________________", len(input_lines), len(output_lines)

# WE got rid of words that are too long or have puntuations between them

# Shuffle the order of input and output lines' pair
import random

c = list(zip(input_lines, output_lines))
random.shuffle(c)
# c is a of type list
input_lines, output_lines = zip(*c)

import numpy as np

input_ = np.zeros((len(input_lines), 50))
labels_ = np.zeros((len(output_lines), 50))

for i, (inp, out) in enumerate(c):
    inp = inp + '_' * (50 - len(inp))
    out = out + '_' * (50 - len(out))

    for j, letter in enumerate(inp):
        input_[i][j] = letter_to_index[letter]
    for j, letter in enumerate(out):
        labels_[i][j] = letter_to_index[letter]

input_ = input_.astype(np.int32)
labels_ = labels_.astype(np.int32)
print "input matrix dimension____________________________________", input_.shape
print "Example: an encoded input sentence_________________________\n", input_[0]

# Data division for testing, evaluation and training

test_input = input_[:3000]
eval_input = input_[3000:6000]
train_input = input_[6000:]
test_labels = labels_[:3000]
eval_labels = labels_[3000:6000]
train_labels = labels_[6000:]

print test_input[:10], 'test input _________'

# Finally these data are gonna be used furthur in the model
test_data = zip(test_input, test_labels)
eval_data = zip(eval_input, eval_labels)
train_data = zip(train_input, train_labels)

print "Training Data Matrix dimensions________________________", train_input.shape, train_labels.shape
data_time = time.time()
print "Data Cleaning and loading part finished. Took %s seconds..\n\n\n" % (data_time - start_time)

#=================DATA CLEANING AND LOADING PART: END==========================#


#=================DECLARE SEQUENC TO SEQUENCE MODEL PART: START================#

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn_cell, seq2seq
tf_time = time.time()
print "Imported tensorflow.Took %s seconds..\n\n" % (tf_time - data_time)

# Reset all current graphs and sessions and start a new one.
ops.reset_default_graph()
try:
    sess.close()
except:
    pass
sess = tf.Session()  # Run a normal session
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Declare model parameters
input_seq_length = 50
ouput_seq_length = 50
batch_size = 128

input_vocab_size = 65
output_vocab_size = 65
embedding_dim = 256

# Model elements
encode_input = [tf.placeholder(tf.int32,
                shape=(None,),
                name="ei_%i" % i) for i in range(input_seq_length)]

labels = [tf.placeholder(tf.int32,
          shape=(None,),
          name="di_%i" % i) for i in range(ouput_seq_length)]

# labels[:-1] will always be an '_', so no need of putting it in decode_input
decode_input = [tf.zeros_like(encode_input[0], dtype=np.int32, name="GO")] + labels[:-1]


keep_prob = tf.placeholder("float")
# Define Model Layers
cells = [rnn_cell.DropoutWrapper(rnn_cell.BasicLSTMCell(embedding_dim), output_keep_prob=keep_prob) for i in range(4)]

stacked_lstm = rnn_cell.MultiRNNCell(cells)

with tf.variable_scope("decoders") as scope:
    decode_outputs, decode_state = seq2seq.embedding_rnn_seq2seq(
        encode_input, decode_input, stacked_lstm, input_vocab_size, output_vocab_size, 1)

    scope.reuse_variables()

    decode_outputs_test, decode_state_test = seq2seq.embedding_rnn_seq2seq(
        encode_input, decode_input, stacked_lstm, input_vocab_size, output_vocab_size, 1, feed_previous=True)

# Model loss optimizers
loss_weights = [tf.ones_like(l, dtype=tf.float32) for l in labels]
loss = seq2seq.sequence_loss(decode_outputs, labels, loss_weights, output_vocab_size)
optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss)

# sess.run(tf.initialize_all_variables())  # Deprecated
sess.run(tf.global_variables_initializer())

sess_time = time.time()
print "Seq2Seq model formation finished. Took %s seconds to make the session. The session is active now..\n\n\n" % (sess_time - data_time)

#=================DECLARE SEQUENC TO SEQUENCE MODEL PART: END==================#

#=================TRAINING MODEL PART: START==================#


class DataIterator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.iter = self.make_random_iter()

    def next_batch(self):
        try:
            idxs = self.iter.next()
        except StopIteration:
            self.iter = self.make_random_iter()
            idxs = self.iter.next()

        X, Y = zip(*[self.data[i] for i in idxs])
        X = np.array(X).T
        Y = np.array(Y).T
        return X, Y

    def make_random_iter(self):
        splits = np.arange(self.batch_size, len(self.data), self.batch_size)  # gives evenly spaced array with 'self.batch_size' interval
        it = np.split(np.random.permutation(range(len(self.data))), splits[:-1])
        return iter(it)

train_iter = DataIterator(train_data, 128)
val_iter = DataIterator(eval_data, 128)
test_iter = DataIterator(test_data, 128)

import sys


def get_feed(X, Y):
    feed_dict = {encode_input[t]: X[t] for t in range(input_seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(input_seq_length)})
    return feed_dict


def train_batch(data_iter):
    X, Y = data_iter.next_batch()
    feed_dict = get_feed(X, Y)
    feed_dict[keep_prob] = 0.5
    _, out = sess.run([train_op, loss], feed_dict)
    return out


def get_eval_batch_data(data_iter):
    X, Y = data_iter.next_batch()
    feed_dict = get_feed(X, Y)
    feed_dict[keep_prob] = 1
    all_output = sess.run([loss] + decode_outputs_test, feed_dict)
    eval_loss = all_output[0]
    decode_output = np.array(all_output[1:]).transpose(1, 0, 2)
    return eval_loss, decode_output, X, Y


def eval_batch(data_iter, num_batches):
    losses = []
    predict_loss = []
    for i in range(num_batches):
        eval_loss, output, X, Y = get_eval_batch_data(data_iter)
        losses.append(eval_loss)

        for index in range(len(output)):
            real = Y.T[index]
            predict = np.argmax(output, axis=2)[index]
            predict_loss.append(all(real == predict))
    return np.mean(losses), np.mean(predict_loss)


saver = tf.train.Saver()

iter_time = time.time()
for i in range(10000):
    try:
        print "In %dth iteration.." % i
        print "Took %s seconds" % (time.time() - iter_time)
        train_batch(train_iter)
        if i % 1000 == 0:
            val_loss, val_predict = eval_batch(val_iter, 16)
            train_loss, train_predict = eval_batch(train_iter, 16)
            print "val loss   : %f, val predict   = %.1f%%" % (val_loss, val_predict * 100)
            print "train loss : %f, train predict = %.1f%%" % (train_loss, train_predict * 100)
            print
            sys.stdout.flush()

            saver.save(sess, "skt.ckpt")  # Saving the model to skt.ckpt file

    except KeyboardInterrupt:
        print "interrupted by user"
        break

training_time = time.time()
print "Classes to train model declared. Took %s seconds.\n\n" % (training_time - sess_time)
#=================TRAINING MODEL PART: END==================#

#==================EXAMINING MODEL PART: START==============================#

# If you wanna do things more "tf" uncomment these 4 lines
# tf.app.flags.DEFINE_string('checkpoint_dir', './',
#                            """Directory where to read model checkpoints.""")
# FLAGS = tf.app.flags.FLAGS
# ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
# saver.restore(sess, ckpt.model_checkpoint_path)
saver.restore(sess, './skt.ckpt')
eval_loss, output, X, Y = get_eval_batch_data(test_iter)

for index in random.sample(range(len(output)), 10):
    inp = [index_to_letter[l] for l in X.T[index]]
    real = [index_to_letter[l] for l in Y.T[index]]
    predict = [index_to_letter[l] for l in np.argmax(output, axis=2)[index]]

    print "input :        " + "".join(inp).split("_")[0]
    print "real output :  " + "".join(real).split("_")[0]
    print "model output : " + "".join(predict).split("_")[0]
    print "is correct :   " + str(real == predict)
    print

correct = 0
for index in range(len(output)):
    inp = [index_to_letter[l] for l in X.T[index]]
    real = [index_to_letter[l] for l in Y.T[index]]
    predict = [index_to_letter[l] for l in np.argmax(output, axis=2)[index]]

    if (real == predict):
        correct += 1
        print "input :        " + "".join(inp).split("_")[0]
        print "real output :  " + "".join(real).split("_")[0]
        print "model output : " + "".join(predict).split("_")[0]
        print "is correct :   " + str(real == predict)
        print


print "Accuracy: ", (float(correct) / len(output))
