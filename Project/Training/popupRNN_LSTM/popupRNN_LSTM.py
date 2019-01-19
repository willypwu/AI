#import preDataOp
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import rnn

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
max_document_length = 30


def loadGloVe():
    vocab = dict()
    embd = []
    # vocab[UNK_SYMBOL] = 0
    # vocab.append('unk') #装载不认识的词
    # embd.append([0]*100) #这个emb_size可能需要指定
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as file:
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab[row[0]] = len(vocab)
            # vocab.append(row[0])
            embd.append([float(_) for _ in row[1:]])
    print('Loaded GloVe!')
    return vocab, embd


# 获取数据
def extract(path):
    with open(path, 'r') as file:
        output_x = []
        output_y = []
        output_lx = []

        for count, line in enumerate(file.readlines()):  # yield/generator
            y = line.split('-')[0]
            s = line.split('-')[1].strip('\n')
            x = [vocab.get(word, UNK_ID) for word in s.split(' ')]
            output_x.append(x)  # [[1, 6, 3], [5, 9]]
            output_lx.append(len(x))
            output_y.append(int(y))

        max_lx = max([len(x) for x in output_x])
        output_x = [x + [UNK_ID] * (max_lx - len(x)) for x in output_x]

    return output_x, output_lx, output_y


def ex(mini_batch, path):

    def extract_generator(path):
        with open(path, 'r') as file:
            for count, line in enumerate(file.readlines()):  # yield/generator
                y = line.split('-')[0]
                s = line.split('-')[1].strip('\n')
                x = [vocab.get(word, UNK_ID) for word in s.split(' ')]
                yield x, len(x), int(y)

    g = extract_generator(path)
    train_x = []
    train_y = []
    train_lx = []
    count = 0
    for x, lx, y in g:
        train_x.append(x)  # [[1, 6, 3], [5, 9]]
        train_y.append(y)
        train_lx.append(lx)
        count = count+1
        if(count >= mini_batch):
            max_lx = max([len(x) for x in train_x])
            train_x = [x + [UNK_ID] * (max_lx - len(x)) for x in train_x]
            yield train_x, train_lx, train_y
            train_x = []
            train_y = []
            train_lx = []
            count = 0


vocab, embd = loadGloVe()  # vocab是词表，embd是词向量
vocab_size = len(vocab)
embedding_dim = len(embd[0])
embedding = np.asarray(embd)


# Training Parameters
learning_rate = 0.001
training_steps = 10000
display_step = 200
mini_batch = 50

# Network Parameters
num_input = 100
num_hidden = 128  # hidden layer num of features
num_classes = 2  # MNIST total classes (0-9 digits)

# tf Graph input
word_ids = tf.placeholder(name="x", shape=[None, None], dtype=tf.int32)  # [B, T]
word_length = tf.placeholder(name="len_x", shape=[None], dtype=tf.int32)  # [B]
Y = tf.placeholder(name="y", shape=[None], dtype=tf.int32)  # [B]

# Define weights
weights = tf.get_variable(shape=[num_hidden, num_classes], name="weight", dtype=tf.float32
                          , initializer=tf.random_normal_initializer(mean=0, stddev=1))
biases = tf.get_variable(shape=[num_classes], name="bias", dtype=tf.float32
                         , initializer=tf.random_normal_initializer(mean=0, stddev=1))
rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden)


UNK_SYMBOL = 'unk'
UNK_ID = vocab[UNK_SYMBOL]

word_embedding_matrix = tf.constant(embedding, name="em_variable", dtype=tf.float32)
# em_variable = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
#                           trainable=False, name="em_variable")
# embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
# embedding_init = em_variable.assign(embedding_placeholder)

input_xs = tf.nn.embedding_lookup(word_embedding_matrix, word_ids)  # [B, T, D]

# 创建RNN网络
# outputs: [B, T, H], state: [B, H]
outputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs=input_xs,
                                   sequence_length=word_length,
                                   dtype=tf.float32)
logits = tf.matmul(state[-1], weights) + biases
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.cast(Y, tf.int64))
# correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

test_x, test_lx, test_y = extract('test.txt')

with tf.Session() as sess:
    sess.run(init)
    # sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
    for step in range(1, training_steps + 1):
        g = ex(mini_batch, 'result.txt')
        for train_x, train_lx, train_y in g:
            sess.run(train_op, feed_dict={word_ids: train_x, Y: train_y, word_length: train_lx})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={word_ids: test_x, Y: test_y, word_length: test_lx})
            print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

    print("Optimization Finished!")



