import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

vocab_size = 100000
size = 256 # 64, 128, 256
learning_rate = 1e-4 # 1e-3, 5e-4, 1e-4, 5e-5
batch_size = 32 # 16, 32
dropout = 0.1 # 0.0, 0.1, 0.2
max_time_unit = 100

inputs = []
for idx in xrange(max_time_unit):
  inputs.append(tf.placeholder(tf.float32, [batch_size, vocab_size]))

cell_fw = rnn_cell.BasicLSTMCell(size)
cell_bw = rnn_cell.BasicLSTMCell(size)

outputs = rnn.bidirectional_rnn(cell_fw, cell_bw, inputs,
                                dtype=tf.float32,
                                sequence_length=[5]*batch_size)
