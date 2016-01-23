import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

from base_model import Model

class DeepLSTM(Model):
  """Deep LSTM model."""
  def __init__(self, vocab_size, size=256,
               learning_rate=1e-4, batch_size=32,
               dropout=0.1, max_time_unit=100,
               checkpoint_dir="checkpoint", forward_only=False):
    """Initialize the parameters for an Deep LSTM model.
    
    Args:
      vocab_size: int, The dimensionality of the input vocab
      size: int, The dimensionality of the inputs into the Deep LSTM cell [32, 64, 256]
      learning_rate: float, [1e-3, 5e-4, 1e-4, 5e-5]
      batch_size: int, The size of a batch [16, 32]
      dropout: unit Tensor or float between 0 and 1 [0.0, 0.1, 0.2]
      max_time_unit: int, The max time unit [100]
    """
    super(DeepLSTM, self).__init__()

    self.vocab_size = int(vocab_size)
    self.size = int(size)
    self.learning_rate = float(learning_rate)
    self.batch_size = int(batch_size)
    self.dropout = float(dropout)
    self.max_time_unit = int(max_time_unit)

    self.inputs = []

    self.emb = tf.Variable(tf.truncated_normal([self.vocab_size, self.size], -0.1, 0.1), name='emb')

    for idx in xrange(max_time_unit):
      #self.inputs.append(tf.placeholder(tf.float32, [self.batch_size, vocab_size]))
      self.inputs.append(tf.nn.embedding_lookup(self.emb,
                                                tf.placeholder(tf.int32, 1)))

    self.cell_fw = rnn_cell.BasicLSTMCell([1, size])
    self.cell_bw = rnn_cell.BasicLSTMCell([1, size])
    self.state_fw = tf.zeros([1, size])
    self.state_bw = tf.zeros([1, size])

  def build_model(self, sequence_length):
    return rnn.bidirectional_rnn(self.cell_fw,
                                 self.cell_bw,
                                 self.inputs,
                                 #dtype=tf.float32,
                                 sequence_length=sequence_length,
                                 initial_state_fw=self.state_fw,
                                 initial_state_bw=self.state_bw)
