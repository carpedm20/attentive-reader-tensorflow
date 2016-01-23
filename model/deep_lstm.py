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
    super(DeepLSTM, self).__init__(FLAGS)

    self.vocab_size = vocab_size
    self.size = size
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.dropout = dropout
    self.max_time_unit = max_time_unit

    self.inputs = []
    for idx in xrange(max_time_unit):
      self.inputs.append(tf.placeholder(tf.float32, [batch_size, vocab_size]))

    self.cell_fw = rnn_cell.BasicLSTMCell(size)
    self.cell_bw = rnn_cell.BasicLSTMCell(size)

  def build_model(self, sequence_length):
    return rnn.bidirectional_rnn(self.cell_fw,
                                 self.cell_bw,
                                 self.inputs,
                                 dtype=tf.float32,
                                 sequence_length=sequence_length)
