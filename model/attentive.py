import tensorflow as tf
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

from base_model import Model

class AttentiveReader(Model):
  """Attentive Reader."""
  def __init__(self, vocab_size, size=256,
               learning_rate=1e-4, batch_size=32,
               dropout=0.1, max_time_unit=100):
    """Initialize the parameters for an  Attentive Reader model.

    Args:
      vocab_size: int, The dimensionality of the input vocab
      size: int, The dimensionality of the inputs into the Deep LSTM cell [32, 64, 256]
      learning_rate: float, [1e-3, 5e-4, 1e-4, 5e-5]
      batch_size: int, The size of a batch [16, 32]
      dropout: unit Tensor or float between 0 and 1 [0.0, 0.1, 0.2]
      max_time_unit: int, The max time unit [100]
    """
    super(DeepLSTM, self).__init__()

    self.vocab_size = vocab_size
    self.size = size
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.dropout = dropout
    self.max_time_unit = max_time_unit

    self.inputs = []
    for idx in xrange(max_time_unit):
      self.inputs.append(tf.placeholder(tf.float32, [batch_size, vocab_size]))

