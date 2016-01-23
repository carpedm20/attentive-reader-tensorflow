import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

from base_model import Model
from data_utils import load_vocab, load_dataset

class DeepLSTM(Model):
  """Deep LSTM model."""
  def __init__(self, vocab_size, size=256,
               learning_rate=1e-4, batch_size=32,
               dropout=0.1, seq_length=100,
               checkpoint_dir="checkpoint", forward_only=False):
    """Initialize the parameters for an Deep LSTM model.
    
    Args:
      vocab_size: int, The dimensionality of the input vocab
      size: int, The dimensionality of the inputs into the Deep LSTM cell [32, 64, 256]
      learning_rate: float, [1e-3, 5e-4, 1e-4, 5e-5]
      batch_size: int, The size of a batch [16, 32]
      dropout: unit Tensor or float between 0 and 1 [0.0, 0.1, 0.2]
      seq_length: int, The max time unit [100]
    """
    super(DeepLSTM, self).__init__()

    self.vocab_size = int(vocab_size)
    self.size = int(size)
    self.learning_rate = float(learning_rate)
    self.batch_size = int(batch_size)
    self.dropout = float(dropout)
    self.seq_length = int(seq_length)

    self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
    self.input_lengths = tf.placeholder(tf.int64, [self.batch_size])

    self.emb = tf.Variable(tf.truncated_normal([self.vocab_size, self.size], -0.1, 0.1), name='emb')
    self.embed_inputs = tf.nn.embedding_lookup(self.emb, tf.transpose(self.inputs))

    self.cell_fw = rnn_cell.BasicLSTMCell(size)
    self.cell_bw = rnn_cell.BasicLSTMCell(size)

    self.output = rnn.bidirectional_rnn(self.cell_fw,
                                        self.cell_bw,
                                        tf.unpack(self.embed_inputs),
                                        dtype=tf.float32,
                                        sequence_length=self.input_lengths)

    output = tf.reduce_sum(tf.pack(self.output), 0)

  def train(self, epoch=25, batch_size=1,
            learning_rate=0.0002, momentum=0.9, decay=0.95,
            data_dir="data", dataset_name="cnn", vocab_size=1000000):
    if not self.vocab:
      self.vocab, self.rev_vocab = load_vocab(data_dir, dataset_name, vocab_size)

    self.opt = tf.train.RMSPropOptimizer(learning_rate,
                                         decay=decay,
                                         momentum=momentum)

    for epoch_idx in xrange(epoch):
      data_loader = load_dataset(data_dir, dataset_name, vocab_size)

      contexts, questions, answers = [], [], []
      for batch_idx in xrange(batch_size):
        _, context, question, answer, _ = data_loader.next()
        contexts.append(context)
        questions.append(question)
        answers.append(answers)

      self.model.
