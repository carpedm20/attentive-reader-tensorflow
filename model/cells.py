import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import RNNCell, linear

class LSTMCell(RNNCell):
  """Almost same with tf.models.rnn.rnn_cell.BasicLSTMCell
  except adding c to inputs and h to calculating gates,
  adding a skip connection from the input of current time t,
  and returning only h not concat of c and h.
  """

  def __init__(self, num_units, forget_bias=1.0):
    self._num_units = num_units
    self._forget_bias = forget_bias
    self.c = None

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope("BasicLSTMCell"):
      h = state
      if self.c == None: self.c = tf.reshape(tf.zeros_like(h), [-1, self._num_units])
      concat = linear([inputs, h, self.c], 4 * self._num_units, True)

      i, j, f, o = tf.split(1, 4, concat)

      self.c = self.c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
      new_h = tf.tanh(self.c) * tf.sigmoid(o)

      softmax_w = tf.get_variable("softmax_w", [self._num_units, self._num_units])
      softmax_b = tf.get_variable("softmax_b", [self._num_units])

      new_y = tf.nn.xw_plus_b(new_h, softmax_w, softmax_b)

    return new_y, new_y


class MultiRNNCellWithSkipConn(RNNCell):
  """Almost same with tf.models.rnn.rnn_cell.MultiRnnCell adding
  a skip connection from the input of current time t and using 
  _num_units not state size because LSTMCell returns only [h] not [c, h].
  """

  def __init__(self, cells):
    """Create a RNN cell composed sequentially of a number of RNNCells.
    Args:
      cells: list of RNNCells that will be composed in this order.
    Raises:
      ValueError: if cells is empty (not allowed) or if their sizes don't match.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    for i in xrange(len(cells) - 1):
      if cells[i + 1].input_size != cells[i].output_size:
        raise ValueError("In MultiRNNCell, the input size of each next"
                         " cell must match the output size of the previous one."
                         " Mismatched output size in cell %d." % i)
    self._cells = cells

  @property
  def input_size(self):
    return self._cells[0].input_size

  @property
  def output_size(self):
    return self._cells[-1].output_size

  @property
  def state_size(self):
    return sum([cell.state_size for cell in self._cells])

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with tf.variable_scope("MultiRNNCellWithConn"):
      cur_state_pos = 0
      first_layer_input = cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with tf.variable_scope("Cell%d" % i):
          cur_state = tf.slice(
              state, [0, cur_state_pos], [-1, cell.state_size])
          cur_state_pos += cell.state_size
          # Add skip connection from the input of current time t.
          if i != 0:
            first_layer_input = first_layer_input
          else:
            first_layer_input = tf.zeros_like(first_layer_input)
          cur_inp, new_state = cell(tf.concat(1, [inputs, first_layer_input]), cur_state)
          new_states.append(new_state)
    return cur_inp, tf.concat(1, new_states)
