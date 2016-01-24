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

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return 2 * self._num_units

  def __call__(self, inputs, state, first_layer_input, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope("BasicLSTMCell"):
      inputs = tf.concat(1, [inputs, first_layer_input])
      c, h = tf.split(1, 2, state)
      concat = linear([inputs, h, c], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(1, 4, concat)

      new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)

      softmax_w = tf.get_variable("softmax_w", [self._num_units, self._num_units])
      softmax_b = tf.get_variable("softmax_b", [self._num_units])

      new_y = tf.nn.xw_plus_b(new_h, softmax_w, softmax_b)

    return new_y, new_y

class MultiRNNCellWithSkipConn(RNNCell):
  """Almost same with tf.models.rnn.rnn_cell.MultiRnnCell
  adding a skip connection from the input of current time t"""

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

          cur_inp, new_state = cell(cur_inp, cur_state, first_layer_input)
          new_states.append(new_state)
    return cur_inp, tf.concat(1, new_states)

class DropoutWrapper(RNNCell):
  """Almost same with tf.models.rnn.rnn_cell.DropoutWrapper
  except adding a skip connection from the input of current time t.
  """

  def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
               seed=None):
    """Create a cell with added input and/or output dropout.
    Dropout is never used on the state.
    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      input_keep_prob: unit Tensor or float between 0 and 1, input keep
        probability; if it is float and 1, no input dropout will be added.
      output_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is float and 1, no output dropout will be added.
      seed: (optional) integer, the randomness seed.
    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if keep_prob is not between 0 and 1.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")
    if (isinstance(input_keep_prob, float) and
        not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
      raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                       % input_keep_prob)
    if (isinstance(output_keep_prob, float) and
        not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
      raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                       % output_keep_prob)
    self._cell = cell
    self._input_keep_prob = input_keep_prob
    self._output_keep_prob = output_keep_prob
    self._seed = seed

  @property
  def input_size(self):
    return self._cell.input_size

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, inputs, state, first_layer_input, scope=None):
    """Run the cell with the declared dropouts."""
    if (not isinstance(self._input_keep_prob, float) or
        self._input_keep_prob < 1):
      inputs = tf.nn.dropout(inputs, self._input_keep_prob, seed=self._seed)
    # Execute __call__ with first layer input
    output, new_state = self._cell(inputs, state, first_layer_input)
    if (not isinstance(self._output_keep_prob, float) or
        self._output_keep_prob < 1):
      output = tf.nn.dropout(output, self._output_keep_prob, seed=self._seed)
    return output, new_state

def rnn(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
  """Creates a recurrent neural network specified by RNNCell "cell".
  """

  if not isinstance(cell, rnn_cell.RNNCell):
    raise TypeError("cell must be an instance of RNNCell")
  if not isinstance(inputs, list):
    raise TypeError("inputs must be a list")
  if not inputs:
    raise ValueError("inputs must not be empty")

  outputs = []
  states = []
  with vs.variable_scope(scope or "RNN"):
    batch_size = array_ops.shape(inputs[0])[0]
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, dtype must be.")
      state = cell.zero_state(batch_size, dtype)

    if sequence_length:  # Prepare variables
      max_sequence_length = math_ops.reduce_max(sequence_length)

    for time, input_ in enumerate(inputs):
      if time > 0: vs.get_variable_scope().reuse_variables()
      # pylint: disable=cell-var-from-loop
      def output_state():
        return cell(input_, state)
      # pylint: enable=cell-var-from-loop
      if sequence_length:
        (output, state) = control_flow_ops.cond(
            time >= max_sequence_length,
            lambda: input_, output_state)
      else:
        (output, state) = output_state()

      outputs.append(output)
      states.append(state)

    return (outputs, states)
