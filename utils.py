import pprint
import numpy as np

pp = pprint.PrettyPrinter()

def zero_pad(array, width, force=False):
  max_length = max(map(len, array))
  if max_length > width and force != True:
    raise Exception(" [!] Max length of array %s is bigger than given %s" % (max_length, width))
  result = np.zeros([len(array), width], dtype=np.int64)
  for i, row in enumerate(array):
    for j, val in enumerate(row[:width]):
      result[i][j] = val
  return result
