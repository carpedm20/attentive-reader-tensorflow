import pprint
import numpy as np

pp = pprint.PrettyPrinter()

def array_pad(array, width, pad=-1, force=False):
  max_length = max(map(len, array))
  if max_length > width and force != True:
    raise Exception(" [!] Max length of array %s is bigger than given %s" % (max_length, width))
  result = np.full([len(array), width], pad, dtype=np.int64)
  for i, row in enumerate(array):
    for j, val in enumerate(row[:width-1]):
      result[i][j] = val
  return result
