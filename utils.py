import pprint
import numpy as np

pp = pprint.PrettyPrinter()

def pad_array(array, width):
  map(lambda x: x.extend([0]*(width-len(x))), array)
  return np.array(array)
