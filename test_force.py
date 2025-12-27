from ase.io import read
import numpy as np


print(read("test.xyz").get_forces())
