import torch
import numpy as np

np_a = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
np_b = np.array([[6,7,8,9]])
a = torch.from_numpy(np_a)
b = torch.from_numpy(np_b)
print(a)
print(b)
c = torch.cat([a,b],dim=1)
print(c)