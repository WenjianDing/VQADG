import numpy as np
import torch
random_indices = np.random.permutation(8)
data = torch.tensor([[1,1,1],
                    [2,2,2],
                    [3,3,3],
                    [4,4,4]])
data2 = torch.tensor([[1,1,2],
                    [2,2,3],
                    [3,3,4],
                    [4,4,5]])
data_total = torch.cat((data, data2), dim=0)
new = data_total[random_indices, :]
# print(data_total)
# print(data_total.shape)
# print(new)
data22 = torch.tensor([[1],[1],[1],[1]])
print(data22.shape)
B=4
target = data22.new_zeros((B,1))
print(target)
print(target.shape)