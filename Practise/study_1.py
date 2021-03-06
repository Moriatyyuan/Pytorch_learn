
# %%
from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import numpy as np
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()


# %%
x = torch.empty(5, 3)
print(x)
# %%
x = torch.tensor([5.5, 3])
print(x)

# %%
x = x.new_ones(5, 3, dtype=torch.double)
print(x)


# %%
x = torch.randn_like(x, dtype=torch.float)
print(x)

# %%
print(x.size())

# %%
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
# %%
result = torch.empty(5, 3)
torch.add(x, y, out=result)

# %%
y.add_(x)
print(y)

# %%
print(x[:, 1])

# %%
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())


# %%
x = torch.rand(1)
print(x)
print(x.item())

# %%
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# %%
a.add_(1)
print(a)
print(b)

# %%
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# %%


def getNthFib(n):
    if n == 1:
        return 0
    if n == 2:
        return 1
    if n > 2:
        return getNthFib(n - 1) + getNthFib(n - 2)


print(getNthFib(6))

# %%
