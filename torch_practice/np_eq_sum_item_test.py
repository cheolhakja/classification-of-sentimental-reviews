import numpy as np

a1 = np.array([1,2,3,4,5])
a2 = np.array([10,2,3,4,5])

print(a1.__eq__(a2) )
print(a1.__eq__(a2).sum() )
print(a1.__eq__(a2).sum().item() )
