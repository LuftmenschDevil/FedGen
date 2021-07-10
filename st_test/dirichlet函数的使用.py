import numpy as np
'''
    一个骰子，每个面丢1000次，求丢到每个面的概率；这个实验重复30次
'''
dirichlet = np.random.dirichlet((1000,1000,1000,1000,1000,1000),30)
print(dirichlet)

a = np.repeat(0.01, 10)
print(a)
proportions=np.random.dirichlet(a)
print(proportions)
proportions=proportions / proportions.sum() # 归一化
proportions=(np.cumsum(proportions) * len(idx_l)).astype(int)[:-1]