import numpy as np
from sklearn.linear_model import LinearRegression
from hyppo.independence import Hsic
import matplotlib.pyplot as plt
from hyppo.independence import Dcorr


# mu1, sigma1 = 2, 1     # 第一个高斯分布的均值和标准差
# mu2, sigma2 = -2, 1   # 第二个高斯分布的均值和标准差
# pi1, pi2 = 0.5, 0.5    # 混合权重
# # 生成数据
# data1 = np.random.normal(mu1, sigma1, int(1000*pi1))  # 生成第一个分布的数据
# data2 = np.random.normal(mu2, sigma2, int(1000*pi2))  # 生成第二个分布的数据
# mixed_data = np.concatenate([data1, data2])

# x_1 = mixed_data

# x_1 = np.random.uniform(0, 1, size=1000)

x_1 = np.random.lognormal(mean=0, sigma=1, size=1000)
# x_1 = np.random.exponential(scale=1/10, size=1500)
# x_1 = np.random.random(size = 1000)*np.sqrt(3)*2-np.sqrt(3)

x_1_test = (x_1 - np.mean(x_1)) / np.std(x_1)

b = np.random.random()*0.5+0.5
print(b)
# data1 = np.random.normal(mu1, sigma1, int(1000*pi1))  # 生成第一个分布的数据
# data2 = np.random.normal(mu2, sigma2, int(1000*pi2))  # 生成第二个分布的数据
# mixed_data = np.concatenate([data1, data2])

# x_2 = b*x_1 + mixed_data

# x_2 = b*x_1 + (np.random.random(size = 1000)*np.sqrt(3)*2-np.sqrt(3))
# x_2 = b*x_1 + np.random.exponential(scale=1/10, size=1500)
x_2 = b*x_1 + np.random.lognormal(mean=0, sigma=1, size=1000)
# x_2 = b*x_1 + np.random.uniform(0, 1, size=1000)
x_2_test = (x_2 - np.mean(x_2)) / np.std(x_2)

plt.hist(x_1_test, bins=50)
# plt.title(str(i) + " distribution")
plt.show()
plt.hist(x_2_test, bins=50)
# plt.title(str(i) + " distribution")
plt.show()


_, p = Hsic().test(x_1_test, x_2_test)
# _, p = Dcorr().test(x_1, x_2)

print(p)

reg = LinearRegression(fit_intercept=False)
res = reg.fit(x_1_test.reshape(-1, 1), x_2_test.reshape(-1, 1)) # x, y

r1_2 = x_2_test.reshape(-1, 1) - reg.predict(x_1_test.reshape(-1, 1))
print(x_1.shape)
print(r1_2.shape)

_, p1_2 = Hsic().test(x_1_test, r1_2)
# _, p1_2 = Dcorr().test(x_1, r1_2)

print(p1_2)


res = reg.fit(x_2_test.reshape(-1, 1), x_1_test.reshape(-1, 1)) # x, y

r2_1 = x_1_test.reshape(-1, 1) - reg.predict(x_2_test.reshape(-1, 1))
print(x_2.shape)
print(r2_1.shape)


_, p2_1 = Hsic().test(x_2_test, r2_1)
# _, p2_1 = Dcorr().test(x_2, r2_1)

print(p2_1)