import random
import numpy as np
import data_generator as dg
import draw
import lingam
from sklearn.linear_model import LinearRegression
import utils
from statsmodels.sandbox.regression.gmm import IV2SLS
import statsmodels.api as sm

from causallearn.utils.KCI.KCI import KCI_CInd, KCI_UInd

import causaldag


DAG_1 = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        
    ]
)

draw.draw_directed_graph(DAG_1)

noise = dg.get_noisy(1000, DAG_1)
B_0 = dg.get_B_0(DAG_1)
X_1 = dg.get_x(DAG_1, noise, B_0)

print(X_1.shape)

X_observed_1 = X_1.T[:, 1:]
print(X_observed_1.shape)

RCD_model = lingam.RCD(max_explanatory_num=2)
RCD_model.fit(X_observed_1)
res = RCD_model.adjacency_matrix_
print(res)

draw.draw_directed_graph(res)

#####　try
# reg = LinearRegression(fit_intercept=False)
# reg.fit(X_observed_1[:, 0].reshape(-1, 1), X_observed_1[:, 1].reshape(-1, 1))
# x_2_hat = reg.predict(X_observed_1[:, 0].reshape(-1, 1))

# reg.fit(x_2_hat.reshape(-1, 1), X_observed_1[:, 2].reshape(-1, 1))
# residual_3 = X_observed_1[:, 2].reshape(-1, 1) - reg.predict(X_observed_1[:, 0].reshape(-1, 1))

# print(utils.is_independent(residual_3, X_observed_1[:, 1], 0.01))

# reg.fit(X_observed_1[:, 0].reshape(-1, 1), X_observed_1[:, 2].reshape(-1, 1))
# x_3_hat = reg.predict(X_observed_1[:, 0].reshape(-1, 1))

# reg.fit(x_3_hat.reshape(-1, 1), X_observed_1[:, 1].reshape(-1, 1))
# residual_2 = X_observed_1[:, 1].reshape(-1, 1) - reg.predict(X_observed_1[:, 0].reshape(-1, 1))

# print(utils.is_independent(residual_2, X_observed_1[:, 2], 0.01))

# 生成工具变量
Z = X_observed_1[:, 0]

# 生成内生变量（受工具变量影响）
X = X_observed_1[:, 1]

# 生成因变量
y = X_observed_1[:, 2]

# 准备数据矩阵（添加常数项）
# X_matrix = np.column_stack((np.ones(1000), X))  # 包含内生变量的解释变量
# Z_matrix = np.column_stack((np.ones(1000), Z))  # 工具变量

X_matrix = X  # 包含内生变量的解释变量
Z_matrix = Z  # 工具变量
# print(exog.shape)

# 使用IV2SLS进行估计
iv_model = IV2SLS(y, X_matrix, Z_matrix)
iv_results = iv_results = iv_model.fit()

# 计算残差
iv_predicted = iv_results.predict()
iv_residuals = y - iv_predicted

# print(utils.is_independent(iv_residuals, X, 0.01))

X_model = sm.OLS(X, Z_matrix)
X_results = X_model.fit()
X_hat = X_results.predict()

first_stage_residuals = X - X_hat


kci_object = KCI_CInd()
# p, _ = kci_object.compute_pvalue(data_x=Z.reshape(-1, 1), data_y=y.reshape(-1, 1), data_z=X_hat.reshape(-1, 1))
# print(p)
# print("KCI: ", p>0.01)

# p, _ = kci_object.compute_pvalue(data_x=Z.reshape(-1, 1), data_y=y.reshape(-1, 1), data_z=X.reshape(-1, 1))
# print(p)
# print("KCI: ", p>0.01)

print("1: ", utils.is_independent(iv_residuals, Z, 0.05))

print("2: ", utils.is_independent(first_stage_residuals, y, 0.05))

print("3: ", utils.is_independent(first_stage_residuals, iv_residuals, 0.05))


# 生成工具变量
Z = X_observed_1[:, 0]

# 生成内生变量（受工具变量影响）
X = X_observed_1[:, 2]

# 生成因变量
y = X_observed_1[:, 1]

# 准备数据矩阵（添加常数项）
# X_matrix = np.column_stack((np.ones(1000), X))  # 包含内生变量的解释变量
# Z_matrix = np.column_stack((np.ones(1000), Z))  # 工具变量

X_matrix = X  # 包含内生变量的解释变量
Z_matrix = Z  # 工具变量
# print(exog.shape)

# 使用IV2SLS进行估计
iv_model = IV2SLS(y, X_matrix, Z_matrix)
iv_results = iv_results = iv_model.fit()

# 计算残差
iv_predicted = iv_results.predict()
iv_residuals = y - iv_predicted

# print(utils.is_independent(iv_residuals, X, 0.01))

X_model = sm.OLS(X, Z_matrix)
X_results = X_model.fit()
X_hat = X_results.predict()

first_stage_residuals = X - X_hat


# kci_object = KCI_CInd()
# p, _ = kci_object.compute_pvalue(data_x=Z.reshape(-1, 1), data_y=y.reshape(-1, 1), data_z=X_hat.reshape(-1, 1))
# print(p)
# print("KCI: ", p>0.01)

# p, _ = kci_object.compute_pvalue(data_x=Z.reshape(-1, 1), data_y=y.reshape(-1, 1), data_z=X.reshape(-1, 1))
# print(p)
# print("KCI: ", p>0.01)

print("1: ", utils.is_independent(iv_residuals, Z, 0.05))

print("2: ", utils.is_independent(first_stage_residuals, y, 0.05))

print("3: ", utils.is_independent(first_stage_residuals, iv_residuals, 0.05))
#####
DAG_2 = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        
    ]
)

draw.draw_directed_graph(DAG_2)

noise = dg.get_noisy(1000, DAG_2)
B_0 = dg.get_B_0(DAG_2)
X_2 = dg.get_x(DAG_2, noise, B_0)

print(X_2.shape)

X_observed_2 = X_2.T[:, 1:]
print(X_observed_2.shape)

RCD_model = lingam.RCD(max_explanatory_num=2)
RCD_model.fit(X_observed_2)
res_2 = RCD_model.adjacency_matrix_
print(res_2)

draw.draw_directed_graph(res_2)

#####　try
# reg = LinearRegression(fit_intercept=False)
# reg.fit(X_observed_2[:, 0].reshape(-1, 1), X_observed_2[:, 1].reshape(-1, 1))
# x_2_hat = reg.predict(X_observed_2[:, 0].reshape(-1, 1))

# reg.fit(x_2_hat.reshape(-1, 1), X_observed_2[:, 2].reshape(-1, 1))
# residual_3 = X_observed_2[:, 2].reshape(-1, 1) - reg.predict(X_observed_2[:, 0].reshape(-1, 1))

# print(utils.is_independent(residual_3, X_observed_2[:, 1], 0.01))

# reg.fit(X_observed_2[:, 0].reshape(-1, 1), X_observed_2[:, 2].reshape(-1, 1))
# x_3_hat = reg.predict(X_observed_2[:, 0].reshape(-1, 1))

# reg.fit(x_3_hat.reshape(-1, 1), X_observed_2[:, 1].reshape(-1, 1))
# residual_2 = X_observed_2[:, 1].reshape(-1, 1) - reg.predict(X_observed_2[:, 0].reshape(-1, 1))

# print(utils.is_independent(residual_2, X_observed_2[:, 2], 0.01))

# 生成工具变量
Z = X_observed_2[:, 0]

# 生成内生变量（受工具变量影响）
X = X_observed_2[:, 1]

# 生成因变量
y = X_observed_2[:, 2]

# 准备数据矩阵（添加常数项）
# X_matrix = np.column_stack((np.ones(1000), X))  # 包含内生变量的解释变量
# Z_matrix = np.column_stack((np.ones(1000), Z))  # 工具变量

X_matrix = X  # 包含内生变量的解释变量
Z_matrix = Z  # 工具变量
# print(exog.shape)

# 使用IV2SLS进行估计
iv_model = IV2SLS(y, X_matrix, Z_matrix)
iv_results = iv_results = iv_model.fit()

# 计算残差
iv_predicted = iv_results.predict()
iv_residuals = y - iv_predicted

# print(utils.is_independent(iv_residuals, X, 0.01))

X_model = sm.OLS(X, Z_matrix)
X_results = X_model.fit()
X_hat = X_results.predict()

first_stage_residuals = X - X_hat


# kci_object = KCI_CInd()
# p, _ = kci_object.compute_pvalue(data_x=Z.reshape(-1, 1), data_y=y.reshape(-1, 1), data_z=X_hat.reshape(-1, 1))
# print(p)
# print("KCI: ", p>0.01)
print("1: ", utils.is_independent(iv_residuals, Z, 0.05))

print("2: ", utils.is_independent(first_stage_residuals, y, 0.05))

print("3: ", utils.is_independent(first_stage_residuals, iv_residuals, 0.05))

# p, _ = kci_object.compute_pvalue(data_x=Z.reshape(-1, 1), data_y=y.reshape(-1, 1), data_z=X.reshape(-1, 1))
# print(p)
# print("KCI: ", p>0.01)

# 生成工具变量
Z = X_observed_2[:, 0]

# 生成内生变量（受工具变量影响）
X = X_observed_2[:, 2]

# 生成因变量
y = X_observed_2[:, 1]

# 准备数据矩阵（添加常数项）
# X_matrix = np.column_stack((np.ones(1000), X))  # 包含内生变量的解释变量
# Z_matrix = np.column_stack((np.ones(1000), Z))  # 工具变量

X_matrix = X  # 包含内生变量的解释变量
Z_matrix = Z  # 工具变量
# print(exog.shape)

# 使用IV2SLS进行估计
iv_model = IV2SLS(y, X_matrix, Z_matrix)
iv_results = iv_results = iv_model.fit()

# 计算残差
iv_predicted = iv_results.predict()
iv_residuals = y - iv_predicted

# print(utils.is_independent(iv_residuals, X, 0.01))

X_model = sm.OLS(X, Z_matrix)
X_results = X_model.fit()
X_hat = X_results.predict()

first_stage_residuals = X - X_hat


# kci_object = KCI_CInd()
# p, _ = kci_object.compute_pvalue(data_x=Z.reshape(-1, 1), data_y=y.reshape(-1, 1), data_z=X_hat.reshape(-1, 1))
# print(p)
# print("KCI: ", p>0.01)
print("1: ", utils.is_independent(iv_residuals, Z, 0.05))

print("2: ", utils.is_independent(first_stage_residuals, y, 0.05))

print("3: ", utils.is_independent(first_stage_residuals, iv_residuals, 0.05))

# p, _ = kci_object.compute_pvalue(data_x=Z.reshape(-1, 1), data_y=y.reshape(-1, 1), data_z=X.reshape(-1, 1))
# print(p)
# print("KCI: ", p>0.01)





