import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pylab import mpl

# 指定默认字体
mpl.rcParams['font.sans-serif'] = ['FangSong']
# 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False
#读取数据
dataset = []
f = open("G:\zuoye1\wine.txt")
a = f.readlines()
for line in a[1:]:
    Line = line.strip().split('\t')
    flLine = []
    for i in Line:
        b = float(i)
        flLine.append(b)
    dataset.append(flLine)
df_org= pd.DataFrame(dataset)
df_wine = df_org.iloc[:,1:]
df_std = (df_wine- df_wine.mean()) / df_wine.std()
df_corr = df_std.corr()
eig_value, eig_vector = np.linalg.eig(df_corr)
# 特征值排序
eig = pd.DataFrame({"eig_value": eig_value})
eig = eig.sort_values(by=["eig_value"], ascending=False)
# 获取累积贡献度
eig["eig_cum"] = (eig["eig_value"] / eig["eig_value"].sum()).cumsum()
# 合并入特征向量
eig = eig.merge(pd.DataFrame(eig_vector).T, left_index=True, right_index=True)

#划分数据集，并进行标准化处理
x, y = df_org.iloc[:, 1:].values, df_org.iloc[:, 0].values
x_trian, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
sc = StandardScaler()
x_train_std = sc.fit_transform(x_trian)
x_test_std = sc.fit_transform(x_test)
cov_matrix = np.cov(x_train_std.T)
eigen_val, eigen_vec = np.linalg.eig(cov_matrix)
print("values\n ", eigen_val, "\nvector\n ", eigen_vec)

#主成分方差可视化
tot = sum(eigen_val) # 总特征值和
var_exp = [(i / tot) for i in sorted(eigen_val, reverse=True)] # 计算解释方差比，降序
print(var_exp)
cum_var_exp = np.cumsum(var_exp) # 累加方差比率
plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='独立解释方差') # 柱状 Individual_explained_variance
plt.step(range(1, 14), cum_var_exp, where='mid', label='累加解释方差') # Cumulative_explained_variance
plt.ylabel("解释方差率")
plt.xlabel("主成分索引")
plt.legend(loc='right')
plt.show()

# 假设要求累积贡献度要达到60%，则取2个主成分
# 成分得分系数矩阵（因子载荷矩阵法）
loading = eig.iloc[:2, 2:].T
loading["vars"] = df_std.columns
score = pd.DataFrame(np.dot(df_std, loading.iloc[:, 0:2]))
plt.plot(loading[0], loading[1], "o")
xmin, xmax = loading[0].min(), loading[0].max()
ymin, ymax = loading[1].min(), loading[1].max()
dx = (xmax - xmin) * 0.2
dy = (ymax - ymin) * 0.2
plt.xlim(xmin - dx, xmax + dx)
plt.ylim(ymin - dy, ymax + dy)
plt.xlabel('第1主成分')
plt.ylabel('第2主成分')
for x, y, z in zip(loading[0], loading[1], loading["vars"]):
    plt.text(x, y + 0.1, z, ha='center', va='bottom', fontsize=13)
plt.grid(True)
plt.show()
eigen_pairs = [(np.abs(eigen_val[i]), eigen_vec[:, i]) for i in range(len(eigen_val))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print(w)
x_train_pca = x_train_std.dot(w)

#数据分类结果
color = ['black', 'black', 'black']
marker = ['s', 'x', 'o']
for l,c, m in zip(np.unique(y_train),color, marker):
    plt.scatter(x_train_pca[y_train == l, 0], x_train_pca[y_train == l, 1], label=l,color=c,marker=m)
plt.title('结果')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.legend(loc='lower left')
plt.show()
