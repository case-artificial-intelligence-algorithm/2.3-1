#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 局部异常因子算法（Local Outlier Factor, LOF）是基于密度的异常检测方法中最具有代表性的一类，
# 它通过计算给定样本相对于其邻域的局部密度偏差来实现异常检测。

import numpy as np
from scipy.spatial.distance import cdist

class LOF:
    # 
    def __init__(self, data, k, epsilon=1.0):
        self.data = data
        self.k = k
        self.epsilon = epsilon
        self.N = self.data.shape[0]

    # 计算数据集中样本间相互距离
    def get_dist(self):
        return cdist(self.data, self.data)

    # 计算k距离
    def _kdist(self, arr):
        inds_sort = np.argsort(arr)
        neighbor_ind = inds_sort[1:self.k + 1]
        return neighbor_ind, arr[neighbor_ind[-1]]  # 返回最近的k领域和k距离

    # 计算可达距离
    def get_rdist(self):
        dist = self.get_dist() 
        nei_inds, kdist = [], []    # 记录k领域和k距离
        for i in range(self.N):
            neighbor_ind, k = self._kdist(dist[i])  # 计算所有点的k领域和k距离
            nei_inds.append(neighbor_ind)
            kdist.append(k)
        for i, k in enumerate(kdist):
            ind = np.where(dist[i] < k)
            dist[i][ind] = k    # 记录k领域内的点的可达距离
        return nei_inds, dist   # 返回k领域和可达距离

    # 计算局部可达密度
    def get_lrd(self, nei_inds, rdist):
        lrd = np.zeros(self.N)
        for i, inds in enumerate(nei_inds):
            s = 0
            for j in inds:
                s += rdist[j, i]
            lrd[i] = self.k / s
        return lrd

    # 计算局部离群因子
    def run(self):
        nei_inds, rdist = self.get_rdist()  # 记录k领域和可达距离
        lrd = self.get_lrd(nei_inds, rdist) # 计算局部可达密度
        score = np.zeros(self.N)
        for i, inds in enumerate(nei_inds):

            raise NotImplementedError('补充代码，计算k领域内的局部可达密度之和以及局部离群因子')

        return score, np.where(score > self.epsilon)[0]

# 待测试程序
def solution():
    np.random.seed(42)
    X_inliers = np.random.randn(100, 2) # 随机产生正常样本
    X_outliers = np.random.uniform(low=-4, high=4, size=(30, 2)) # 随机产生异常样本
    data = np.r_[X_inliers, X_outliers]
    k, epsilon = 15, 1.5    # 定义领域大小和异常点阈值

    lof = LOF(data, k, epsilon)
    score, out_ind = lof.run()
    
    outliers, out_score = data[out_ind], score[out_ind] # 获取异常点和异常点的局部离群因子

    for a, b in zip(outliers[:5], out_score[:5]):
        print("异常点坐标:(%.2f，%.2f) " % (a[0], a[1] + 0.001), \
            "局部异常因子:%.4f" % b)

    return out_ind[:5] # 返回前5个异常点的索引


if __name__ == '__main__':
    pass
