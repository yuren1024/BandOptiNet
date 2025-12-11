'''
Author: Yuren
Date: 2024-07-25 16:37:24
LastEditors: Yuren
LastEditTime: 2024-07-26 15:03:28
FilePath: /Wavenumber_select/DQN_unmix/RL_utils/base_utils.py
Description: 

Copyright (c) 2024 by Yuren, All Rights Reserved. 
'''
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from joblib import Parallel, delayed
import random
import torch
from pymcr.mcr import McrAR
from pymcr.regressors import OLS, NNLS
from pymcr.constraints import ConstraintNonneg, ConstraintNorm

def set_seed(rand_seed=1023):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

def con_norm(con):
    row_sums = con.sum(axis=1).reshape(-1, 1)
    row_sums[row_sums == 0] = 1
    con = con/row_sums
    return con

def mcr_amend(input,endmembers,iter_nums = 400):
    try:
        model = McrAR(max_iter=iter_nums, st_regr=NNLS(), c_regr=OLS(),
              c_constraints=[ConstraintNonneg(), ConstraintNorm()])
        model.fit(input,ST=endmembers)
        return model.C_opt_

    except:
        model = McrAR(max_iter=iter_nums, st_regr=NNLS(), c_regr=OLS(),
                      c_constraints=[ConstraintNonneg()])
        model.fit(input,ST=endmembers)
        con = model.C_opt_
        con = con_norm(con)
        return con


def compute_correlation(input_data, endmember_spectrum):
    return pearsonr(input_data, endmember_spectrum)[0]

def get_pearsonr(input, gt, selected_bands, pure_spectra, state_size, thresh = 0.7, cpu_nums = 8, A_flag=False):
    aa, oa, kappa = 0, 0, 0   # 初始化。默认
    label = np.zeros([state_size*state_size])
    input = np.array(input)
    gt = np.array(gt)

    label_n = len(pure_spectra)# 标签数量-1   

    def process_pixel(s_n):
        input_data = input[s_n]
        h_l = [0]
        for j in range(label_n):
            c = compute_correlation(input_data, pure_spectra[j][selected_bands])
            h_l.append(c)
        if np.max(h_l) < thresh:
            h_l[0] = 1
        return np.argmax(h_l)
    results = Parallel(n_jobs=cpu_nums)(delayed(process_pixel)(s_n) for s_n in range(state_size*state_size))

    for index, output_label in enumerate(results):
        # i, s_n = divmod(index, 40000)
        label[index] = output_label

    oa = np.sum(label[gt!=0] == gt[gt!=0]) / len(gt[gt!=0])  #我们舍弃了背景的精度计算

    stat_label = np.zeros(label_n+1)
    n = gt.shape[0]
    for i_n in range(n):
        stat_label[gt[i_n]] += 1  # 统计当前场景下的label情况。实际上我可以存一个数组，而不是每次都计算，这里影响了效率了
    group_acc_record = np.zeros(label_n+1)  #统计8个分类 每个分类的精度
    assert len(label) == len(gt)      #确保label和gt长度一致 ，否则有误
    for i_n in range(n):
        if label[i_n] == gt[i_n]:
            group_acc_record[gt[i_n]] += 1   #统计所有当前类别下正确的情况

    for i_n in range(label_n+1):
        if stat_label[i_n] == 0:
            continue
        group_acc_record[i_n] /= stat_label[i_n] # 统计每一类的精度，含0的我们跳过，否则会除0报错
    aa = np.mean(group_acc_record[group_acc_record!=0][1:])
    if A_flag:
        kappa = cohen_kappa_score(label[gt!=0], gt[gt!=0]) #同样的，我们丢弃了背景0的计算
    return aa, oa, aa, kappa



