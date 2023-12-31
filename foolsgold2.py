import numpy as np
import random
import pdb
import logging
import sklearn.metrics.pairwise as smp

# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads, iffg_smooth):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning

    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

    v = np.max(cs, axis=1)

    # print(v.shape)
    temp = 1
    temp_max = 0

    # 找到最小的vi
    for i in range(len(v)):
        if v[i] < temp:
            temp = v[i]
            mini_v_ind = i + 1
        #     找出最大的vi用于下一步的归一化
        # if v[i] > temp_max:
        #     temp_max = v[i]
        #     max_v_ind = i + 1
    # main.logger.info(f"最小的v为第:{mini_v_ind}个客户\n")
    # temp_v = 0
    # for i in range(n_clients):
    #     temp_v = temp_v + v[i]
    #     # 归一化
    # for i in range(n_clients):
    #     v[i] = v[i]/temp_v
    #     # v[i] = (v[i]/temp_v)/v[max_v_ind]

    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99




    # Logit function
    wv = (np.log(wv / (1 - wv)+ 1e-6) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    temp = 0
    for i in range(n_clients):
        temp = temp + v[i]
        # 归一化
    for i in range(n_clients):
        wv[i] = wv[i]/temp

    # 平滑
    temp = 0

    episl = random.uniform(0, 0.0001)

    for i in range(n_clients):
        wv[i] = wv[i] + episl
            # wv[i] = (wv[i] + 1)/(1 + n_clients)
        temp = temp + wv[i]
        # 归一化
    for i in range(n_clients):
        wv[i] = wv[i]/temp


    return wv, v, mini_v_ind, n_clients