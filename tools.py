from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import numpy as np
import random
import re
import itertools
import math
import collections
from sklearn.model_selection import KFold
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score,log_loss,mean_squared_error
from matplotlib import pyplot as plt
from itertools import product
plt.rcParams['figure.figsize'] = [10, 8]  # 设置图片大小
plt.rcParams['figure.dpi'] = 100         # 设置图片分辨率
from sklearn import   manifold
from sklearn import linear_model
from sklearn import cluster,neighbors,tree
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns
from sklearn.semi_supervised import LabelPropagation
import editdistance
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
holiday = [0, 1, 2]+list(range(44, 44+7))+[94, 95, 96]+[118, 119,
                                                        120]+[168, 167, 166]+list(range(272, 272+7))+[266, 265, 264]
holiday = list(np.array(holiday))
def find_best_combination(class_prob, joint_likelihood, lamada=1):
    '''
    Given the probabilities for each category and the joint probability matrix for each pair of categories as input, 
    output the combination with the highest average total probability
    class_prob:list of pred_probability
    '''
    n = len(class_prob)
    best_likelihood = float('-inf')
    best_combination = None
    # 遍历所有可能的组合
    for length in range(1, n):
        for combination in itertools.combinations(range(n), length):
            # 计算该组合下的总似然
            k = 0
            likelihood = sum(class_prob[i] for i in combination)
            for i, j in itertools.combinations(combination, 2):
                likelihood += joint_likelihood[i][j]*lamada
                k += 1
            likelihood = likelihood/(len(combination)+k)
            # 更新最大似然和对应的组合
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_combination = combination
    return best_combination
def label_propagation(similarity_matrix, labels, k, alpha):
    """
    多标签标签传播算法
    :param similarity_matrix: 相似度矩阵
    :param labels: 样本的标签矩阵，1表示正样本，0表示负样本，-1表示未知标签
    :param k: 已知标签样本在标签传播中只根据最近的k个已知标签样本的标签进行更新
    :param alpha: 阿尔法参数
    :return: 所有样本的各个标签为正的概率矩阵
    """
    num_samples = len(similarity_matrix)
    prob_matrix = labels.copy().astype(np.float32)
    known_labels_idx = np.where(labels[:, 0] != -1)[0]  # 已知标签样本的下标
    label_prob = labels[np.where(labels[:, 0] != -1)[0]].mean(axis=0)
    prob_matrix[np.where(labels[:, 0] == -1)[0]] = label_prob
    prob_matrix_before = prob_matrix.copy()
    prob_matrix_before = prob_matrix.copy()
    for j in range(1000):
        for i in range(num_samples):
            if i in known_labels_idx:
                known_similarity_idx = np.argsort(similarity_matrix[i, known_labels_idx])[-1-k:-1]  # 已知标签样本中最相似的k个样本的下标
                prob = np.sum(
                    prob_matrix[known_similarity_idx], axis=0) / k  # 计算概率
            else:
                similarity_idx = np.argsort(similarity_matrix[i, known_labels_idx])[-1-k:-1]  # 最相似的k个样本的下标
                prob = np.sum(prob_matrix[similarity_idx], axis=0) / k  # 计算概率
            prob_matrix[i] = alpha * prob+(1-alpha)*prob_matrix[i]
        prob_matrix[known_labels_idx] = prob_matrix_before[known_labels_idx]
        if np.abs(prob_matrix-prob_matrix_before).sum() < 0.001:
            return prob_matrix
        else:
            prob_matrix_before = prob_matrix.copy()
    print('no convergence')
    return prob_matrix
def set_compare(set1, set2):
    intersection_size = len(set1 & set2)
    return intersection_size*2 / (len(set1)+len(set2))


def get_simmer_func(time_gap_param, day_param, work_param):
    def get_simmer(d1, d2):
        day_gep = abs(d1-d2)
        day1, day2 = d1 % 7, d2 % 7
        work1, work2 = int(day1 in [5, 6] or (d1 in holiday)), int(
            day2 in [5, 6] or (d2 in holiday))
        return 1/(time_gap_param*day_gep+day_param*int(day1 != day2)+work_param*int(work2 != work1)+1)
    return get_simmer



def label_propagation_test_multi_label(data, a, b, c, train_day, test_day, k, use_label_connection=True, lamba=1):
    ''' one day trip as one class
    data:dataframe,columns:['o','d','hour','day2'],day2:the order of the day/day difference of that day and 2018.1.1
    a,b,c:param in similarity func
    k:nearest sample number in lp
    return :mean score;pred od:as a list of od every day consisted of hour-o-d'''
    similarity_matrix = np.zeros((train_day+test_day, train_day+test_day))
    simmier_func = get_simmer_func(a, b, c)
    for i, j in product(range(train_day+test_day), range(train_day+test_day)):
        similarity_matrix[i, j] = simmier_func(i, j)
    data_od_dict = dict()
    for day in range(train_day):
        data_day = data[data['day2'] == day][['hour', 'o', 'd']]
        if len(data_day) != 0:
            res1 = ['-'.join([str(s[0]), s[1], s[2]])
                    for s in [data_day.iloc[i].tolist() for i in range(len(data_day))]]
        else:
            res1 = ['<not>']
        data_od_dict[day] = res1
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(list(data_od_dict.values()))
    tmp = []
    for od in data_od_dict.values():
        tmp += od
    od_list = mlb.inverse_transform(np.ones((1, len(set(tmp)))))[0]
    for day in range(train_day, train_day+test_day):
        data_day = data[data['day2'] == day][['hour', 'o', 'd']]
        if len(data_day) != 0:
            res1 = ['-'.join([str(s[0]), s[1], s[2]])
                    for s in [data_day.iloc[i].tolist() for i in range(len(data_day))]]
        else:
            res1 = ['<not>']
        data_od_dict[day] = res1
    y_ = np.concatenate([y, -1*np.ones((test_day, y.shape[1]))], axis=0)
    s = label_propagation(similarity_matrix, y_, k, 0.9)
    score = []
    data_od_dict2 = dict()
    for day in range(train_day):
        data_day = data[data['day2'] == day][['hour', 'o', 'd']]
        if len(data_day) != 0:
            res1 = ['-'.join([str(s[0]), s[1], s[2]])
                    for s in [data_day.iloc[i].tolist() for i in range(len(data_day))]]
        else:
            res1 = ['<not>']
        data_od_dict2[day] = res1
    if use_label_connection:
        pred = get_od_combination(
            s[train_day:train_day+test_day], od_list, data_od_dict2, test_day,lamba)
        for i in range(test_day):
            score.append(set_compare(
                set(pred[i]), set(data_od_dict[train_day+i])))
        return np.mean(score), pred
    else:
        pred = []
        for i in range(test_day):
            pred_oneday = np.where(s[train_day+i] > 0.5, 1, 0)
            qq = mlb.inverse_transform(np.array([pred_oneday]))[0]
            pred.append(qq)
            score.append(set_compare(
                set(pred[i]), set(data_od_dict[train_day+i])))
    return np.mean(score), pred
def get_od_combination(pred_prod, od_list, day_od_dict, test_day,lamba=1):
    '''
    pred_prob:test_day*od_num
    day_od_dict:The dictionary has keys that represent dates in sorted order,
                 and the corresponding values are lists of OD pairs represented as "hour-o-d".
    od_list:The list of OD pairs maintains the same order as the order of OD pairs in pred_prod.
    '''
    gram_dict = collections.Counter()
    for i in day_od_dict.values():
        if len(list(i)) > 1:
            gram_dict += collections.Counter([(k, l)
                                             for k, l in zip(list(i)[:-1], list(i)[1:])])
            gram_dict += collections.Counter([(l, k)
                                             for k, l in zip(list(i)[:-1], list(i)[1:])])
    gram_dict = dict(gram_dict)
    joint_prob = np.zeros((len(od_list), len(od_list)))
    for i, j in itertools.product(range(len(joint_prob)), range(len(joint_prob))):
        joint_prob[i, j] = gram_dict.get((od_list[i], od_list[j]), 0)
    joint_prob0,joint_prob1=joint_prob.sum(axis=0),joint_prob.sum(axis=1)
    for i,j in itertools.product(range(len(joint_prob)),range(len(joint_prob))):
        joint_prob[i,j] = joint_prob[i,j]/(joint_prob0[i]+joint_prob1[j]+0.001)
    prid = []
    for c in range(test_day):
        class_prob = pred_prod[c, [np.where(pred_prod[c, :] > 0.3)[0]]][0]
        class_order = np.where(pred_prod[c, :] > 0.3)[0]
        if len(class_order) > 8:
            class_order = np.argsort(pred_prod[c, :])[-8:]
            class_prob = pred_prod[c, class_order]

        else:
            pass
        if len(class_order) > 1:
            best_combination = find_best_combination(
                class_prob, joint_prob[list(class_order)][:, list(class_order)], lamba)
            prid.append([od_list[order]
                        for order in (class_order[list(best_combination)])])
        elif len(class_order) == 1:
            prid.append([od_list[class_order[0]]])
        else:
            prid.append(['<not>'])
    return prid


def emdedding_test_mutil_label(data, a, b, c, train_day, test_day, classifier, with_label_connection=True,lambe=1):
    '''
    classifier:sklearn classifier
    '''
    dissimilarity_matrix = np.zeros((train_day+test_day, train_day+test_day))
    day_od_dict = dict()
    for i in range(train_day):
        tmp = data[data['day2'] == i]
        if len(tmp) > 0:
            day_od_dict[i] = ['-'.join([str(j), k, l]) for j, k, l in zip(
                tmp['hour'].tolist(), tmp['o'].tolist(), tmp['d'].tolist())]
        else:
            day_od_dict[i] = ['<not>']
    mlb = MultiLabelBinarizer()
    day_od_dict2 = day_od_dict.copy()
    y = mlb.fit_transform(list(day_od_dict.values()))
    tmp = []
    for od in day_od_dict.values():
        tmp += od
    od_list = mlb.inverse_transform(np.ones((1, len(set(tmp)))))[0]
    for i in range(train_day, train_day+test_day):
        tmp = data[data['day2'] == i]
        if len(tmp) > 0:
            day_od_dict[i] = ['-'.join([str(j), k, l]) for j, k, l in zip(
                tmp['hour'].tolist(), tmp['o'].tolist(), tmp['d'].tolist())]
        else:
            day_od_dict[i] = ['<not>']
    similarity_func = get_simmer_func(a, b, c)
    for i, j in product(range(train_day+test_day), range(train_day+test_day)):
        dissimilarity_matrix[i, j] = similarity_func(i, j)
    mds = manifold.SpectralEmbedding(n_components=10, affinity='precomputed')
    X = mds.fit_transform(dissimilarity_matrix)
    X_train = X[:train_day]
    y_train = y[:train_day]
    X_test = X[train_day:train_day+test_day]
    multi_rf = MultiOutputClassifier(classifier)
    score = []
    multi_rf.fit(X_train, y_train)
    if with_label_connection:
        y_pred = np.array(multi_rf.predict_proba(X_test))[:, :, 1].T
        pred = get_od_combination(
            y_pred, od_list, day_od_dict2, test_day, lambe)
        for i in range(test_day):
            score.append(set_compare(
                set(pred[i]), set(day_od_dict[train_day+i])))
        return np.mean(score), pred
    else:
        pred = mlb.inverse_transform(multi_rf.predict(X_test))
        for i in range(test_day):
            score.append(set_compare(
                set(pred[i]), set(day_od_dict[train_day+i])))
    return np.mean(score), pred
def validation_param_lp(data, func, a_list, b_list, c_list, k_list, lamba_list, train_day, test_day):
    best_score = 0
    for a, b, c, k,lamba in itertools.product(a_list,b_list,c_list,k_list,lamba_list):
        score, pred = func(
            data, a, b, c, train_day, test_day, k,True,lamba)

        if score > best_score:
            best_score = score
            best_a = a
            best_b = b
            best_c = c
            best_k = k
            best_lamba=lamba
    return best_a,best_b,best_c,best_k,best_lamba


def validation_param_embedding(data, func, a_list, b_list, c_list, lamba_list, train_day, test_day, classifier):
    best_score = 0
    for a, b, c,lamba in itertools.product(a_list, b_list, c_list,lamba_list):
        score, pred = func(
            data, a, b, c, train_day, test_day, classifier,True,lamba)

        if score > best_score:
            best_score = score
            best_a = a
            best_b = b
            best_c = c
            best_lamba = lamba
    return best_a, best_b, best_c,best_lamba