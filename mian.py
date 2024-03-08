'''
raw data is a csv file "od.zip", 
this main.py is an example for predicting future bus trip chain for 1000 users (specificly, say the 5000th to 6000th users in od.zip) by methods :Last,Ngrama,Graph_based_method
'''

from tools import emdedding_test_mutil_label,label_propagation_test_multi_label,holiday,validation_param_embedding,validation_param_lp,set_compare
import pandas as pd
from bus_model import N_gram,Last
import numpy as np
from sklearn.ensemble import RandomForestClassifier
res_ngarm,res_last,res_lp,res_lp_ml,res_lp_ml_c,res_eb,res_eb_ml,res_eb_mlc=[],[],[],[],[],[],[],[]
pre_ngarm,pre_last,pre_lp,pre_lp_ml,pre_lp_ml_c,pre_eb,pre_eb_ml,pre_eb_mlc=dict(),dict(),dict(),dict(),dict(),dict(),dict(),dict()
tab1 = pd.read_csv('od.zip')
'''cloumns as ['上车站点名字', '下车站点名字', '上车时间', 'week', 'day',
       'hour', 'id', 'day2'] where,day means the day in one week ,which {1:Monday,2:Tuseday ...},day2 means the day difference to 2018.1.1
       e.g.
        上车站点名字 下车站点名字    上车时间            week  day  hour       id      day2
0       荔香公园   科技园②   2018-01-02T08:20:16.000Z     0   1   8.0         XXXX     1
1       科技园①    南头①    2018-01-02T11:48:45.000Z     0    1  11.0         XXXX     1'''
tab1.dropna(inplace=True)
tab1.rename(columns={'上车站点名字': 'o', '下车站点名字': 'd'}, inplace=True)
loac = [0]+list(np.where((tab1['id'][:-1].values ==
                tab1['id'][1:].values) == 0)[0]+1)
tab1 = tab1.iloc[loac[4000]:loac[5001]]
loac = [0]+list(np.where((tab1['id'][:-1].values ==
                tab1['id'][1:].values) == 0)[0]+1)
id_best_lp=dict()
id_best_eb=dict()
for j in range(1000):
    data = tab1.iloc[loac[j]:loac[j+1]]
    i=j+5000
    day_od_dict = dict()
    for m in range(360):
        tmp = data[data['day2'] == m]
        if len(tmp) > 0:
            day_od_dict[m] = ['-'.join([str(j), k, l]) for j, k, l in zip(
                tmp['hour'].tolist(), tmp['o'].tolist(), tmp['d'].tolist())]
        else:
            day_od_dict[m] = ['<not>']
    best_a_lp, best_b_lp, best_c_lp, best_k_lp, best_lamba_lp = validation_param_lp(
        data, label_propagation_test_multi_label, [1], [10, 20], [10, 20], [2, 4], [1, 2], 250, 30)
    id_best_lp[data['id'].tolist()[0]] = [best_a_lp, best_b_lp,
                                        best_c_lp, best_k_lp, best_lamba_lp]
    best_a, best_b, best_c, best_lamba = validation_param_embedding(
        data, emdedding_test_mutil_label, [1], [10, 20], [10, 20], [1, 2], 250, 30, RandomForestClassifier(n_estimators=20))
    id_best_eb[data['id'].tolist()[0]] = [best_a, best_b, best_c, best_lamba]
    for pred_day in [7,14,28,56]:
        print('------pred day : {}------'.format(pred_day))
        pred = (Last(data[data['day2'] < 280]).prid(pred_day, 0))
        score = []
        for j_ in range(len(pred)):
            score.append(set_compare(set(pred[j_]), set(day_od_dict[j_+280])))
        res_last.append(np.mean(score))
        print('score lp:{} last:{}'.format(score, res_last[i]))
        pre_last[data['id'].tolist()[0]]=pred
        pred = N_gram(data[["day", "hour", "o", "d"]], 40).pred(pred_day, 0)
        score = []
        for j_ in range(len(pred)):
            score.append(set_compare(
                set(pred[j_]), set(day_od_dict[j_+280])))
        res_ngarm.append(np.mean(score))
        pre_ngarm[data['id'].tolist()[0]]=pred
        print('score ngrama:{} last:{}'.format(score, res_ngarm[i]))
       
        score, pred = label_propagation_test_multi_label(
            data, best_a_lp, best_b_lp, best_c_lp, 280, 7, best_k_lp)
        
        res_lp.append(score)
        print('score lp:{} '.format(score))
        score, pred = emdedding_test_mutil_label(
            data, best_a, best_b, best_c, 280, 7, RandomForestClassifier(n_estimators=20))
        print('score eb:{}'.format(score, res_last[i]))
        res_eb.append(score)