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
from sklearn import   manifold
from sklearn import linear_model
from sklearn import cluster,neighbors,tree
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns
from sklearn.semi_supervised import LabelPropagation
import editdistance
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
def get_next(gram_dict, prefix):
    anx = pd.DataFrame([gram_dict.keys(), gram_dict.values()]).T
    anx['csum'] = anx[1].cumsum()
    candidate = [i for i in list(gram_dict.keys()) if i[0] == prefix]
    if candidate == []:
        k = random.randint(0, anx['csum'].tolist()[-1])
        # print(anx,k)
        return anx[(anx['csum'] >= k)].iloc[0][0][1]
    else:
        r = []
        for i in candidate:
            r.append(gram_dict.get(i))
        idx = np.array(r).argmax()
    return candidate[idx][1]
class Last:
    """ 输入pandas dataframe，将最后一周的出行作为未来的预测
    """

    def __init__(self, input_data):
        self.data = input_data.copy()
        self.week_last = self.data['week'].tolist()[-1]
        self.day_last = self.data['day'].tolist()[-1]

    def prid(self, time, start_time):
        '''输出格式：每天的出行为一个列表，【time-o-d】组成的列表
        '''
        res = []
        for i in range(time):
            day = (i+start_time) % 7
            # week=self.week_last-(i+start_time)//7
            if day <= self.day_last:
                rr = self.data[(self.data['day'] == day) & (
                    self.data['week'] == self.week_last)]
            else:
                rr = self.data[(self.data['day'] == day) & (
                    self.data['week'] == self.week_last-1)]
            if len(rr) != 0:
                res.append(['-'.join([str(rr.iloc[i]['hour']), rr.iloc[i]
                           ['o'], rr.iloc[i]['d']]) for i in range(len(rr))])
            else:
                res += [['<not>']]
        return res
class N_gram:
    def __init__(self, input_data, input_week):
        '''
        input_data:pandas dataframe with column names "day", "hour", "o", and "d".
'''
        text_ = input_data.copy()
        res_ = [[str(s[0])+'-'+str(s[1]), s[2], s[3]]
                for s in [text_.iloc[i].tolist() for i in range(len(text_))]]
        res = []
        for i in res_:
            res += i
        self.t_dict = dict(collections.Counter(
            [(' '.join(res[3*i:3*i+3]), res[3*i+3]) for i in range(int(len(res)/3)-1)]))
        self.o_dict = dict(collections.Counter(
            [(' '.join(res[3*i:3*i+4]), res[3*i+4]) for i in range(int(len(res)/3)-1)]))
        self.d_dict = dict(collections.Counter(
            [(' '.join(res[3*i:3*i+5]), res[3*i+5]) for i in range(int(len(res)/3)-1)]))
        day_first = [input_data.iloc[0].tolist()]
        for i, j in zip(list(range(len(input_data)-1)), list(range(1, len(input_data)))):
            a, b = input_data.iloc[i].tolist(), input_data.iloc[j].tolist()
            if int(a[0]) != int(b[0]):
                day_first.append(b)
        self.day_prob = dict()
        self.day_first = pd.DataFrame(
            day_first, columns=['day', 'hour', 'o', 'd'])
        for i in range(7):
            self.day_prob[i] = len(
                self.day_first[self.day_first['day'] == i])/input_week

    def pred_one_day(self, day):
        if random.random() > self.day_prob[day]:
            return []
        else:
            prefix = self.day_first[self.day_first['day']
                                    == day].sample(1).iloc[0].tolist()
            prid = [str(prefix[0])+'-'+str(prefix[1]), prefix[2], prefix[3]]
            for i in range(100):
                prid.append(get_next(self.t_dict, ' '.join(prid[-3:])))
                prid.append(get_next(self.o_dict, ' '.join(prid[-4:])))
                prid.append(get_next(self.d_dict, ' '.join(prid[-5:])))
                if int(prid[-3][0]) != int(prid[-6][0]):
                    break
            return prid[:-3]


    def pred(self, time, start_day):
        '''
        return:list including daily travels where each day's travels are represented as a list of OD pairs in "hour-o-d" format.
                Days without any travels are represented as "<not>".
        '''
        res = []
        for i in range(time):
            day_pred = self.pred_one_day((start_day+i) % 7)
            if len(day_pred) > 0:
                res.append(['-'.join([day_pred[3*i], day_pred[3*i+1], day_pred[3*i+2]])[2:]
                           for i in range(int(len(day_pred)/3))])
            else:
                res.append(['<not>'])
        return res
