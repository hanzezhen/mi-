from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
import os
import scipy.io as sio
import numpy as np
import time
from tqdm import *

def load_data(person):
    basepath = os.getcwd()
    path = os.path.join(basepath,"person{}.mat".format(person))
    data_dic = sio.loadmat(path)
    data,label = data_dic["DATA"],data_dic["LABEL"]
    data = data.transpose((2, 0, 1))
    _label = np.zeros([label.shape[1]])
    for i in range(0, label.shape[1]):
        _label[i] = label[0, i]
    _label[0] = label[0, 0]
    return data,_label

def seperate_test(data,label,second,fs=250):
    begin = int(second*fs)
    end = int((second+1)*fs)
    train_data = data[:100,:,begin:end]
    train_label = label[:100]
    test_data,test_label =data[100:,:,begin:end],label[100:]
    return train_data,train_label,test_data,test_label

def get_acc(person=1):
    data,label = load_data(person)
    sec = 0
    seconds = []
    res = []
    while sec<=1.4:
        seconds.append(sec)
        sec = sec + 0.1
    for second in seconds:
        print(second)
        train_data,train_label,test_data,test_label = seperate_test(data, label, second)
        csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
        lda = LinearDiscriminantAnalysis()
        clf = Pipeline([('CSP', csp), ('LDA', lda)])
        clf.fit(train_data,train_label)
        print("第 {} 秒开始的数据分类正确率为 ：{}".format(second,clf.score(test_data,test_label)))
        res.append(clf.score(test_data,test_label))
    return res

def main(person):
    h1_list = get_acc(person)
    h0_list =[]
    for i in h1_list:
        j = 1-i
        h0_list.append(j)
    data,label = load_data(person)
    train_data,train_label,_,_=seperate_test(data, label, 0)
    csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    clf.fit(train_data, train_label)
    sec = 0
    seconds = []
    acc_dic = {}
    for i in range(20):
        acc_dic["test{}".format(i)] = []
    while sec <= 1.4:
        seconds.append(sec)
        sec = sec + 0.1
    j = 0
    with tqdm(total=len(seconds)*20) as pbar:
        for second in seconds:
            _,_,test_data,test_label=seperate_test(data, label, second)
            test_feature = csp.transform(test_data)
            p = lda.predict_proba(test_feature)
            for i in range(20):
                p_h1 = max(p[i][0],p[i][1])
                p_h0 = min(p[i][0],p[i][1])
                predict_ = 0 if p[i][0]>p[i][1] else 1
                pre_p = p_h1*h1_list[j]/(p_h1*h1_list[j]+p_h0*h0_list[j])
                acc_dic["test{}".format(i)].append([p_h1,p_h0,pre_p,predict_])
                pbar.update(1)
            j= j+1
    return acc_dic,test_label

if __name__ == '__main__':
    acc_dic,tl = main(2)
    j = 0
    for key,value in acc_dic.items():
        res = ""
        for i in value:
            res = res + "[{:.3f} {:.3f} {:.3f} {}] ".format(i[0],i[1],i[2],i[3])
        res = res + " label:{}".format(tl[j])
        print("{} acc: {}".format(key,res))