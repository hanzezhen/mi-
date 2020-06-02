from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import signal
from mne.decoding import CSP
import sklearn.feature_selection
import argparse
import scipy.io as sio
import os
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def butter_bandpass(lowcut,highcut,fs):
    nyq=0.5*fs
    low=lowcut/nyq
    high=highcut/nyq
    b,a = signal.butter(8,[low,high],'bandpass')
    return b,a

def butter_bandpass_filter(data,lowcut,highcut,fs=250):
    b,a=butter_bandpass(lowcut,highcut,fs)
    y=signal.filtfilt(b,a,data,axis=2)
    return y

def selectBounds(data,label,bounds_list):
    n_components = 4
    csp=CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    num = len(bounds_list)
    all_features = np.zeros([120,n_components*num])
    i = 0
    for bound in bounds_list:
        data = butter_bandpass_filter(data,bound[0],bound[1])
        data_features = csp.fit_transform(data, label)
        all_features[:,i:i+4] = data_features
        i = i + 4
    select_K = sklearn.feature_selection.SelectKBest(mutual_info_classif, k=10).fit(all_features, label)
    selected_list = select_K.get_support(indices=True)
    selected_bounds = [] #[ 4  5 10 17 20 21 23 31 33 39]
    selected_dic = {}
    for i in range(len(selected_list)):
        bound_index = (selected_list[i]+1)%4 -1
        if bound_index in selected_dic:
            selected_dic[bound_index] = selected_dic[bound_index] +1
        else:
            selected_dic[bound_index] = 1
    tmp = sorted(selected_dic.items(),key=lambda x: x[1], reverse=True)
    for i in tmp:
        selected_bounds.append(bounds_list[i[0]])
    return selected_bounds

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


def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--person",type=int,choices=[x for x in range(14)],default=0)
    parser.add_argument("-l","--lowcut",type=float,default=5)
    parser.add_argument("-r","--highcut",type=float,default=45)
    parser.add_argument("-n","--number",type=int,default=20)
    args = parser.parse_args()
    return args

def calBounds(lowcut,highcut,num_of_bounds):
    width = (highcut+lowcut)/num_of_bounds
    res = []
    begin = lowcut
    for i in range(num_of_bounds):
        end = begin+width
        res.append([begin,end])
        begin = end
    return res

def print_res(selected_bounds,person):
    assert person>0
    print("person:{} 根据CSP和互信息熵频带挑选结果如下:".format(person))
    j = 1
    for i in selected_bounds:
        print("{} :: [{}-{}]".format(j,i[0],i[1]))
        j=j+1

def main():
    parse = _parse()

    lowcut = parse.lowcut
    highcut = parse.highcut
    assert lowcut<highcut
    num_of_bounds = parse.number

    bounds_list = calBounds(lowcut,highcut,num_of_bounds)

    persons = parse.person

    if not persons:
        personlist = list(range(1,13))
    else:
        personlist = [persons]

    for person in personlist:
        data,label = load_data(person)
        selected_bounds = selectBounds(data,label,bounds_list)
        print_res(selected_bounds,person)
        print()

if __name__ == '__main__':
    main()