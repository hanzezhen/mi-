from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import signal
import matlab.engine
from mne.decoding import CSP
import sklearn.feature_selection

def butter_bandpass(lowcut,highcut,fs):
    nyq=0.5*fs
    low=lowcut/nyq
    high=highcut/nyq
    b,a = signal.butter(8,[low,high],'bandpass')
    return b,a

def butter_bandpass_filter(data,lowcut,highcut,fs):
    b,a=butter_bandpass(lowcut,highcut,fs)
    y=signal.filtfilt(b,a,data,axis=2)
    return y

def preprocess(data):
    pass

def load_data():
    eng1 = matlab.engine.start_matlab()



def main():
    data = load_data()
    data = preprocess(data["signal"])
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
