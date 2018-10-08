# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : libri_preprocess.py
# Description  : Feature preprocessing for LibriSpeech dataset
# ******************************************************

import os
import glob
import sklearn
import argparse
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
from subprocess import check_call, CalledProcessError
# from speechvalley.feature.core import calcfeat_delta_delta
import numpy
from scipy.fftpack import dct
import numpy
import math

try:
    xrange(1)
except:
    xrange=range



def audio2frame(signal,frame_length,frame_step,winfunc=lambda x:numpy.ones((x,))):
    """ Framing audio signal. Uses numbers of samples as unit.

    Args:
    signal: 1-D numpy array.
	frame_length: In this situation, frame_length=samplerate*win_length, since we
        use numbers of samples as unit.
    frame_step:In this situation, frame_step=samplerate*win_step,
        representing the number of samples between the start point of adjacent frames.
	winfunc:lambda function, to generate a vector with shape (x,) filled with ones.

    Returns:
        frames*win: 2-D numpy array with shape (frames_num, frame_length).
    """
    signal_length=len(signal)
    # Use round() to ensure length and step are integer, considering that we use numbers
    # of samples as unit.
    frame_length=int(round(frame_length))
    frame_step=int(round(frame_step))
    if signal_length<=frame_length:
        frames_num=1
    else:
        frames_num=1+int(math.ceil((1.0*signal_length-frame_length)/frame_step))
    pad_length=int((frames_num-1)*frame_step+frame_length)
    # Padding zeros at the end of signal if pad_length > signal_length.
    zeros=numpy.zeros((pad_length-signal_length,))
    pad_signal=numpy.concatenate((signal,zeros))
    # Calculate the indice of signal for every sample in frames, shape (frams_nums, frams_length)
    indices=numpy.tile(numpy.arange(0,frame_length),(frames_num,1))+numpy.tile(
        numpy.arange(0,frames_num*frame_step,frame_step),(frame_length,1)).T
    indices=numpy.array(indices,dtype=numpy.int32)
    # Get signal data according to indices.
    frames=pad_signal[indices]
    win=numpy.tile(winfunc(frame_length),(frames_num,1))
    return frames*win

def deframesignal(frames,signal_length,frame_length,frame_step,winfunc=lambda x:numpy.ones((x,))):
    '''定义函数对原信号的每一帧进行变换，应该是为了消除关联性
    参数定义：
    frames:audio2frame函数返回的帧矩阵
    signal_length:信号长度
    frame_length:帧长度
    frame_step:帧间隔
    winfunc:对每一帧加window函数进行分析，默认此处不加window
    '''
    #对参数进行取整操作
    signal_length=round(signal_length) #信号的长度
    frame_length=round(frame_length) #帧的长度
    frames_num=numpy.shape(frames)[0] #帧的总数
    assert numpy.shape(frames)[1]==frame_length,'"frames"矩阵大小不正确，它的列数应该等于一帧长度'  #判断frames维度，处理异常
    indices=numpy.tile(numpy.arange(0,frame_length),(frames_num,1))+numpy.tile(numpy.arange(0,frames_num*frame_step,frame_step),(frame_length,1)).T  #相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
    indices=numpy.array(indices,dtype=numpy.int32)
    pad_length=(frames_num-1)*frame_step+frame_length #铺平后的所有信号
    if signal_length<=0:
        signal_length=pad_length
    recalc_signal=numpy.zeros((pad_length,)) #调整后的信号
    window_correction=numpy.zeros((pad_length,1)) #窗关联
    win=winfunc(frame_length)
    for i in range(0,frames_num):
        window_correction[indices[i,:]]=window_correction[indices[i,:]]+win+1e-15 #表示信号的重叠程度
        recalc_signal[indices[i,:]]=recalc_signal[indices[i,:]]+frames[i,:] #原信号加上重叠程度构成调整后的信号
    recalc_signal=recalc_signal/window_correction #新的调整后的信号等于调整信号处以每处的重叠程度
    return recalc_signal[0:signal_length] #返回该新的调整信号

def spectrum_magnitude(frames,NFFT):
    '''Apply FFT and Calculate magnitude of the spectrum.
    Args:
        frames: 2-D frames array calculated by audio2frame(...).
        NFFT:FFT size.
    Returns:
        Return magnitude of the spectrum after FFT, with shape (frames_num, NFFT).
    '''
    complex_spectrum=numpy.fft.rfft(frames,NFFT)
    return numpy.absolute(complex_spectrum)

def spectrum_power(frames,NFFT):
    """Calculate power spectrum for every frame after FFT.
    Args:
        frames: 2-D frames array calculated by audio2frame(...).
        NFFT:FFT size
    Returns:
        Power spectrum: PS = magnitude^2/NFFT
    """
    return 1.0/NFFT * numpy.square(spectrum_magnitude(frames,NFFT))

def log_spectrum_power(frames,NFFT,norm=1):
    '''Calculate log power spectrum.
    Args:
        frames:2-D frames array calculated by audio2frame(...)
        NFFT：FFT size
        norm: Norm.
    '''
    spec_power=spectrum_power(frames,NFFT)
    # In case of calculating log0, we set 0 in spec_power to 0.
    spec_power[spec_power<1e-30]=1e-30
    log_spec_power=10*numpy.log10(spec_power)
    if norm:
        return log_spec_power-numpy.max(log_spec_power)
    else:
        return log_spec_power

def pre_emphasis(signal,coefficient=0.95):
    '''Pre-emphasis.
    Args:
        signal: 1-D numpy array.
        coefficient:Coefficient for pre-emphasis. Defauted to 0.95.
    Returns:
        pre-emphasis signal.
    '''
    return numpy.append(signal[0],signal[1:]-coefficient*signal[:-1])


def calcfeat_delta_delta(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True,mode='mfcc',feature_len=13):
    """Calculate features, fist order difference, and second order difference coefficients.
        13 Mel-Frequency Cepstral Coefficients(MFCC), 13 first order difference
       coefficients, and 13 second order difference coefficients. There are 39 features
       in total.

    Args:
        signal: 1-D numpy array.
        samplerate: Sampling rate. Defaulted to 16KHz.
        win_length: Window length. Defaulted to 0.025, which is 25ms/frame.
        win_step: Interval between the start points of adjacent frames.
            Defaulted to 0.01, which is 10ms.
        feature_len: Numbers of features. Defaulted to 13.
        filters_num: Numbers of filters. Defaulted to 26.
        NFFT: Size of FFT. Defaulted to 512.
        low_freq: Lowest frequency.
        high_freq: Highest frequency.
        pre_emphasis_coeff: Coefficient for pre-emphasis. Pre-emphasis increase
            the energy of signal at higher frequency. Defaulted to 0.97.
        cep_lifter: Numbers of lifter for cepstral. Defaulted to 22.
        appendEnergy: Wheter to append energy. Defaulted to True.
        mode: 'mfcc' or 'fbank'.
            'mfcc': Mel-Frequency Cepstral Coefficients(MFCC).
                    Complete process: Mel filtering -> log -> DCT.
            'fbank': Apply Mel filtering -> log.

    Returns:
        2-D numpy array with shape:(NUMFRAMES, 39). In each frame, coefficients are
            concatenated in (feature, delta features, delta delta feature) way.
    """
    filters_num = 2*feature_len
    feat = calcMFCC(signal, samplerate, win_length, win_step, feature_len, filters_num, NFFT, low_freq, high_freq,
                    pre_emphasis_coeff, cep_lifter, appendEnergy, mode=mode)   # 首先获取13个一般MFCC系数
    feat_delta = delta(feat)
    feat_delta_delta = delta(feat_delta)

    result = numpy.concatenate((feat,feat_delta,feat_delta_delta),axis=1)
    return result

def delta(feat, N=2):
    """Compute delta features from a feature vector sequence.

    Args:
        feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
        N: For each frame, calculate delta features based on preceding and following N frames.
    Returns:
        A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    NUMFRAMES = len(feat)
    feat = numpy.concatenate(([feat[0] for i in range(N)], feat, [feat[-1] for i in range(N)]))
    denom = sum([2*i*i for i in range(1,N+1)])
    dfeat = []
    for j in range(NUMFRAMES):
        dfeat.append(numpy.sum([n*feat[N+j+n] for n in range(-1*N,N+1)], axis=0)/denom)
    return dfeat

def calcMFCC(signal,samplerate=16000,win_length=0.025,win_step=0.01,feature_len=13,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,appendEnergy=True,mode='mfcc'):
    """Caculate Features.
    Args:
        signal: 1-D numpy array.
        samplerate: Sampling rate. Defaulted to 16KHz.
        win_length: Window length. Defaulted to 0.025, which is 25ms/frame.
        win_step: Interval between the start points of adjacent frames.
            Defaulted to 0.01, which is 10ms.
        feature_len: Numbers of features. Defaulted to 13.
        filters_num: Numbers of filters. Defaulted to 26.
        NFFT: Size of FFT. Defaulted to 512.
        low_freq: Lowest frequency.
        high_freq: Highest frequency.
        pre_emphasis_coeff: Coefficient for pre-emphasis. Pre-emphasis increase
            the energy of signal at higher frequency. Defaulted to 0.97.
        cep_lifter: Numbers of lifter for cepstral. Defaulted to 22.
        appendEnergy: Wheter to append energy. Defaulted to True.
        mode: 'mfcc' or 'fbank'.
            'mfcc': Mel-Frequency Cepstral Coefficients(MFCC).
                    Complete process: Mel filtering -> log -> DCT.
            'fbank': Apply Mel filtering -> log.

    Returns:
        2-D numpy array with shape (NUMFRAMES, features). Each frame containing feature_len of features.
    """
    filters_num = 2*feature_len
    feat,energy=fbank(signal,samplerate,win_length,win_step,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff)
    feat=numpy.log(feat)
    # Performing DCT and get first 13 coefficients
    if mode == 'mfcc':
        feat=dct(feat,type=2,axis=1,norm='ortho')[:,:feature_len]
        feat=lifter(feat,cep_lifter)
    elif mode == 'fbank':
        feat = feat[:,:feature_len]
    if appendEnergy:
        # Replace the first coefficient with logE and get 2-13 coefficients.
        feat[:,0]=numpy.log(energy)
    return feat

def fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    """Perform pre-emphasis -> framing -> get magnitude -> FFT -> Mel Filtering.
    Args:
        signal: 1-D numpy array.
        samplerate: Sampling rate. Defaulted to 16KHz.
        win_length: Window length. Defaulted to 0.025, which is 25ms/frame.
        win_step: Interval between the start points of adjacent frames.
            Defaulted to 0.01, which is 10ms.
        cep_num: Numbers of cepstral coefficients. Defaulted to 13.
        filters_num: Numbers of filters. Defaulted to 26.
        NFFT: Size of FFT. Defaulted to 512.
        low_freq: Lowest frequency.
        high_freq: Highest frequency.
        pre_emphasis_coeff: Coefficient for pre-emphasis. Pre-emphasis increase
            the energy of signal at higher frequency. Defaulted to 0.97.
    Returns:
        feat: Features.
        energy: Energy.
    """
    # Calculate the highest frequency.
    high_freq=high_freq or samplerate/2
    # Pre-emphasis
    signal=pre_emphasis(signal,pre_emphasis_coeff)
    # rames: 2-D numpy array with shape (frame_num, frame_length)
    frames=audio2frame(signal,win_length*samplerate,win_step*samplerate)
    # Caculate energy and modify all zeros to eps.
    spec_power=spectrum_power(frames,NFFT)
    energy=numpy.sum(spec_power,1)
    energy=numpy.where(energy==0,numpy.finfo(float).eps,energy)
    # Get Mel filter banks.
    fb=get_filter_banks(filters_num,NFFT,samplerate,low_freq,high_freq)
    # Get MFCC and modify all zeros to eps.
    feat=numpy.dot(spec_power,fb.T)
    feat=numpy.where(feat==0,numpy.finfo(float).eps,feat)

    return feat,energy

def log_fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    """Calculate log of features.
    """
    feat,energy=fbank(signal,samplerate,win_length,win_step,filters_num,NFFT,low_freq,high_freq,pre_emphasis_coeff)
    return numpy.log(feat)

def ssc(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    '''
    待补充
    '''
    high_freq=high_freq or samplerate/2
    signal=pre_emphasis(signal,pre_emphasis_coeff)
    frames=audio2frame(signal,win_length*samplerate,win_step*samplerate)
    spec_power=spectrum_power(frames,NFFT)
    spec_power=numpy.where(spec_power==0,numpy.finfo(float).eps,spec_power) #能量谱
    fb=get_filter_banks(filters_num,NFFT,samplerate,low_freq,high_freq)
    feat=numpy.dot(spec_power,fb.T)  #计算能量
    R=numpy.tile(numpy.linspace(1,samplerate/2,numpy.size(spec_power,1)),(numpy.size(spec_power,0),1))
    return numpy.dot(spec_power*R,fb.T)/feat

def hz2mel(hz):
    """Convert frequency to Mel frequency.
    Args:
        hz: Frequency.
    Returns:
        Mel frequency.
    """
    return 2595*numpy.log10(1+hz/700.0)

def mel2hz(mel):
    """Convert Mel frequency to frequency.
    Args:
        mel:Mel frequency
    Returns:
        Frequency.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filter_banks(filters_num=26,NFFT=512,samplerate=16000,low_freq=0,high_freq=None):
    '''Calculate Mel filter banks.
    Args:
        filters_num: Numbers of Mel filters.
        NFFT:FFT size. Defaulted to 512.
        samplerate: Sampling rate. Defaulted to 16KHz.
        low_freq: Lowest frequency.
        high_freq: Highest frequency.
    '''
    # Convert frequency to Mel frequency.
    low_mel=hz2mel(low_freq)
    high_mel=hz2mel(high_freq)
    # Insert filters_num of points between low_mel and high_mel. In total there are filters_num+2 points
    mel_points=numpy.linspace(low_mel,high_mel,filters_num+2)
    # Convert Mel frequency to frequency and find corresponding position.
    hz_points=mel2hz(mel_points)
    # Find corresponding position of these hz_points in fft.
    bin=numpy.floor((NFFT+1)*hz_points/samplerate)
    # Build Mel filters' expression.First and third points of each filter are zeros.

    # fbank=numpy.zeros([filters_num,NFFT/2+1])
    fbank = numpy.zeros([26, 256 + 1])
    for j in xrange(0,filters_num):
        for i in xrange(int(bin[j]),int(bin[j+1])):
            fbank[j,i]=(i-bin[j])/(bin[j+1]-bin[j])
        for i in xrange(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i]=(bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra,L=22):
    '''Lifter function.
    Args:
        cepstra: MFCC coefficients.
        L: Numbers of lifters. Defaulted to 22.
    '''
    if L>0:
        nframes,ncoeff=numpy.shape(cepstra)
        n=numpy.arange(ncoeff)
        lift=1+(L/2)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        return cepstra


def preprocess(root_directory):
    """
    Function to walk through the directory and convert flac to wav files
    """
    try:
        check_call(['flac'])
    except OSError:
        raise OSError("""Flac not installed. Install using apt-get install flac""")
    for subdir, dirs, files in os.walk(root_directory):
        for f in files:
            filename = os.path.join(subdir, f)
            if f.endswith('.flac'):
                try:
                    check_call(['flac', '-d', filename])
                    os.remove(filename)
                except CalledProcessError as e:
                    print("Failed to convert file {}".format(filename))
            elif f.endswith('.TXT'):
                os.remove(filename)
            elif f.endswith('.txt'):
                with open(filename, 'r') as fp:
                    lines = fp.readlines()
                    for line in lines:
                        sub_n = line.split(' ')[0] + '.label'
                        subfile = os.path.join(subdir, sub_n)
                        sub_c = ' '.join(line.split(' ')[1:])
                        sub_c = sub_c.lower()
                        with open(subfile, 'w') as sp:
                            sp.write(sub_c)
            elif f.endswith('.wav'):
                if not os.path.isfile(os.path.splitext(filename)[0] +
                                      '.label'):
                    raise ValueError(".label file not found for {}".format(filename))
            else:
                pass


def wav2feature(root_directory, save_directory, name, win_len, win_step, mode, feature_len, seq2seq, save):
    count = 0
    dirid = 0
    level = 'cha' if seq2seq is False else 'seq2seq'
    data_dir = os.path.join(root_directory, name)
    preprocess(data_dir)
    for subdir, dirs, files in os.walk(data_dir):       # 遍历 data_dir 中包括中所有文件
        for f in files:
            fullFilename = os.path.join(subdir, f)
            filenameNoSuffix = os.path.splitext(fullFilename)[0]  # 删除后缀取前面文件路径
            if f.endswith('.wav'):                                # 以‘wav’后缀结尾
                rate = None
                sig = None
                try:
                    (rate, sig) = wav.read(fullFilename)
                except ValueError as e:
                    if e.message == "File format 'NIST'... not understood.":
                        sf = Sndfile(fullFilename, 'r')
                    nframes = sf.nframes
                    sig = sf.read_frames(nframes)
                    rate = sf.samplerate
                feat = calcfeat_delta_delta(sig, rate, win_length=win_len, win_step=win_step, mode=mode, feature_len=feature_len)
                feat = preprocessing.scale(feat)  # 标准化
# sklearn.preprocessing.scale(X, axis=0, with_mean=True,with_std=True,copy=True)
                feat = np.transpose(feat)  # 转置
                print(feat.shape)          # 返回行列
                labelFilename = filenameNoSuffix + '.label'
                with open(labelFilename, 'r') as f:
                    characters = f.readline().strip().lower()
                targets = []
                if seq2seq is True:
                    targets.append(28)
                for c in characters:
                    if c == ' ':
                        targets.append(0)
                    elif c == "'":                 # 英文中的won't等
                        targets.append(27)
                    else:
                        targets.append(ord(c)-96)  # 因为a从97开始
                if seq2seq is True:
                    targets.append(29)
                print(targets)
                if save:
                    count += 1
                    if count % 4000 == 0:
                        dirid += 1
                    print('file index:', count)
                    print('dir index:', dirid)
                    label_dir = os.path.join(save_directory, level, name, str(dirid), 'label')
                    feat_dir = os.path.join(save_directory, level, name, str(dirid), 'feature')
                    if not os.path.isdir(label_dir):
                        os.makedirs(label_dir)
                    if not os.path.isdir(feat_dir):
                        os.makedirs(feat_dir)
                    featureFilename = os.path.join(feat_dir, filenameNoSuffix.split('/')[-1] +'.npy')
                    np.save(featureFilename, feat)
                    t_f = os.path.join(label_dir, filenameNoSuffix.split('/')[-1] +'.npy')
                    print(t_f)
                    np.save(t_f, targets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='libri_preprocess',
                                     description='Script to preprocess libri data')
    parser.add_argument("path", help="Directory of LibriSpeech dataset", type=str)

    parser.add_argument("save", help="Directory where preprocessed arrays are to be saved",
                        type=str)
    parser.add_argument("-n", "--name", help="Name of the dataset",
                        choices=['dev-clean', 'dev-other', 'test-clean',
                                 'test-other', 'train-clean-100', 'train-clean-360',
                                 'train-other-500'], type=str, default='dev-clean')

    parser.add_argument("-m", "--mode", help="Mode",
                        choices=['mfcc', 'fbank'],
                        type=str, default='mfcc')
    parser.add_argument("--featlen", help='Features length', type=int, default=13)
    parser.add_argument("-s", "--seq2seq", default=False,
                        help="set this flag to use seq2seq", action="store_true")

    parser.add_argument("-wl", "--winlen", type=float,
                        default=0.02, help="specify the window length of feature")

    parser.add_argument("-ws", "--winstep", type=float,
                        default=0.01, help="specify the window step length of feature")

    args = parser.parse_args()
    root_directory = args.path
    save_directory = args.save
    mode = args.mode
    feature_len = args.featlen
    seq2seq = args.seq2seq
    name = args.name
    win_len = args.winlen
    win_step = args.winstep

    if root_directory == '.':
        root_directory = os.getcwd()

    if save_directory == '.':
        save_directory = os.getcwd()

    if not os.path.isdir(root_directory):
        raise ValueError("LibriSpeech Directory does not exist!")

    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    wav2feature(root_directory, save_directory, name=name, win_len=win_len, win_step=win_step,
                mode=mode, feature_len=feature_len, seq2seq=seq2seq, save=True)
