import librosa
import speechpy
import numpy as np
import librosa.display
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


class mfcc:
    def __init__(self, n_fff=1024, n_mels=40, n_mfcc=13, window='hamming',
                 mode_spicy=False, hop_length=0.023, win_length=0.011):

        self.n_fff = n_fff
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.window = window
        self.mode_spicy = mode_spicy
        self.hop_length = hop_length
        self.win_length = win_length
        plt.style.use('ggplot')

    def load_audio(self, path, mono=True, sample_rate=44100):
        if self.mode_spicy:
            sr, signal = wav.read(path)
            if mono:
                return (signal[:, 0], sr) if len(signal.shape) > 1 else (signal, sr)
            else:
                return signal, sr
        else:
            return librosa.load(path, mono=mono, sr=sample_rate)

    def get_spectogram(self, signal, sample_rate):
        if self.mode_spicy:
            return speechpy.feature.lmfe(signal, sample_rate, frame_length=self.hop_length,
                                         frame_stride=self.win_length, num_filters=self.n_mels,
                                         fft_length=self.n_fff)
        else:
            return librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=self.n_fff,
                                                  hop_length=round(self.hop_length * sample_rate),
                                                  win_length=round(self.win_length * sample_rate),
                                                  window=self.window, n_mels=self.n_mels)

    def get_mfcc(self, signal, sample_rate, mode_spicy=True):
        if self.mode_spicy:
            return speechpy.feature.mfcc(signal, sampling_frequency=sample_rate,
                                         frame_length=self.hop_length, frame_stride=self.win_length,
                                         num_cepstral=self.n_mfcc, num_filters=self.n_mels,
                                         fft_length=self.n_fff)
        else:
            if mode_spicy:
                return librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=self.n_mfcc)
            else:
                return librosa.feature.mfcc(S=librosa.power_to_db(signal), sr=sample_rate,
                                            n_mfcc=self.n_mfcc)

    def normalize_cmvn(self, mfcc, variance_normalization=True):
        if not self.mode_spicy:
            mfcc = np.transpose(mfcc)
        cmvn = speechpy.processing.cmvn(mfcc, variance_normalization=variance_normalization)
        return cmvn if self.mode_spicy else np.transpose(cmvn)

    def normalize_cmvnw(self, mfcc, sample_rate, variance_normalization=True):
        if not self.mode_spicy:
            mfcc = np.transpose(mfcc)
        cmvnw = speechpy.processing.cmvnw(mfcc, win_size=round(self.win_length * sample_rate),
                                         variance_normalization=variance_normalization)
        return cmvnw if self.mode_spicy else np.transpose(cmvnw)

    def get_delta_mfcc(self, mfcc, order=1):
        if self.mode_spicy:
            return speechpy.processing.derivative_extraction(mfcc, DeltaWindows=order)
        else:
            return librosa.feature.delta(mfcc, order=order)

    def show_spectogram(self, signal, sample_rate, title='Signal plot',
                        x_axis='time', y_axis='mel', colorbar='%+2.0f dB',
                        transpose=False, save=False, path_save=None):

        if self.mode_spicy or transpose:
            signal = np.transpose(signal)
        plt.title(title)
        librosa.display.specshow(signal, sr=sample_rate, hop_length=round(self.hop_length * sample_rate),
                                 n_fft=self.n_fff, win_length=round(self.win_length * sample_rate),
                                 x_axis=x_axis, y_axis=y_axis)
        plt.colorbar(format=colorbar)
        plt.show()
        if save:
            plt.tight_layout()
            plt.savefig(path_save)
        plt.clf()
        plt.close()

    def show_signal(self, signal, sample_rate, title='Signal plot',
                    x_axis='time', transpose=False, save=False,
                    path_save=None):

        if self.mode_spicy or transpose:
            signal = np.transpose(signal)
        plt.title(title)
        librosa.display.specshow(signal, sr=sample_rate, x_axis=x_axis)
        plt.show()
        if save:
            plt.tight_layout()
            plt.savefig(path_save)
        plt.clf()
        plt.close()

    def plot_signal(self, signal, title='Signal plot', transpose=False,
                    save=False, path_save=None):

        if not self.mode_spicy or transpose:
            signal = np.transpose(signal)
        plt.title(title)
        plt.plot(signal)
        plt.show()
        if save:
            plt.tight_layout()
            plt.savefig(path_save)
        plt.clf()
        plt.close()


mode_spicy = False
obj = mfcc(mode_spicy=mode_spicy)
signal, sr = obj.load_audio('../../data/audio/filter/268_16_ee05cc4e.wav')
spe = obj.get_spectogram(signal, sr)
mfcc = obj.get_mfcc(signal if mode_spicy else spe, sr, mode_spicy=False)
d1 = obj.get_delta_mfcc(mfcc, 1)
d2 = obj.get_delta_mfcc(mfcc, 2)

cont_all = np.concatenate((mfcc, d1, d2), axis=0 if mode_spicy else 1)
normsv = obj.normalize_cmvn(cont_all)
normwv = obj.normalize_cmvnw(cont_all, sr)

obj.plot_signal(spe)
obj.show_spectogram(spe, sr)
obj.plot_signal(mfcc)

obj.show_signal(mfcc, sr)
obj.plot_signal(d1)
obj.show_signal(d1, sr)
obj.plot_signal(d2)
obj.show_signal(d2, sr)

obj.plot_signal(cont_all)
obj.show_signal(cont_all, sr)
obj.plot_signal(normsv)
obj.show_signal(normsv, sr)
obj.plot_signal(normwv)
obj.show_signal(normwv, sr)

print(cont_all.shape)
