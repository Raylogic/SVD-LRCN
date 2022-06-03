import numpy as np
from chainer import config
from librosa.core import stft, load, istft
from pydub import AudioSegment
import soundfile as sf
import Network

SR = 16000
H = 512
FFT_SIZE = 1024
BATCH_SIZE = 64
PATCH_LENGTH = 128


def LoadAudio(fname):
    y, sr = load(fname, sr=SR)
    dur = len(y) / sr
    y = np.concatenate((y, np.zeros(sr * 60)))
    spec = stft(y, n_fft=FFT_SIZE, hop_length=H, win_length=FFT_SIZE)
    mag = np.abs(spec)
    mag /= np.max(mag)
    phase = np.exp(1.j * np.angle(spec))
    return mag, phase, dur


def SaveAudio(fname, mag, phase, dur):
    y = istft(mag * phase, hop_length=H, win_length=FFT_SIZE)
    # REVISAR ESTA LÍNEA DE CÓDIGO PARA AJUSTES
    sf.write(fname, y[:int(dur * SR)], SR)
    

def ComputeMask(input_mag, unet_model="unet.model", hard=True):
    unet = Network.UNet()
    unet.load(unet_model)
    config.train = False
    config.enable_backprop = False
    mask = unet(input_mag[np.newaxis, np.newaxis, 1:, :]).data[0, 0, :, :]
    mask = np.vstack((np.zeros(mask.shape[1], dtype="float32"), mask))
    if hard:
        hard_mask = np.zeros(mask.shape, dtype="float32")
        hard_mask[mask > 0.5] = 1
        return hard_mask
    else:
        return mask


def convert_wav_mp3(wav_path):
    song = AudioSegment.from_file(wav_path, 'wav')
    song.export(wav_path.replace('.wav', '.mp3'), 'mp3')
    return wav_path.replace('.wav', '.mp3')


def use_model_get_VA(mag, start, end):
    mask = ComputeMask(mag[:, start:end], unet_model="unet.model", hard=False)
    V = mag[:, start:end] * mask
    A = mag[:, start:end] * (1 - mask)
    return V, A


def split_vocal(fname):
    mag, phase, dur = LoadAudio(fname)

    start = 0
    end = mag.shape[-1] // 1024 * 1024
    V = 'V'
    A = 'A'
    for item in range(start, end, 1024):
        if type(V) == str and type(A) == str:
            V, A = use_model_get_VA(mag, item, item + 1024)
        else:
            Vv, Aa = use_model_get_VA(mag, item, item + 1024)
            V = np.hstack((V, Vv))
            A = np.hstack((A, Aa))

    SaveAudio(
        "%s.unet.Vocal.wav" % fname[:-4], V, phase[:, start:end], dur)
    # SaveAudio(
    #     "%s-A.wav" % fname.strip('.wav'), A, phase[:, start:end], dur)

if __name__ == '__main__':
    split_vocal('.\Datasets\Electrobyte\\audio\\train\\TheFatRat & RIELL - Hiding In The Blue [Chapter 1].wav')
