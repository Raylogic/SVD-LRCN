import numpy
import librosa
from librosa import util
from librosa.feature import mfcc, chroma_stft, zero_crossing_rate, rms, spectral_centroid, spectral_rolloff, poly_features, spectral_contrast, spectral_bandwidth, spectral_flatness, delta
from Extraction.rasta_plp import rastaplp


class AudioFeat:
    # Constructor
    def __init__(self, win_size_n=1024, hop_size_n=512):
        self.nfft = win_size_n
        self.hop = hop_size_n
        self.n_mfcc = 80

    # Extract the audio features
    def get_audio_features(self, audio_data, sample_rate):

        # Extract the features with the librosa library
        feat_mfcc = mfcc(audio_data, sample_rate, n_mfcc=self.n_mfcc, n_fft=self.nfft, hop_length=self.hop).T
        feat_MFCC_delta = delta(feat_mfcc)
        feat_MFCC_delta_delta = delta(feat_MFCC_delta)
        feat_chroma = chroma_stft(audio_data, sample_rate, n_fft=self.nfft, hop_length=self.hop).T
        feat_ZCR = zero_crossing_rate(audio_data, frame_length=self.nfft, hop_length=self.hop).T
        feat_RMSE = rms(audio_data, frame_length=self.nfft, hop_length=self.hop).T
        feat_Centroid = spectral_centroid(audio_data, n_fft=self.nfft, hop_length=self.hop).T
        feat_Rolloff = spectral_rolloff(audio_data, n_fft=self.nfft, hop_length=self.hop).T
        feat_Poly = poly_features(audio_data, sample_rate, n_fft=self.nfft, hop_length=self.hop).T
        feat_Contrast = spectral_contrast(audio_data, sample_rate, n_fft=self.nfft, hop_length=self.hop).T
        feat_Bandwidth = spectral_bandwidth(audio_data, sample_rate, n_fft=self.nfft, hop_length=self.hop).T
        feat_Flatness = spectral_flatness(audio_data, n_fft=self.nfft, hop_length=self.hop).T
        feat_lpc = self.get_lpcc(audio_data, win_time=self.nfft, hop_time=self.hop, lpc_order=12)

        audio_data = numpy.pad(audio_data, int(self.nfft // 2), mode='reflect')
        feat_plp = rastaplp(audio_data, sample_rate, win_time=self.nfft / sample_rate, hop_time=self.hop / sample_rate, dorasta=True, modelorder=8).T
        min_cut_frames = min(feat_Bandwidth.shape[0], feat_plp.shape[0])

        # Fuse the features into a vector
        combineFeat = numpy.hstack((feat_Bandwidth[:min_cut_frames, :], 
                                    feat_chroma[:min_cut_frames, :],
                                    feat_Contrast[:min_cut_frames, :], 
                                    feat_Flatness[:min_cut_frames, :],
                                    feat_Centroid[:min_cut_frames, :], 
                                    feat_lpc[:min_cut_frames, :],
                                    feat_mfcc[:min_cut_frames, :], 
                                    feat_MFCC_delta[:min_cut_frames, :],
                                    feat_MFCC_delta_delta[:min_cut_frames, :], 
                                    feat_Poly[:min_cut_frames, :],
                                    feat_RMSE[:min_cut_frames, :], 
                                    feat_plp[:min_cut_frames, :],
                                    feat_Rolloff[:min_cut_frames, :], 
                                    feat_ZCR[:min_cut_frames, :]))

        # Return the feature vector in Float32 mode
        self.total_features = combineFeat.astype('float32')
        return combineFeat.astype('float32')

    # Extract the audio LPCC
    def get_lpcc(self, audio_data, win_time, hop_time, lpc_order):
        audio_data = numpy.pad(audio_data, int(win_time // 2), mode='reflect')
        y_frames = util.frame(audio_data, frame_length=win_time, hop_length=hop_time).T
        lpccs = []
        for frame in y_frames:
            lpccs.append(librosa.lpc(frame, lpc_order)[1:])
        res = numpy.vstack(lpccs)
        return res

    # Assign the features values to a part of the final input vector
    def specify_feature_list_index(self, total_features, feature_list):
        feature_dict = {

            'feat_plp': [277, 278, 279, 280, 281, 282, 283, 284, 285],
            'feat_chroma': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'feat_lpc': [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            'feat_mfcc': [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                          57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                          80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
                          103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
            'feat_MFCC_delta': [114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
                                131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
                                148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
                                165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
                                182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193],
            'feat_MFCC_delta_delta': [194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                                      210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225,
                                      226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
                                      242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257,
                                      258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273],
            'feat_spec_all': [0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 274, 275, 276, 286, 287],
            'feat_Bandwidth': [0],
            'feat_Contrast': [13, 14, 15, 16, 17, 18, 19],
            'feat_Flatness': [20],
            'feat_Centroid': [21],
            'feat_Poly': [274, 275],
            'feat_RMSE': [276],
            'feat_Rolloff': [286],
            'feat_ZCR': [287]
        }
        specify_features = []
        for item in feature_list:
            specify_features.extend(feature_dict[item])
        final_features = total_features[:, specify_features]
        return final_features


def wav_2_feats(wav_path, feature_list):

    # Read the song file
    # signal, samplerate = soundfile.read(wav_path)
    # signal, samplerate = wavfile.read(wav_path)
    signal, samplerate = librosa.load(wav_path)

    # Create an AudioFeat instance
    audio_feater = AudioFeat(win_size_n=2048, hop_size_n=512)

    # Extract the song features
    fuse_features = audio_feater.get_audio_features(signal, samplerate)

    # Fuse the features into an input vector
    final_features = audio_feater.specify_feature_list_index(fuse_features, feature_list)

    # Return the input vector
    return final_features

if __name__ == '__main__':
    final_features = wav_2_feats('./Datasets/Electrobyte/audio/test/Becko - Mindflayer.mp3', ['feat_mfcc'])

