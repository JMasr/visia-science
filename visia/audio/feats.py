# import librosa
# import numpy as np
# import opensmile
# import torch
# import torchaudio
# from spafe.features.bfcc import bfcc
# from spafe.features.cqcc import cqcc
# from spafe.features.gfcc import gfcc
# from spafe.features.lfcc import lfcc
# from spafe.features.lpc import lpc
# from spafe.features.lpc import lpcc
# from spafe.features.mfcc import mfcc, imfcc
# from spafe.features.msrcc import msrcc
# from spafe.features.ngcc import ngcc
# from spafe.features.pncc import pncc
# from spafe.features.psrcc import psrcc
# from spafe.features.rplp import plp, rplp
#
#
# class FeatureExtractor:
#     """
#     Extracts the features from the audio file.
#
#     :param arguments: dictionary with the arguments
#     :type arguments: dict
#     :return: FeatureExtractor object.
#     :rtype: FeatureExtractor
#
#     """
#
#     def __init__(self, arguments: dict):
#
#         self.args = arguments
#         self.audio_path = None
#         self.resampling_rate = self.args['resampling_rate']
#         assert (arguments['feature_type'] in ['MFCC', 'MelSpec', 'logMelSpec',
#                                               'ComParE_2016_energy', 'ComParE_2016_voicing',
#                                               'ComParE_2016_spectral', 'ComParE_2016_basic_spectral',
#                                               'ComParE_2016_mfcc', 'ComParE_2016_rasta', 'ComParE_2016_llds',
#                                               'Spafe_mfcc', 'Spafe_imfcc', 'Spafe_cqcc', 'Spafe_gfcc', 'Spafe_lfcc',
#                                               'Spafe_lpc', 'Spafe_lpcc', 'Spafe_msrcc', 'Spafe_ngcc', 'Spafe_pncc',
#                                               'Spafe_psrcc', 'Spafe_plp', 'Spafe_rplp']), \
#             'Expected the feature_type to be MFCC / MelSpec / logMelSpec / ComParE_2016'
#
#         nfft = int(float(self.args.get('window_size', 0) * 1e-3 * self.resampling_rate))
#         hop_length = int(float(self.args.get('hop_length', 0)) * 1e-3 * self.resampling_rate)
#
#         if self.args['feature_type'] == 'MFCC':
#             self.feature_transform = torchaudio.transforms.MFCC(sample_rate=self.resampling_rate,
#                                                                 n_mfcc=int(self.args['n_mfcc']),
#                                                                 melkwargs={'n_fft': nfft,
#                                                                            'n_mels': int(self.args['n_mels']),
#                                                                            'f_max': int(self.args['f_max']),
#                                                                            'hop_length': hop_length})
#         elif self.args['feature_type'] in ['MelSpec', 'logMelSpec']:
#             self.feature_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.resampling_rate,
#                                                                           n_fft=nfft,
#                                                                           n_mels=int(self.args['n_mels']),
#                                                                           f_max=int(self.args['f_max']),
#                                                                           hop_length=hop_length)
#         elif 'ComParE_2016' in self.args['feature_type']:
#             self.feature_transform = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
#                                                      feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
#                                                      sampling_rate=self.resampling_rate)
#         elif 'Spafe_' in self.args['feature_type']:
#             spafe_feature_transformers = {'Spafe_mfcc': mfcc,
#                                           'Spafe_imfcc': imfcc,
#                                           'Spafe_bfcc': bfcc,
#                                           'Spafe_cqcc': cqcc,
#                                           'Spafe_gfcc': gfcc,
#                                           'Spafe_lfcc': lfcc,
#                                           'Spafe_lpc': lpc,
#                                           'Spafe_lpcc': lpcc,
#                                           'Spafe_msrcc': msrcc,
#                                           'Spafe_ngcc': ngcc,
#                                           'Spafe_pncc': pncc,
#                                           'Spafe_psrcc': psrcc,
#                                           'Spafe_plp': plp,
#                                           'Spafe_rplp': rplp}
#             self.feature_transform = spafe_feature_transformers[self.args['feature_type']]
#         else:
#             raise ValueError('Feature type not implemented')
#
#     def _read_audio(self, audio_file_path):
#         """
#          The code above implements SAD using the librosa.effects.split() function with a threshold of top_db=30, which
#          separates audio regions where the amplitude is lower than the threshold.
#          The pre-emphasis filter is applied using the librosa.effects.preemphasis() function with a coefficient of 0.97.
#          This filter emphasizes the high-frequency components of the audio signal,
#          which can improve the quality of the speech signal.
#          Finally, the code normalizes the audio signal to have maximum amplitude of 1
#          :param audio_file_path: audio file path
#          :return: audio signal and sampling rate
#          """
#         # load the audio file
#         s, sr = librosa.load(audio_file_path, mono=True)
#         # resample
#         if (self.resampling_rate is not None) or (sr < self.resampling_rate):
#             s = librosa.resample(y=s, orig_sr=sr, target_sr=self.resampling_rate)
#             sr = self.resampling_rate
#         # apply speech activity detection
#         speech_indices = librosa.effects.split(s, top_db=30)
#         s = np.concatenate([s[start:end] for start, end in speech_indices])
#         # apply a pre-emphasis filter
#         s = librosa.effects.preemphasis(s, coef=0.97)
#         # normalize
#         s /= np.max(np.abs(s))
#         return torch.from_numpy(s), sr
#
#     @staticmethod
#     def compute_sad(sig, fs, threshold=0.0001, sad_start_end_sil_length=100, sad_margin_length=50):
#         """
#         Compute threshold based sound activity detection.
#
#         :param sig: audio signal.
#         :type sig: np.array
#         :param fs: sampling rate.
#         :type fs: int
#         :param threshold: threshold for SAD.
#         :type threshold: float
#         :param sad_start_end_sil_length: length of leading/trailing silence.
#         :type sad_start_end_sil_length: int
#         :param sad_margin_length: length of margin around active samples.
#         :type sad_margin_length: int
#         :return: SAD vector.
#         :rtype: np.array
#         """
#         # Leading/Trailing margin
#         sad_start_end_sil_length = int(sad_start_end_sil_length * 1e-3 * fs)
#         # Margin around active samples
#         sad_margin_length = int(sad_margin_length * 1e-3 * fs)
#
#         sample_activity = np.zeros(sig.shape)
#         sample_activity[np.power(sig, 2) > threshold] = 1
#         sad = np.zeros(sig.shape)
#         for i in range(sample_activity.shape[1]):
#             if sample_activity[0, i] == 1:
#                 sad[0, i - sad_margin_length:i + sad_margin_length] = 1
#         sad[0, 0:sad_start_end_sil_length] = 0
#         sad[0, -sad_start_end_sil_length:] = 0
#         return sad
#
#     def _do_feature_extraction(self, s, sr):
#         """
#         Feature preparation.
#
#         Steps:
#         1. Apply feature extraction to waveform.
#         2. Convert amplitude to dB if required.
#         3. Append delta and delta-delta features.
#
#         :param s: audio signal.
#         :type s: np.array
#         :param sr: sampling rate.
#         :type sr: int
#         :return: features.
#         :rtype: np.array
#         """
#         feature_matrix = None
#
#         if self.args['feature_type'] == 'MelSpec':
#             feature_matrix = self.feature_transform(s)
#
#         if self.args['feature_type'] == 'logMelSpec':
#             feature_matrix = self.feature_transform(s)
#             feature_matrix = torchaudio.functional.amplitude_to_DB(feature_matrix,
#                                                                    multiplier=10,
#                                                                    amin=1e-10,
#                                                                    db_multiplier=0)
#
#         if self.args['feature_type'] == 'MFCC':
#             feature_matrix = self.feature_transform(s)
#
#         if 'ComParE_2016' in self.args['feature_type']:
#             #
#             s = s[None, :]
#             feature_matrix = self.feature_transform.process_signal(s, sr)
#
#             # feature subsets
#             feature_subset = {}
#             if self.args['feature_type'] == 'ComParE_2016_voicing':
#                 feature_subset['subset'] = ['F0final_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma',
#                                             'jitterDDP_sma', 'shimmerLocal_sma', 'logHNR_sma']
#
#             if self.args['feature_type'] == 'ComParE_2016_energy':
#                 feature_subset['subset'] = ['audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma',
#                                             'pcm_RMSenergy_sma', 'pcm_zcr_sma']
#
#             if self.args['feature_type'] == 'ComParE_2016_spectral':
#                 feature_subset['subset'] = ['audSpec_Rfilt_sma[0]', 'audSpec_Rfilt_sma[1]', 'audSpec_Rfilt_sma[2]',
#                                             'audSpec_Rfilt_sma[3]', 'audSpec_Rfilt_sma[4]', 'audSpec_Rfilt_sma[5]',
#                                             'audSpec_Rfilt_sma[6]', 'audSpec_Rfilt_sma[7]', 'audSpec_Rfilt_sma[8]',
#                                             'audSpec_Rfilt_sma[9]', 'audSpec_Rfilt_sma[10]', 'audSpec_Rfilt_sma[11]',
#                                             'audSpec_Rfilt_sma[12]', 'audSpec_Rfilt_sma[13]', 'audSpec_Rfilt_sma[14]',
#                                             'audSpec_Rfilt_sma[15]', 'audSpec_Rfilt_sma[16]', 'audSpec_Rfilt_sma[17]',
#                                             'audSpec_Rfilt_sma[18]', 'audSpec_Rfilt_sma[19]', 'audSpec_Rfilt_sma[20]',
#                                             'audSpec_Rfilt_sma[21]', 'audSpec_Rfilt_sma[22]', 'audSpec_Rfilt_sma[23]',
#                                             'audSpec_Rfilt_sma[24]', 'audSpec_Rfilt_sma[25]',
#                                             'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
#                                             'pcm_fftMag_spectralRollOff25.0_sma', 'pcm_fftMag_spectralRollOff50.0_sma',
#                                             'pcm_fftMag_spectralRollOff75.0_sma', 'pcm_fftMag_spectralRollOff90.0_sma',
#                                             'pcm_fftMag_spectralFlux_sma',
#                                             'pcm_fftMag_spectralCentroid_sma',
#                                             'pcm_fftMag_spectralEntropy_sma',
#                                             'pcm_fftMag_spectralVariance_sma',
#                                             'pcm_fftMag_spectralSkewness_sma',
#                                             'pcm_fftMag_spectralKurtosis_sma',
#                                             'pcm_fftMag_spectralSlope_sma',
#                                             'pcm_fftMag_psySharpness_sma',
#                                             'pcm_fftMag_spectralHarmonicity_sma',
#                                             'mfcc_sma[1]', 'mfcc_sma[2]', 'mfcc_sma[3]', 'mfcc_sma[4]', 'mfcc_sma[5]',
#                                             'mfcc_sma[6]', 'mfcc_sma[7]', 'mfcc_sma[8]', 'mfcc_sma[9]', 'mfcc_sma[10]',
#                                             'mfcc_sma[11]', 'mfcc_sma[12]', 'mfcc_sma[13]', 'mfcc_sma[14]']
#
#             if self.args['feature_type'] == 'ComParE_2016_mfcc':
#                 feature_subset['subset'] = ['mfcc_sma[1]', 'mfcc_sma[2]', 'mfcc_sma[3]', 'mfcc_sma[4]', 'mfcc_sma[5]',
#                                             'mfcc_sma[6]', 'mfcc_sma[7]', 'mfcc_sma[8]',
#                                             'mfcc_sma[9]', 'mfcc_sma[10]', 'mfcc_sma[11]', 'mfcc_sma[12]',
#                                             'mfcc_sma[13]', 'mfcc_sma[14]']
#
#             if self.args['feature_type'] == 'ComParE_2016_rasta':
#                 feature_subset['subset'] = ['audSpec_Rfilt_sma[0]', 'audSpec_Rfilt_sma[1]', 'audSpec_Rfilt_sma[2]',
#                                             'audSpec_Rfilt_sma[3]',
#                                             'audSpec_Rfilt_sma[4]', 'audSpec_Rfilt_sma[5]', 'audSpec_Rfilt_sma[6]',
#                                             'audSpec_Rfilt_sma[7]', 'audSpec_Rfilt_sma[8]', 'audSpec_Rfilt_sma[9]',
#                                             'audSpec_Rfilt_sma[10]', 'audSpec_Rfilt_sma[11]', 'audSpec_Rfilt_sma[12]',
#                                             'audSpec_Rfilt_sma[13]',
#                                             'audSpec_Rfilt_sma[14]', 'audSpec_Rfilt_sma[15]', 'audSpec_Rfilt_sma[16]',
#                                             'audSpec_Rfilt_sma[17]',
#                                             'audSpec_Rfilt_sma[18]', 'audSpec_Rfilt_sma[19]', 'audSpec_Rfilt_sma[20]',
#                                             'audSpec_Rfilt_sma[21]',
#                                             'audSpec_Rfilt_sma[22]', 'audSpec_Rfilt_sma[23]', 'audSpec_Rfilt_sma[24]',
#                                             'audSpec_Rfilt_sma[25]']
#
#             if self.args['feature_type'] == 'ComParE_2016_basic_spectral':
#                 feature_subset['subset'] = ['pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
#                                             'pcm_fftMag_spectralRollOff25.0_sma',
#                                             'pcm_fftMag_spectralRollOff50.0_sma',
#                                             'pcm_fftMag_spectralRollOff75.0_sma',
#                                             'pcm_fftMag_spectralRollOff90.0_sma',
#                                             'pcm_fftMag_spectralFlux_sma',
#                                             'pcm_fftMag_spectralCentroid_sma',
#                                             'pcm_fftMag_spectralEntropy_sma',
#                                             'pcm_fftMag_spectralVariance_sma',
#                                             'pcm_fftMag_spectralSkewness_sma',
#                                             'pcm_fftMag_spectralKurtosis_sma',
#                                             'pcm_fftMag_spectralSlope_sma',
#                                             'pcm_fftMag_psySharpness_sma',
#                                             'pcm_fftMag_spectralHarmonicity_sma']
#
#             if self.args['feature_type'] == 'ComParE_2016_llds':
#                 feature_subset['subset'] = list(feature_matrix.columns)
#
#             feature_matrix = feature_matrix[feature_subset['subset']].to_numpy()
#             feature_matrix = np.nan_to_num(feature_matrix)
#             feature_matrix = torch.from_numpy(feature_matrix).T
#
#         if 'Spafe_' in self.args['feature_type']:
#             # Spafe feature selected
#             nfft = int(float(self.args.get('window_size', 0) * 1e-3 * self.resampling_rate))
#
#             if self.args['feature_type'] in ['Spafe_mfcc', 'Spafe_imfcc', 'Spafe_gfcc', 'Spafe_lfcc', 'Spafe_msrcc',
#                                              'Spafe_ngcc', 'Spafe_psrcc']:
#                 feature_matrix = self.feature_transform(s, sr,
#                                                         num_ceps=int(self.args.get('n_mfcc')),
#                                                         low_freq=int(self.args.get('f_min')),
#                                                         high_freq=int(sr // 2),
#                                                         nfilts=int(self.args.get('n_mels')),
#                                                         nfft=nfft,
#                                                         use_energy=self.args.get('use_energy') == 'True')
#             elif self.args['feature_type'] in ['Spafe_pncc']:
#                 feature_matrix = self.feature_transform(s, sr, num_ceps=int(self.args.get('n_mfcc')),
#                                                         low_freq=int(self.args.get('f_min')),
#                                                         high_freq=int(sr // 2),
#                                                         nfilts=int(self.args.get('n_mels')),
#                                                         nfft=nfft)
#
#             elif self.args['feature_type'] in ['Spafe_cqcc']:
#                 feature_matrix = self.feature_transform(s, sr,
#                                                         num_ceps=int(self.args.get('n_mfcc')),
#                                                         low_freq=int(self.args.get('f_min')),
#                                                         high_freq=int(sr // 2),
#                                                         nfft=nfft)
#             elif self.args['feature_type'] in ['Spafe_lpc', 'Spafe_lpcc', ]:
#                 feature_matrix = self.feature_transform(s, sr, order=int(self.args.get('plp_order')))
#                 if isinstance(feature_matrix, tuple):
#                     feature_matrix = feature_matrix[0]
#
#             elif self.args['feature_type'] in ['Spafe_plp', 'Spafe_rplp']:
#                 feature_matrix = self.feature_transform(s, sr,
#                                                         order=int(self.args.get('plp_order')),
#                                                         conversion_approach=self.args.get('conversion_approach'),
#                                                         low_freq=int(self.args.get('f_min')),
#                                                         high_freq=int(sr // 2),
#                                                         normalize=self.args.get('normalize'),
#                                                         nfilts=int(self.args.get('n_mels')),
#                                                         nfft=nfft)
#             feature_matrix = np.nan_to_num(feature_matrix)
#             feature_matrix = torch.from_numpy(feature_matrix).T
#
#         if self.args.get('compute_deltas', False):
#             feature_matrix_delta = torchaudio.functional.compute_deltas(feature_matrix)
#             feature_matrix = torch.cat((feature_matrix, feature_matrix_delta), dim=0)
#
#             if self.args.get('compute_deltas_deltas', False):
#                 feature_matrix_delta_delta = torchaudio.functional.compute_deltas(feature_matrix_delta)
#                 feature_matrix = torch.cat((feature_matrix, feature_matrix_delta_delta), dim=0)
#
#         if self.args.get('apply_mean_norm', False):
#             feature_matrix = feature_matrix - torch.mean(feature_matrix, dim=0)
#
#         if self.args.get('apply_var_norm', False):
#             feature_matrix = feature_matrix / torch.std(feature_matrix, dim=0)
#
#         # own feature selection
#         if self.args.get('extra_features', False) and 'ComParE_2016' not in self.args['feature_type']:
#             s = s[None, :]
#             # Config OpenSMILE
#             feature_subset = {'subset': [
#                 # Voicing
#                 'F0final_sma', 'voicingFinalUnclipped_sma',
#                 'jitterLocal_sma', 'jitterDDP_sma',
#                 'shimmerLocal_sma',
#                 'logHNR_sma',
#                 # Energy
#                 'audspec_lengthL1norm_sma',
#                 'audspecRasta_lengthL1norm_sma',
#                 'pcm_RMSenergy_sma',
#                 'pcm_zcr_sma',
#                 # Spectral
#                 'pcm_fftMag_fband250-650_sma',
#                 'pcm_fftMag_fband1000-4000_sma',
#                 'pcm_fftMag_spectralRollOff25.0_sma',
#                 'pcm_fftMag_spectralRollOff50.0_sma',
#                 'pcm_fftMag_spectralRollOff75.0_sma',
#                 'pcm_fftMag_spectralRollOff90.0_sma',
#                 'pcm_fftMag_spectralFlux_sma',
#                 'pcm_fftMag_spectralCentroid_sma',
#                 'pcm_fftMag_spectralEntropy_sma',
#                 'pcm_fftMag_spectralVariance_sma',
#                 'pcm_fftMag_spectralSkewness_sma',
#                 'pcm_fftMag_spectralKurtosis_sma',
#                 'pcm_fftMag_spectralSlope_sma',
#                 'pcm_fftMag_psySharpness_sma',
#                 'pcm_fftMag_spectralHarmonicity_sma'
#             ]}
#             extra_transform = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
#                                               feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
#                                               sampling_rate=self.resampling_rate)
#             # Extract features
#             feature_matrix_extra = extra_transform.process_signal(s, sr)
#             feature_matrix_extra = feature_matrix_extra[feature_subset['subset']].to_numpy()
#             feature_matrix_extra = np.nan_to_num(feature_matrix_extra)
#             feature_matrix_extra = torch.from_numpy(feature_matrix_extra).T
#             # Concatenate the features
#             common_shape = min(feature_matrix.shape[1], feature_matrix_extra.shape[1])
#             feature_matrix = torch.cat((feature_matrix[:, :common_shape],
#                                         feature_matrix_extra[:, :common_shape]),
#                                        dim=0)
#
#         return feature_matrix.T
#
#     def extract(self, filepath):
#         """
#         Extracts the features from the audio file.
#
#         :param filepath: path to the audio file
#         :type filepath: str
#         :return: features.
#         :rtype: np.array
#         """
#         if not isinstance(filepath, str):
#             return np.NAN
#         else:
#             self.audio_path = filepath
#             s, fs = self._read_audio(filepath)
#             return self._do_feature_extraction(s, fs)
