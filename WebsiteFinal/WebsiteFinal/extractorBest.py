import numpy as np
import os
import librosa
from scipy.stats import skew, kurtosis
from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
def get_formants(audio, sample_rate):
    audio_preemphasized = lfilter([1., -0.63], 1, audio)
    lpc_order = int(2 + sample_rate / 1000)
    lpc_coeffs = librosa.core.lpc(audio_preemphasized, order=lpc_order)
    roots = np.roots(lpc_coeffs)
    roots = [r for r in roots if np.imag(r) >= 0]
    ang_freqs = np.arctan2(np.imag(roots), np.real(roots))
    formants = ang_freqs * (sample_rate / (2 * np.pi))
    formants = sorted(formants)
    return formants[:3] if len(formants) >= 3 else [0, 0, 0]

def calculate_jitter(pitches, periods):
    jitter = np.abs(np.diff(pitches)) / pitches[:-1]
    jitter = jitter[~np.isnan(jitter)]
    return np.mean(jitter) if len(jitter) > 0 else 0

def calculate_shimmer(amplitudes):
    shimmer = np.abs(np.diff(amplitudes)) / amplitudes[:-1]
    shimmer = shimmer[~np.isnan(shimmer)]
    return np.mean(shimmer) if len(shimmer) > 0 else 0

def extract_features(audio_path, n_fft=1024):
    if not os.path.isfile(audio_path):
        print(f"Error: {audio_path} does not exist.")
        return None
    try:
        audio, sample_rate = librosa.load(audio_path, sr=None)
        if len(audio) < n_fft:
            print(f"Audio file {audio_path} is too short for feature extraction. Skipping...")
            return None
        
        # Temporal Features
        envelope = np.abs(librosa.onset.onset_strength(y=audio, sr=sample_rate))
        rms_energy = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=n_fft//2)[0]
        silence_ratio = np.mean(rms_energy < 0.01)
        
        # Pitch and Amplitude Extraction for Jitter and Shimmer
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate, n_fft=n_fft)
        pitches = pitches[pitches > 0]  # Filter out 0 values which represent no pitch found
        amplitudes = np.max(magnitudes, axis=0)
        amplitudes = amplitudes[amplitudes > 0]  # Filter out 0 values
        
        jitter = calculate_jitter(pitches, sample_rate / pitches)
        shimmer = calculate_shimmer(amplitudes)
        
        # Existing Features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20, n_fft=n_fft)
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta_delta = librosa.feature.delta(mfccs, order=2)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_bandwidths = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        formants = get_formants(audio, sample_rate)
        
        # Additional Temporal Features
        spectral_flux = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
        beat_times = librosa.frames_to_time(beats, sr=sample_rate)
        ltas = librosa.amplitude_to_db(np.mean(librosa.stft(audio, n_fft=n_fft), axis=1), ref=np.max)
        
        # Aggregate Features
        agg_features = np.hstack((
            mfccs.mean(axis=1), mfccs.std(axis=1), skew(mfccs, axis=1), kurtosis(mfccs, axis=1),
            mfccs_delta.mean(axis=1), mfccs_delta.std(axis=1),
            mfccs_delta_delta.mean(axis=1), mfccs_delta_delta.std(axis=1),
            spectral_centroids.mean(axis=1), spectral_centroids.std(axis=1),
            spectral_bandwidths.mean(axis=1), spectral_bandwidths.std(axis=1),
            spectral_contrast.mean(axis=1), spectral_contrast.std(axis=1),
            chroma.mean(axis=1), chroma.std(axis=1),
            zero_crossing_rate.mean(axis=1), zero_crossing_rate.std(axis=1),
            formants,
            envelope.mean(), envelope.std(), skew(envelope), kurtosis(envelope),
            rms_energy.mean(), rms_energy.std(),
            silence_ratio,
            jitter, shimmer,
            spectral_flux.mean(), spectral_flux.std(), skew(spectral_flux), kurtosis(spectral_flux),
            [tempo], [len(beat_times)],
            ltas.mean(), ltas.std()
        ))
        
        return agg_features
    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")
        return None

def shuffle_and_split(features, labels, test_size=0.15, val_size=0.15):
    features_temp, features_test, labels_temp, labels_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    val_size_adjusted = val_size / (1 - test_size)
    features_train, features_val, labels_train, labels_val = train_test_split(features_temp, labels_temp, test_size=val_size_adjusted, random_state=42)
    return features_train, features_val, features_test, labels_train, labels_val, labels_test

def process_and_save_dataset(root_dir):
    features_list = []
    labels_list = []
    for dataset_version in ['for-norm', 'for-2sec', 'for-rerec']:
        for data_type in ['training', 'validation', 'testing']:
            for label_type in ['fake', 'real']:
                folder_path = os.path.join(root_dir, dataset_version, data_type, label_type)
                if not os.path.exists(folder_path):
                    print(f"Directory {folder_path} not found.")
                    continue
                print(f"Processing {folder_path}...")
                for filename in tqdm(os.listdir(folder_path), desc=f"Processing files in {folder_path}"):
                    if filename.endswith('.wav'):
                        file_path = os.path.join(folder_path, filename)
                        features = extract_features(file_path)
                        if features is not None:
                            features_list.append(features)
                            labels_list.append(1 if label_type == 'real' else 0)
    
    all_features = np.array(features_list)
    all_labels = np.array(labels_list)
    
    features_train, features_val, features_test, labels_train, labels_val, labels_test = shuffle_and_split(all_features, all_labels)
    
    save_dataset(features_train, labels_train, './npy3/training_features.npy', './npy3/training_labels.npy')
    save_dataset(features_val, labels_val, './npy3/validation_features.npy', './npy3/validation_labels.npy')
    save_dataset(features_test, labels_test, './npy3/testing_features.npy', './npy3/testing_labels.npy')

def save_dataset(features, labels, feature_path, label_path):
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    np.save(feature_path, features)
    np.save(label_path, labels)

root_dir = '.'  # Adjust this path to your dataset directory
process_and_save_dataset(root_dir)
