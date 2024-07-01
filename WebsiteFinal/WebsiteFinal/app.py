from flask import Flask, request, jsonify
import numpy as np
import librosa
from scipy.stats import skew, kurtosis
from scipy.signal import lfilter
import os
import tempfile
from joblib import load

app = Flask(__name__)

# Load the trained Random Forest model
model_filename = './random_forest_model.joblib'
rf_classifier = load(model_filename)

# Ensure an uploads directory exists
uploads_dir = os.path.join(tempfile.gettempdir(), 'uploads')
os.makedirs(uploads_dir, exist_ok=True)


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
        return None
    try:
        audio, sample_rate = librosa.load(audio_path, sr=None)
        if len(audio) < n_fft:
            return None
        
        envelope = np.abs(librosa.onset.onset_strength(y=audio, sr=sample_rate))
        rms_energy = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=n_fft//2)[0]
        silence_ratio = np.mean(rms_energy < 0.01)
        
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate, n_fft=n_fft)
        pitches = pitches[pitches > 0]
        amplitudes = np.max(magnitudes, axis=0)
        amplitudes = amplitudes[amplitudes > 0]
        
        jitter = calculate_jitter(pitches, sample_rate / (pitches + np.finfo(float).eps))
        shimmer = calculate_shimmer(amplitudes)
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20, n_fft=n_fft)
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta_delta = librosa.feature.delta(mfccs, order=2)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_bandwidths = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        formants = get_formants(audio, sample_rate)
        
        spectral_flux = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
        beat_times = librosa.frames_to_time(beats, sr=sample_rate)
        ltas = librosa.amplitude_to_db(np.mean(librosa.stft(audio, n_fft=n_fft), axis=1), ref=np.max)
        
        agg_features = np.hstack((
            mfccs.mean(axis=1), mfccs.std(axis=1), skew(mfccs, axis=1), kurtosis(mfccs, axis=1),
            mfccs_delta.mean(axis=1), mfccs_delta.std(axis=1),
            mfccs_delta_delta.mean(axis=1), mfccs_delta_delta.std(axis=1),
            spectral_centroids.mean(), spectral_centroids.std(),
            spectral_bandwidths.mean(), spectral_bandwidths.std(),
            spectral_contrast.mean(axis=1), spectral_contrast.std(axis=1),
            chroma.mean(axis=1), chroma.std(axis=1),
            zero_crossing_rate.mean(), zero_crossing_rate.std(),
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
        return None

@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <head>
        <title>Upload .wav File</title>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
        <style>
            #progressBarContainer {
                width: 100%;
                background-color: #ddd;
            }
            #progressBar {
                width: 0%;
                height: 30px;
                background-color: #4CAF50;
                text-align: center;
                line-height: 30px;
                color: white;
            }
        </style>
    </head>
    <body>
        <h1>Upload .wav File for Classification</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="button" value="Upload" id="uploadButton">
        </form>
        <div id="progressBarContainer">
            <div id="progressBar">0%</div>
        </div>
        <p id="predictionResult">Prediction will appear here</p>
        <script>
            $(document).ready(function() {
                $("#uploadButton").click(function() {
                    var formData = new FormData();
                    formData.append('file', $('input[type=file]')[0].files[0]);
                    $.ajax({
                        xhr: function() {
                            var xhr = new window.XMLHttpRequest();
                            xhr.upload.addEventListener("progress", function(evt) {
                                if (evt.lengthComputable) {
                                    var percentComplete = evt.loaded / evt.total;
                                    percentComplete = parseInt(percentComplete * 100);
                                    $("#progressBar").width(percentComplete + '%');
                                    $("#progressBar").text(percentComplete + '%');
                                }
                            }, false);
                            return xhr;
                        },
                        url: "/predict",
                        type: "POST",
                        data: formData,
                        contentType: false,
                        processData: false,
                        beforeSend: function() {
                            $("#progressBar").width('0%');
                            $("#progressBar").text('0%');
                        },
                        success: function(response) {
                            $("#predictionResult").text("Predicted class: " + response.class);
                            $("#progressBar").text('Upload Complete');
                        },
                        error: function(err) {
                            $("#predictionResult").text("Failed to get prediction.");
                            $("#progressBar").text('Upload Failed');
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file and file.filename.endswith('.wav'):
        filepath = os.path.join(uploads_dir, file.filename)
        file.save(filepath)
        features = extract_features(filepath)
        os.remove(filepath)  # Clean up the uploaded file
        if features is not None:
            features_reshaped = features.reshape(1, -1)
            prediction = rf_classifier.predict(features_reshaped)
            class_label = 'real' if prediction[0] == 1 else 'fake'
            return jsonify({'class': class_label})
        else:
            return jsonify({'error': 'Failed to extract features or invalid file format.'})
    else:
        return jsonify({'error': 'Please upload a .wav file.'})

if __name__ == '__main__':
    app.run()