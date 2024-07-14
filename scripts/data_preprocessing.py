import os
import librosa
import numpy as np

def load_data(dataset_path, augment=False):
    labels = []
    features = []

    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            y, sr = librosa.load(file_path, duration=3.0)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            features.append(mfcc_scaled)
            label = os.path.basename(subdir)
            labels.append(label)

            if augment:
                # Time Stretching
                y_stretch = librosa.effects.time_stretch(y, rate=0.8)
                mfcc_stretch = librosa.feature.mfcc(y=y_stretch, sr=sr, n_mfcc=40)
                mfcc_stretch_scaled = np.mean(mfcc_stretch.T, axis=0)
                features.append(mfcc_stretch_scaled)
                labels.append(label)

                y_stretch = librosa.effects.time_stretch(y, rate=1.2)
                mfcc_stretch = librosa.feature.mfcc(y=y_stretch, sr=sr, n_mfcc=40)
                mfcc_stretch_scaled = np.mean(mfcc_stretch.T, axis=0)
                features.append(mfcc_stretch_scaled)
                labels.append(label)

                # Pitch Shifting
                y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
                mfcc_shift = librosa.feature.mfcc(y=y_shift, sr=sr, n_mfcc=40)
                mfcc_shift_scaled = np.mean(mfcc_shift.T, axis=0)
                features.append(mfcc_shift_scaled)
                labels.append(label)

                y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
                mfcc_shift = librosa.feature.mfcc(y=y_shift, sr=sr, n_mfcc=40)
                mfcc_shift_scaled = np.mean(mfcc_shift.T, axis=0)
                features.append(mfcc_shift_scaled)
                labels.append(label)

                # Adding Noise
                noise = np.random.randn(len(y))
                y_noise = y + 0.005 * noise
                mfcc_noise = librosa.feature.mfcc(y=y_noise, sr=sr, n_mfcc=40)
                mfcc_noise_scaled = np.mean(mfcc_noise.T, axis=0)
                features.append(mfcc_noise_scaled)
                labels.append(label)

                # Shifting Time
                y_roll = np.roll(y, 1600)
                mfcc_roll = librosa.feature.mfcc(y=y_roll, sr=sr, n_mfcc=40)
                mfcc_roll_scaled = np.mean(mfcc_roll.T, axis=0)
                features.append(mfcc_roll_scaled)
                labels.append(label)

    return np.array(features), np.array(labels)
