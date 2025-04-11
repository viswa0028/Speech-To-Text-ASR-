import PIL.Image as Image
import librosa
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer

directory1 = './LibriSpeech/dev-clean/84/121123'

# Text Cleaning and Mapping
def clean_text(text):
    try:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return text
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return text.strip().lower()

def process_text_directory(directory):
    transcriptions = {}
    transcription_file = os.path.join(directory, "84-121123.trans.txt")
    if not os.path.exists(transcription_file):
        print(f"No transcription file in {directory}")
        return {}
    with open(transcription_file, 'r') as f:
        for line in f:
            if ' ' in line:
                filename, text = line.split(' ', 1)
                filename = filename.strip()
                text = clean_text(text)
                transcriptions[filename] = text
    audio_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    audio_text_mapping = {}
    for audio_file in audio_files:
        filename_no_ext = os.path.splitext(audio_file)[0]
        if filename_no_ext in transcriptions:
            audio_text_mapping[audio_file] = transcriptions[filename_no_ext]
        else:
            print(f"Warning: No transcription for {audio_file} in {directory}")
    output_file = os.path.join(directory, "cleaned_mapping.txt")
    with open(output_file, 'w') as f:
        for audio, text in audio_text_mapping.items():
            f.write(f"{audio} - {text}\n")
    print(f"Processed text for {directory}")
    return audio_text_mapping

# FLAC to WAV Conversion
from pydub import AudioSegment

def convert_flac_to_wav(input_path, sample_rate=16000):
    try:
        output_path = os.path.splitext(input_path)[0] + ".wav"
        audio = AudioSegment.from_file(input_path, format="flac")
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        audio.export(output_path, format="wav")
        os.remove(input_path)
        print(f"Converted and replaced {input_path} with {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def convert_flac_directory(directory, sample_rate=16000):
    flac_files = [f for f in os.listdir(directory) if f.endswith('.flac')]
    print(f"Found {len(flac_files)} FLAC files in {directory}")
    for flac_file in flac_files:
        input_path = os.path.join(directory, flac_file)
        convert_flac_to_wav(input_path, sample_rate)

# Feature Extraction
def plot_spectrogram(Y, sr, hop_length, y_axis='linear', output_path=None):
    plt.figure(figsize=(2.56, 2.56))
    librosa.display.specshow(Y, hop_length=hop_length, sr=sr, y_axis=y_axis)
    plt.colorbar(format="%+2.0f")
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()

def feature_extraction(directory, audio_length=3.0, sample_rate=16000, n_mels=128, hop_size=512):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    features = []
    for wav_file in wav_files:
        wav_path = os.path.join(directory, wav_file)
        scale, sr = librosa.load(wav_path, sr=sample_rate)
        target_samples = int(audio_length * sr)
        if len(scale) < target_samples:
            scale = np.pad(scale, (0, target_samples - len(scale)), mode='constant')
        else:
            scale = scale[:target_samples]
        mel_spec = librosa.feature.melspectrogram(y=scale, sr=sr, n_mels=n_mels, hop_length=hop_size)
        Y_log_scale = librosa.power_to_db(mel_spec)
        output_filename = os.path.splitext(wav_file)[0] + "_spectrogram.png"
        output_path = os.path.join(directory, output_filename)
        plot_spectrogram(Y_log_scale, sr, hop_length=hop_size, y_axis='log', output_path=output_path)
        print(f"Saved spectrogram for {wav_file} to {output_path}")
        Y_log_scale = torch.from_numpy(Y_log_scale).float().unsqueeze(0)
        Y_log_scale = transform(Y_log_scale)
        features.append((Y_log_scale, wav_file))
    return features

# Model Classes
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.linear = nn.Linear(1024 * 32 * 32, 1000)  # Adjusted after testing

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear(x))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size=300):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 500)
        self.lstm1 = nn.LSTM(500, 1024, batch_first=True)
        self.lstm2 = nn.LSTM(1024, 512, batch_first=True)
        self.linear = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.linear(x)
        return x
tokenizer = TfidfVectorizer()


image_files = [f for f in os.listdir(directory1) if f.endswith('.png')]

transforms1 = transforms.Compose([
    transforms.ToPILImage(),          # If input is tensor or numpy; skip if already PIL
    transforms.Resize((128, 128)),    # Match encoder input size
    transforms.ToTensor()             # Convert to tensor [1, 128, 128]
])

X_train = []
for i in image_files:
    img_path = os.path.join(directory1, i)
    img = Image.open(img_path).convert('L')
    img_transformed = transforms1(img)
    X_train.append(img_transformed)
X_train = torch.stack(X_train)
