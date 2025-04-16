import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pydub import AudioSegment
import gc

torch.cuda.empty_cache()
gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

root_dir = 'Your Path'

def convert_flac_to_wav(directory):
    flac_files = [f for f in os.listdir(directory) if f.endswith('.flac')]
    for flac_file in flac_files:
        wav_file = os.path.splitext(flac_file)[0] + '.wav'
        wav_path = os.path.join(directory, wav_file)

        if os.path.exists(wav_path):
            print(f"Skipping existing WAV: {wav_path}")
            continue
        flac_path = os.path.join(directory, flac_file)
        audio = AudioSegment.from_file(flac_path, format='flac')
        audio.export(wav_path, format='wav')
        print(f"Converted {flac_file} to {wav_file}")

def plot_spectrogram(Y, sr, hop_length, y_axis='linear', output_path=None):
    plt.figure(figsize=(2.56, 2.56))
    librosa.display.specshow(Y, hop_length=hop_length, sr=sr, y_axis=y_axis)
    plt.colorbar(format="%+2.0f")
    if output_path:
        plt.savefig(output_path, dpi=1000)
        plt.close()

def feature_extraction(directory, audio_length=3.0, sample_rate=16000, n_mels=128, hop_size=512):
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    for wav_file in wav_files:
        wav_path = os.path.join(directory, wav_file)
        output_filename = os.path.splitext(wav_file)[0] + "_spectrogram.png"
        output_path = os.path.join(directory, output_filename)
        if os.path.exists(output_path):
            print(f"Skipping existing spectrogram: {output_path}")
            continue
        scale, sr = librosa.load(wav_path, sr=sample_rate)
        target_samples = int(audio_length * sr)
        if len(scale) < target_samples:
            scale = np.pad(scale, (0, target_samples - len(scale)), mode='constant')
        else:
            scale = scale[:target_samples]
        mel_spec = librosa.feature.melspectrogram(y=scale, sr=sr, n_mels=n_mels, hop_length=hop_size)
        Y_log_scale = librosa.power_to_db(mel_spec)
        plot_spectrogram(Y_log_scale, sr, hop_length=hop_size, y_axis='log', output_path=output_path)
        print(f"Saved spectrogram for {wav_file} to {output_path}")

def process_text_directory(directory):
    speaker_id = os.path.basename(os.path.dirname(directory))
    book_id = os.path.basename(directory)
    transcription_file = os.path.join(directory, f"{speaker_id}-{book_id}.trans.txt")
    if not os.path.exists(transcription_file):
        print(f"No transcription file {transcription_file}")
        return {}
    transcriptions = {}
    with open(transcription_file, 'r') as f:
        for line in f:
            if ' ' in line:
                filename, text = line.split(' ', 1)
                filename = filename.strip()
                text = text.strip().lower()
                transcriptions[filename] = text
    audio_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    audio_text_mapping = {}
    for audio_file in audio_files:
        filename_no_ext = os.path.splitext(audio_file)[0]
        if filename_no_ext in transcriptions:
            audio_text_mapping[audio_file] = transcriptions[filename_no_ext]
        else:
            print(f"Warning: No transcription for {audio_file} in {transcription_file}")
    output_file = os.path.join(directory, "cleaned_mapping.txt")
    with open(output_file, 'w') as f:
        for audio, text in audio_text_mapping.items():
            f.write(f"{audio} - {text}\n")
    print(f"Processed text for {directory}")
    return audio_text_mapping

def collect_image_files(root_dir):
    image_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.png'):
                image_files.append(os.path.join(dirpath, f))
    return image_files

image_files = collect_image_files(root_dir)
transforms1 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
print(f"Found {len(image_files)} image files")

def collect_text_mappings(root_dir):
    text_mappings = {}
    for dirpath, _, filenames in os.walk(root_dir):
        if 'cleaned_mapping.txt' in filenames:
            mapping_file = os.path.join(dirpath, 'cleaned_mapping.txt')
            with open(mapping_file, 'r') as f:
                for line in f:
                    if ' - ' in line:
                        filenames, text = line.split(" - ", 1)
                        filenames = filenames.strip()
                        text = text.strip().lower()
                        png_name = filenames.replace('.wav', '_spectrogram.png')
                        rel_png_path = os.path.join(dirpath, png_name)
                        text_mappings[rel_png_path] = text
                    else:
                        print(f"Skipping malformed line in {mapping_file}: {line.strip()}")
    return text_mappings

text_mappings = collect_text_mappings(root_dir)
print(f"Text mappings for {len(text_mappings)} files")

def build_vocab(texts):
    chars = set(''.join(texts))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
    char_to_idx['<PAD>'] = 0
    char_to_idx['<SOS>'] = len(char_to_idx)
    char_to_idx['<EOS>'] = len(char_to_idx)
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

texts = list(text_mappings.values())
char_to_idx, idx_to_char = build_vocab(texts)
vocab_size = len(char_to_idx)

def text_to_indices(text, char_to_idx):
    return [char_to_idx['<SOS>']] + [char_to_idx[c] for c in text if c in char_to_idx] + [char_to_idx['<EOS>']]

y_train_dict = {filename: text_to_indices(text, char_to_idx) for filename, text in text_mappings.items()}
print(f"y_train_dict keys: {len(y_train_dict)}")

class SpeechDataset(Dataset):
    def __init__(self, image_files, y_train_dict, transform):
        self.image_files = image_files
        self.y_train_dict = y_train_dict
        self.transform = transform
        self.valid_pairs = []
        for i, filename in enumerate(self.image_files):
            if filename in y_train_dict:
                self.valid_pairs.append((i, filename))
            else:
                print(f'Filename {filename} not found')
        if len(self.valid_pairs) == 0:
            print("Error: No valid spectrogram-text pairs found")
        else:
            print(f"Created dataset with {len(self.valid_pairs)} valid pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        x_idx, filename = self.valid_pairs[idx]
        img = Image.open(filename).convert('L')
        spectrogram = self.transform(img)
        text_indices = self.y_train_dict[filename]
        return spectrogram, torch.tensor(text_indices, dtype=torch.long)

dataset = SpeechDataset(image_files, y_train_dict, transforms1)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, collate_fn=lambda batch: (
    torch.stack([x for x, _ in batch]),
    torch.nn.utils.rnn.pad_sequence([y for _, y in batch], batch_first=True, padding_value=0)
))

class Encoder(nn.Module):
    def __init__(self, input_size=64):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.linear = nn.Linear(256 * (input_size // 4) * (input_size // 4), 512)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        features = F.relu(self.linear(x))
        hidden = features.unsqueeze(0)
        cell = torch.zeros_like(hidden)
        return features, (hidden, cell)
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size=128, hidden_size=512):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm1 = nn.LSTM(embedding_size + hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, x, encoder_features, hidden=None):
    
      if x.dim() == 1:
          x = x.view(1, 1)
      elif x.dim() > 2:
          x = x.squeeze()
          if x.dim() == 1:
              x = x.view(1, 1)
      batch_size, seq_len = x.size()
      x = self.embedding(x)
      encoder_features = encoder_features.unsqueeze(1).expand(-1, seq_len, -1)
      x = torch.cat([x, encoder_features], dim=2)
      if hidden is None:
          hidden = (torch.zeros(1, batch_size, self.hidden_size).to(x.device),
                    torch.zeros(1, batch_size, self.hidden_size).to(x.device))
      else:
         
          if not isinstance(hidden, tuple) or len(hidden) != 2:
              hidden = (hidden, torch.zeros_like(hidden))
      x, (hidden1, cell1) = self.lstm1(x, hidden)
      x = self.linear(x)
      return x, (hidden1, cell1)
# Training
encoder = Encoder().to(device)
decoder = Decoder(vocab_size=vocab_size).to(device)
loss_function = nn.CrossEntropyLoss(ignore_index=0)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

# Debugging
for i, (filename, indices) in enumerate(list(y_train_dict.items())[:3]):
    original_text = text_mappings[filename]
    print(f"Example {i}:")
    print(f"Original: {original_text}")
    print(f"Tokenized: {indices}")

    
    print("Character-by-character tokenization:")
    for j, char in enumerate(original_text):
        if j < len(indices) - 2:  
            token_idx = indices[j + 1] 
            print(f"  '{char}' -> {token_idx}")

    print(f"SOS token: {indices[0]}")
    print(f"EOS token: {indices[-1]}")
    print("----")
#epochs
num_epochs = 10
accumulation_steps = 4
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0
    batch_count = 0

    for batch_idx, (spectrogram, text) in enumerate(dataloader):
        spectrogram = spectrogram.to(device)
        text = text.to(device)

        if (batch_idx % accumulation_steps) == 0:
            encoder_optimizer.zero_grad(set_to_none=True)
            decoder_optimizer.zero_grad(set_to_none=True)

        encoder_features, encoder_hidden = encoder(spectrogram)
        decoder_input = text[:, :-1]
        decoder_target = text[:, 1:]
        decoder_output, _ = decoder(decoder_input, encoder_features, None)
        loss = loss_function(
            decoder_output.reshape(-1, vocab_size),
            decoder_target.reshape(-1)
        )
        loss = loss / accumulation_steps

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            encoder_optimizer.step()
            decoder_optimizer.step()
            torch.cuda.empty_cache()

        total_loss += loss.item() * accumulation_steps
        batch_count += 1

        del spectrogram, text, encoder_features, encoder_hidden
        del decoder_input, decoder_target, decoder_output, loss

        if (batch_idx + 1) % (10 * accumulation_steps) == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {total_loss/batch_count:.4f}")
            gc.collect()
            torch.cuda.empty_cache()

    avg_loss = total_loss / batch_count
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

torch.save(encoder.state_dict(), '/content/drive/MyDrive/encoder.pth')

torch.save(decoder.state_dict(), '/content/drive/MyDrive/decoder.pth')

# Load models for testing
encoder.load_state_dict(torch.load(os.path.join('/content/drive/MyDrive', 'encoder.pth'), map_location=device))
decoder.load_state_dict(torch.load(os.path.join('/content/drive/MyDrive', 'decoder.pth'), map_location=device))

decoder.eval()

encoder.eval()

input_flac = '/content/drive/MyDrive/dev-clean/174/50561/174-50561-0000.flac'
output_wav = input_flac.replace('.flac', '.wav')
output_png = input_flac.replace('.flac', '_spectrogram.png')

def convert_flac_to_wav(flac_path, wav_path):
    if os.path.exists(wav_path):
        print(f"Skipping existing WAV: {wav_path}")
    else:
        try:
            audio = AudioSegment.from_file(flac_path, format='flac')
            audio.export(wav_path, format='wav')
            print(f"Converted {flac_path} to {wav_path}")
        except Exception as e:
            print(f"Error converting {flac_path}: {e}")
            raise

def generate_spectrogram(wav_path, output_path, audio_length=3.0, sample_rate=16000, n_mels=128, hop_size=512):
    if os.path.exists(output_path):
        print(f"Skipping existing spectrogram: {output_path}")
    else:
        try:
            scale, sr = librosa.load(wav_path, sr=sample_rate)
            target_samples = int(audio_length * sr)
            scale = np.pad(scale, (0, max(0, target_samples - len(scale))))[:target_samples]
            mel_spec = librosa.feature.melspectrogram(y=scale, sr=sr, n_mels=n_mels, hop_length=hop_size)
            Y_log_scale = librosa.power_to_db(mel_spec)
            plot_spectrogram(Y_log_scale, sr, hop_length=hop_size, y_axis='log', output_path=output_path)
            print(f"Saved spectrogram to {output_path}")
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            raise

convert_flac_to_wav(input_flac, output_wav)
generate_spectrogram(output_wav, output_png)

# Load and transform spectrogram
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
img = Image.open(output_png).convert('L')
spectrogram = transform(img).unsqueeze(0).to(device)

print("Special tokens in vocab:")
for key in char_to_idx:
    if '<' in key:
        print(f"Token: '{key}', Index: {char_to_idx[key]}")

sos_token = '<SOS>'
eos_token = '<EOS>'
if sos_token not in char_to_idx or eos_token not in char_to_idx:
    print("Error: <SOS> or <EOS> token missing in vocabulary!")
else:
    max_length = 23
    output = [char_to_idx[sos_token]]
    with torch.no_grad():
        encoder_features, _ = encoder(spectrogram)
        decoder_input = torch.tensor([output[-1]], dtype=torch.long).view(1, 1).to(device)
        hidden = None
        for _ in range(max_length):
            decoder_output, hidden = decoder(decoder_input, encoder_features, hidden)
            _, predicted = decoder_output.max(-1)
            predicted_idx = predicted.item()
            if predicted_idx == char_to_idx[eos_token]:
                break
            output.append(predicted_idx)
            decoder_input = predicted.view(1, 1)
        text = ''.join(idx_to_char[idx] for idx in output[1:])
        print(f"Predicted transcription: {text}")
