# 🧠 Audio-to-Text ASR System | Mel-Spectrogram + Seq2Seq 🗣️➡️📝

Welcome to my weekend (and late-night ☕) experiment where I taught a neural network to **"see" sound and speak it back as text**!  
This project transforms audio into **mel-spectrograms**, feeds them into a **sequence-to-sequence model**, and outputs **natural language transcriptions**.

---

## 🎯 What This Project Does

- Converts `.wav` audio files into colorful **log-mel spectrograms**
- Uses a **CNN-based encoder** to extract time-frequency patterns
- Passes encoded features through a **sequence decoder (LSTM/Transformer)**
- Outputs **textual transcriptions** of the spoken content

---

## 🚀 How It Works
🎧 Audio (.wav) 
    ↓
📸 Log-Mel Spectrogram (Image-like representation)
    ↓
🧠 Seq2Seq Model (Encoder-Decoder with attention)
    ↓
📝 Predicted Text (Transcript)


---

## 🛠️ Tech Stack

- **Python** & **PyTorch** (or TensorFlow, depending on your version)
- **Librosa** for audio processing
- **Matplotlib / PIL** for spectrogram visualization
- **Numpy** for Data Manipulation
- **TorchVision** for Image Processing
- **CNN + RNN/Transformer** architecture
- Optional: **CTC Loss / Attention** mechanism for alignment

---
## 🤓 Future Improvements
- Switch to Wav2Vec2 or Whisper for better performance

- Add beam search decoding

- Build a simple demo web app using Streamlit or Gradio
## Sample Inputs
