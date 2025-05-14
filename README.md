
#Emotion-Aware Speech Recognition using Whisper, Librosa, and SVM
  This project combines **speech recognition** and **emotion detection** from video/audio input. It uses [OpenAI Whisper](https://github.com/openai/whisper) for transcribing speech and [Librosa](https://librosa.org/) for extracting audio features to detect emotions using a Support Vector Machine (SVM) classifier.

## Features
* Converts **MP4 video** files into audio.
* Transcribes **spoken language** using Whisper.
* Extracts **audio features** (MFCCs and pitch).
* Detects **emotion** (happy, sad, angry, neutral) from voice.
* Prints **transcription**, **language**, and **emotion**.

#Requirements
Install all dependencies using pip:
pip install openai-whisper librosa scikit-learn moviepy
> Note: FFmpeg is required for Whisper and MoviePy to process audio. Install via:
sudo apt install ffmpeg

## Project Structure
emotion_speech_recognition/
│
├── main.py                # Main Python script (your code)
├── README.md              # This file

## How It Works
### 1. **Load Whisper Model**

model = whisper.load_model("base")

Loads the pre-trained Whisper ASR model.

### 2. Transcribe Audio

result = model.transcribe(audio_path)

Converts speech from audio into text and detects the spoken language.

### 3. Extract Audio Features

mfcc = librosa.feature.mfcc(...)
pitch, _ = librosa.piptrack(...)

Extracts **13 MFCCs** and **pitch** features from audio. These features are essential for emotion classification.

### 4. Train Emotion Classifier

classifier = SVC(kernel='linear')
classifier.fit(X, y)

Trains a **Support Vector Machine** using synthetic (random) audio features for 4 emotions: `happy`, `sad`, `angry`, and `neutral`. In practice, use real labeled data.

### 5. Classify Emotion
emotion = classifier.predict([features])

Predicts emotion from extracted features using the trained model.

### 6. Convert MP4 to WAV

audio_clip = AudioFileClip(mp4_path)
audio_clip.write_audiofile(wav_path)

Converts input MP4 video to WAV audio for processing.

### 7. Complete Workflow

emotion_aware_speech_recognition(mp4_path)

Runs all steps:
1. Convert MP4 → WAV
2. Transcribe speech
3. Extract features
4. Detect emotion
5. Print results

## Example Usage


audio_path = "/path/to/your/video.mp4"
emotion_aware_speech_recognition(audio_path)

### Output:
Transcription: Hello, this is a test.
Language Detected: en
Detected Emotion: happy

##Notes
* The emotion classifier is trained on **random data** in this example. Replace it with real labeled voice emotion datasets (e.g., RAVDESS, CREMA-D).
* Audio quality significantly affects feature extraction and transcription accuracy.
* Whisper may take time depending on audio length and model size.

## Future Improvements
* Integrate a real emotion dataset.
* Add a GUI or Web API (e.g., Flask or Streamlit).
* Fine-tune Whisper for domain-specific speech.
* Visualize emotion predictions over time.

## License
This project is for educational and research use. For commercial use, check the respective licenses of Whisper, Librosa, and MoviePy.
