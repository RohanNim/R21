import torch
from TTS import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the TTS model
tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)

# Clone a voice and generate audio
tts.tts_to_file(
    text="Hello world!",
    speaker_wav="path/to/your/cloning/audio.wav",
    language="en",
    file_path="output_en.wav"
)

# Generate audio in other languages
tts.tts_to_file(
    text="C'est le clonage de la voix.",
    speaker_wav="path/to/your/cloning/audio.wav",
    language="fr-fr",
    file_path="output_fr.wav"
)

tts.tts_to_file(
    text="Isso é clonagem de voz.",
    speaker_wav="path/to/your/cloning/audio.wav",
    language="pt-br",
    file_path="output_pt.wav"
)
