import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
audio_path = 'ChloeMeowAngry.mp3'
y, sr = librosa.load(audio_path)

# Compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))

# Plot the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max), sr=sr, y_axis='log', x_axis='time')
plt.title('Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

# Save the figure
plt.savefig('spectrogram.png', dpi=300)  # Save as PNG with high DPI
plt.close()  # Close the figure to free memory
