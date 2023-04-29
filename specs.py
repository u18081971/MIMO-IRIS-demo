import glob
import os.path
import librosa
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

plt.rcParams["font.size"] = 10
plt.rcParams["figure.autolayout"] = True


root_folder="./examples/reverb"
#flac_files = glob.glob(os.path.join(root_folder, "**/*.flac"))
#for f in flac_files:
 #   audio, fs = torchaudio.load(f)
  #  torchaudio.save(os.path.join(Path(f).parent, Path(f).stem + ".wav"), audio, fs)


wav_files = glob.glob(os.path.join(root_folder, "**/*.wav"), recursive=True)
for w in wav_files:
    c_audio, fs = torchaudio.load(w)
    if c_audio.shape[0] > 1:
        c_audio = c_audio[0].unsqueeze(0)
    c_audio = c_audio / torch.amax(torch.abs(c_audio))
    #torchaudio.save(w, c_audio, fs)

def get_spectrogram(w, target_len=None):
    audio, fs = torchaudio.load(w)
    if target_len is not None:
        audio = torch.nn.functional.pad(audio, (0, target_len - audio.shape[-1]), mode="constant")
    audio = audio[0].numpy()
    stft_repr = librosa.stft(audio, n_fft=512, hop_length=128,
                             win_length=400, window='hann')

    return stft_repr, audio


noisy_f = [f for f in wav_files if Path(f).stem.startswith("mix")][0]
_, noisy_audio = get_spectrogram(noisy_f)

for w in wav_files:
     #if w == "/media/samco/Data/multiIRIS-demo/examples/REVERB/Clean/c30_SimData_et_for_8ch_far_room1_c30c020a.wav":

      #  stft_repr, audio = get_spectrogram(w, target_len=noisy_audio.shape[-1])
    #else:
    stft_repr, audio = get_spectrogram(w, target_len=None)

    time = len(audio) / fs
    maxval = np.max(np.abs(stft_repr))
    #plt.figure(figsize=(40, 30), frameon=False)
    fig, ax = plt.subplots()
    f = ax.imshow(20 * np.log10(np.abs(stft_repr / maxval + 1e-15)), vmin=-70, vmax=-10, extent=[0, time, int(fs/2000), 0], aspect=0.30)
    plt.ylabel('Frequency [kHz]')
    plt.xlabel('Time [s]')
    im_ratio = stft_repr.shape[0] / stft_repr.shape[1]
    cbar = plt.colorbar(f, fraction=0.026*0.6, pad=0.04)
    cbar.set_label('Intensity [dB]')


    #plt.text(0.2, 5.7, '{}', bbox={'facecolor': 'white', 'pad': 8})
    plt.ylim(0, int(fs/2000))
    plt.tight_layout()
    plt.savefig(os.path.join(Path(w).parent, Path(w).stem) + ".png", bbox_inches='tight')
