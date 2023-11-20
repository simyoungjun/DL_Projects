from .tools import *

_stft = stft.TacotronSTFT(
    args.filter_length, args.hop_length, args.win_length,
    args.n_mels, args.sr, args.mel_fmin, args.mel_fmax)

def get_hpss(filename, trim_silence=False, frame_length=1024, hop_length=256, top_db=10):
    audio, sampling_rate = load_wav_to_numpy(filename)

    if sampling_rate != _stft.sampling_rate:
        raise ValueError("{} SR doesn't match target SR {}".format(
            sampling_rate, _stft.sampling_rate))
        
    # harmonic_audio, percussive_audio = librosa.effects.hpss(audio_norm)
    
    audio_norm = audio / args.max_wav_value
    if trim_silence:   #harmonic, percussive, mel 을 trim 따로 해도 되는지?? 세개 동시에 해야할 것 같음.
        audio_norm = audio_norm[200:-200]
        audio_norm, idx = librosa.effects.trim(audio_norm, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
        
    harmonic_audio, percussive_audio = librosa.effects.hpss(audio_norm) 
    
    
    harmonic_audio = torch.FloatTensor(harmonic_audio)
    harmonic_audio = harmonic_audio.unsqueeze(0)
    harmonic_audio = torch.autograd.Variable(harmonic_audio, requires_grad=False)
    percussive_audio = torch.FloatTensor(percussive_audio)
    percussive_audio = percussive_audio.unsqueeze(0)
    percussive_audio = torch.autograd.Variable(percussive_audio, requires_grad=False)
    
    
    harmonic_melspec, harmonic_energy  =   _stft.mel_spectrogram(harmonic_audio)
    percussive_melspec, percussive_energy  =   _stft.mel_spectrogram(percussive_audio)

    harmonic_melspec = torch.squeeze(harmonic_melspec, 0).detach().cpu().numpy().T
    harmonic_energy = torch.squeeze(harmonic_energy, 0).detach().cpu().numpy()
    percussive_melspec = torch.squeeze(percussive_melspec, 0).detach().cpu().numpy().T
    percussive_energy = torch.squeeze(percussive_energy, 0).detach().cpu().numpy()
    
    # melspec, energy  = _stft.mel_spectrogram(audio_norm)
    
    # plt.figure()  # Define the figure size (optional)
    # plt.imshow(mel_plot[0], cmap='viridis')  # Plot the 2D array
    # plt.colorbar()  # Add a colorbar (optional)
    # plt.title('2D NumPy Array Plot')  # Add a title (optional)
    # plt.show()
    # plt.gca().invert_yaxis()
    # plt.savefig('mel[0].png')
    
    # melspec = torch.squeeze(melspec, 0).detach().cpu().numpy().T
    # energy = torch.squeeze(energy, 0).detach().cpu().numpy()

    # mel_plot = melspec.T
    # plt.figure()  # Define the figure size (optional)
    # plt.imshow(mel_plot, cmap='viridis')  # Plot the 2D array
    # plt.colorbar()  # Add a colorbar (optional)
    # plt.title('2D NumPy Array Plot')  # Add a title (optional)
    # plt.show()
    # plt.gca().invert_yaxis()
    # plt.savefig('mel[0].png')
    
    return harmonic_melspec, harmonic_energy, percussive_melspec, percussive_energy