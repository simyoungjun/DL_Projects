from utils.path import *
from utils.audio.tools import get_mel

from tqdm import tqdm
import numpy as np
import glob, os, sys
from multiprocessing import Pool

from scipy.io.wavfile import write
import librosa, ffmpeg
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import librosa.display

def job(wav_filename):
    
# /root/sim/VoiceConversion/Datasets/VCTK/VCTK-Corpus/wavs/p240/p240_001_mic1.flac

	original_wav_filename, prepro_wav_dir, sampling_rate = wav_filename
	filename = original_wav_filename.split("/")[-1]
	formatted_path =prepro_wav_dir[:-5]
	# new_wav_filename = get_path(formatted_path,'/formatted/',filename)
	new_wav_filename = formatted_path+'/formatted/'+filename

	if not os.path.exists(new_wav_filename):
		try:
			out, err = (ffmpeg
					.input(original_wav_filename)
					# .output(new_wav_filename, acodec='pcm_s16le', ac=1, ar=sampling_rate)
					.output(new_wav_filename, ar=sampling_rate)
     
					.overwrite_output()
					.run(capture_stdout=True, capture_stderr=True))

		except ffmpeg.Error as err:
			print(err.stderr, file=sys.stderr)
			raise


def preprocess(data_path, prepro_wav_dir, prepro_path, mel_path, sampling_rate, n_workers=4, filter_length=1024, hop_length=256, trim_silence=True, top_db=60):
	p = Pool(n_workers)

	mel_scaler = StandardScaler(copy=False)
 
	prepro_wav_dir = create_dir(prepro_wav_dir)
	# wav_paths=[[filename, prepro_wav_dir, sampling_rate] for filename in list(glob.glob(get_path(data_path, "wav48", "**", "*.wav")))]
	wav_paths=[[filename, prepro_wav_dir, sampling_rate] for filename in list(glob.glob(get_path(data_path, "wavs", "**", "*.flac")))]
 

	print("\t[LOG] converting wav format...")
	with tqdm(total=len(wav_paths)) as pbar:
		for _ in tqdm(p.imap_unordered(job, wav_paths),):
			pbar.update()	

	formatted_path =prepro_wav_dir[:-5]+'/formatted'
 
	print("\t[LOG] saving mel-spectrogram...")
	with tqdm(total=len(wav_paths)) as pbar:
		for wav_filename in tqdm(glob.glob(get_path(formatted_path, "*.flac"))):
			mel_filename = wav_filename.split("/")[-1].replace("flac", "npy")
			mel_savepath = get_path(mel_path, mel_filename)

   
			mel_spectrogram, _ = get_mel(wav_filename, trim_silence=trim_silence, frame_length=filter_length, hop_length=hop_length, top_db=top_db)

			mel_scaler.partial_fit(mel_spectrogram)

   
			# plt.figure()  # Define the figure size (optional)
			# plt.imshow(mel_spectrogram.T, cmap='viridis')  # Plot the 2D array
			# plt.colorbar()  # Add a colorbar (optional)
			# plt.title('2D NumPy Array Plot')  # Add a title (optional)
			# plt.show()
			# plt.gca().invert_yaxis()
			# plt.savefig('mel[1].png')

   
			np.save(mel_savepath, mel_spectrogram)

   
	np.save(get_path(prepro_path, "mel_stats.npy"), np.array([mel_scaler.mean_, mel_scaler.scale_]))

	print("Done!")


def subplot_MHP(mel_spectrogram, harmonic_mel, percussive_mel):
 
	# 서브플롯 설정
	n_rows = 3  # 행 수
	n_cols = 1  # 열 수

	# 서브플롯에 멜 스펙트로그램 그리기
	for sub_idx, mel in zip(range(1, n_rows * n_cols + 1),[mel_spectrogram, harmonic_mel, percussive_mel]):
		plt.subplot(n_rows, n_cols, sub_idx)
		plt.imshow(mel.T, cmap='viridis')
		plt.gca().invert_yaxis()
  
		plt.colorbar(format='%+2.0f dB')
		plt.title(f'Subplot {sub_idx}')

	# 그래프 출력
	plt.tight_layout()
	plt.savefig('MHP_prepro')


def split_unseen_speakers(prepro_mel_dir):

	print("[LOG] 6 UNSEEN speakers:  \n\t p226(Male, English, Surrey) \n\t p256(Male, English, Birmingham) \
					 \n\t p266(Female, Irish, Athlone) \n\t p297(Female, American, Newyork) \
					 \n\t p323 (Female, SouthAfrican, Pretoria)\n\t p376(Male, Indian)")

	unseen_speaker_list = ["p226", "p256", "p266", "p297", "p323", "p376"]

	seen_speaker_files, unseen_speaker_files = [], []

	preprocessed_file_list = glob.glob(get_path(prepro_mel_dir, "*.npy"))
	
	for preprocessed_mel_file in preprocessed_file_list:
		speaker = preprocessed_mel_file.split("/")[-1].split("_")[0]
		if speaker in unseen_speaker_list:
			unseen_speaker_files.append(preprocessed_mel_file)
		else:
			seen_speaker_files.append(preprocessed_mel_file)	
	
	return seen_speaker_files, unseen_speaker_files




