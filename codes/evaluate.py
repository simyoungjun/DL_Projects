from config import Arguments as args

from utils.vocoder import vocgan_infer
from utils.path import create_dir, get_path
from utils.dataset import de_normalize

import wandb
import torch
import matplotlib.pyplot as plt
import numpy as np

def evaluate(model, vocoder, eval_data_loader, criterion, global_step, mel_stat, writer=None, DEVICE=None):

	eval_path = create_dir(args.eval_path)
	model.eval()
	mel_mean, mel_std = mel_stat 

	with torch.no_grad():
		eval_loss, eval_recon_loss, eval_perplexity, eval_commitment_loss = 0, 0, 0, 0

		for step, (mels, _) in enumerate(eval_data_loader):

			mels = mels.float().to(DEVICE)

			mels_hat, mels_code, mels_style, commitment_loss, perplexity = model.evaluate(mels.detach())

			commitment_loss = args.commitment_cost * commitment_loss
			recon_loss = criterion(mels, mels_hat)

			total_loss = commitment_loss + recon_loss

			eval_perplexity += perplexity.item()
			eval_recon_loss += recon_loss.item()
			eval_commitment_loss += commitment_loss.item()
			eval_loss += total_loss.item()

		mel = de_normalize(mels[0], mean=mel_mean, std=mel_std).float()
		mel_hat = de_normalize(mels_hat[0], mean=mel_mean, std=mel_std).float()
		mel_code = de_normalize(mels_code[0], mean=mel_mean, std=mel_std).float()
		mel_style = de_normalize(mels_style[0], mean=mel_mean, std=mel_std).float()

		vocgan_infer(mel.transpose(0, 1), vocoder, path=get_path(args.eval_path, "{:0>3}_GT.wav".format(global_step//1000)))
		vocgan_infer(mel_hat.transpose(0, 1), vocoder, path=get_path(args.eval_path, "{:0>3}_reconstructed.wav".format(global_step//1000)))
		vocgan_infer(mel_code.transpose(0, 1), vocoder, path=get_path(args.eval_path, "{:0>3}_code.wav".format(global_step//1000)))
		vocgan_infer(mel_style.transpose(0, 1), vocoder, path=get_path(args.eval_path, "{:0>3}_style.wav".format(global_step//1000)))

		mel =  np.flipud(mel.view(-1, args.n_mels).detach().cpu().numpy().T)
		mel_hat = np.flipud(mel_hat.view(-1, args.n_mels).detach().cpu().numpy().T)
		mel_code = np.flipud(mel_code.view(-1, args.n_mels).detach().cpu().numpy().T)
		mel_style = np.flipud(mel_style.view(-1, args.n_mels).detach().cpu().numpy().T)

		# if args.log_tensorboard:
		# 	writer.add_scalars(mode="eval_reconstruction_loss", global_step=global_step, loss=eval_recon_loss / len(eval_data_loader))
		# 	writer.add_scalars(mode="eval_commitment_loss", global_step=global_step, loss=eval_commitment_loss / len(eval_data_loader))
		# 	writer.add_scalars(mode="eval_perplexity", global_step=global_step, loss=eval_perplexity / len(eval_data_loader))
		# 	writer.add_scalars(mode="eval_total_loss", global_step=global_step, loss=eval_loss / len(eval_data_loader))
		# 	writer.add_mel_figures(mode="eval-mels_", global_step=global_step, mel=mel, mel_hat=mel_hat, mel_code=mel_code, mel_style=mel_style)


		# plt.figure()  # Define the figure size (optional)
		# plt.imshow(mel, cmap='viridis', interpolation='nearest')  # Plot the 2D array
		# plt.colorbar()  # Add a colorbar (optional)
		# plt.title('2D NumPy Array Plot')  # Add a title (optional)

		# # Display the plot
		# plt.show()
  
		if args.log_wandb:
			wandb.log({
				"eval_reconstruction_loss": eval_recon_loss / len(eval_data_loader),
				"eval_commitment_loss": eval_commitment_loss / len(eval_data_loader),
				"eval_perplexity": eval_perplexity / len(eval_data_loader),
				"train_total_loss": eval_loss / len(eval_data_loader),
				"eval_mel": wandb.Image(mel),
    
				"eval_mel_hat": wandb.Image(mel_hat),
				"eval_mel_code": wandb.Image(mel_code),
				"eval_mel_style": wandb.Image(mel_style),
	
			})
		

	
