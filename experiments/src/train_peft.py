import os
import sys
import click
import yaml
import time
import shutil
import random
import copy
import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from torch.utils.tensorboard import SummaryWriter

# Add root directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from meldataset import build_dataloader
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from optimizers import build_optimizer

# PEFT Wrappers
from peft_models import PEFTStyleTTS2

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

@click.command()
@click.option('-p', '--config_path', default='Configs/config_ft.yml', type=str)
@click.option('-m', '--mode', default='A', type=str, help='PEFT Mode: A (LoRA), B (Style-Adapter), C (Prefix-Tuning), D (Duration-Adapter)')
@click.option('-r', '--rank', default=8, type=int, help='LoRA rank (only for mode A)')
@click.option('-c', '--checkpoint_path', default='Models/Base_stage2/epoch_2nd_00020.pth', type=str, help='Path to frozen base model checkpoint (Indonesian)')
def main(config_path, mode, rank, checkpoint_path):
    config = yaml.safe_load(open(config_path))
    
    # Set unique log directory for each PEFT scenario to avoid mixing files
    log_dir = f"Models/peft_scenario_{mode}"
    config['log_dir'] = log_dir
    
    if not osp.exists(log_dir): 
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 8)
    epochs = config.get('epochs', 20) # 20 epochs is normally plenty for adapters to converge on 30m
    save_freq = config.get('save_freq', 5)
    log_interval = config.get('log_interval', 10)

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']

    max_len = config.get('max_len', 200)
    
    loss_params = Munch(config['loss_params'])
    
    optimizer_params = Munch(config['optimizer_params'])
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        OOD_data=OOD_data,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        num_workers=2,
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=0,
                                      device=device,
                                      dataset_config={})
    
    # Load support models
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)
    
    # Build model dictionary
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    
    # DP
    for key in model:
        if key != "mpd" and key != "msd" and key != "wd":
            model[key] = MyDataParallel(model[key])
            
    # Load Pretrained Base Model Checkpoint (Freezing target)
    print(f"PEFT: Loading base model checkpoint from {checkpoint_path}...")
    model, _, _, _, _ = load_checkpoint(model, None, checkpoint_path, load_only_params=True)

    # Wrap model with PEFT Module & Freeze Base Parameters
    peft_wrapper = PEFTStyleTTS2(model, mode=mode, rank=rank)
    _ = [model[key].to(device) for key in model]

    # Filter parameters to only train PEFT weights
    trainable_params_dict = {}
    for key in model:
        # We also need WavLM discriminator (wd) to train if joint stage is active, 
        # but for PEFT we keep discriminators frozen or exclude them. Let's keep them frozen for stability.
        params = [p for p in model[key].parameters() if p.requires_grad]
        if len(params) > 0:
            trainable_params_dict[key] = params
            print(f"PEFT: Module '{key}' has {len(params)} trainable parameter tensors.")

    # Configure learning rates & Optimizers
    scheduler_params = {
        "max_lr": optimizer_params.ft_lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict = {key: scheduler_params.copy() for key in trainable_params_dict}
    
    # Initialize MultiOptimizer only on trainable parameter keys
    optimizer = build_optimizer(trainable_params_dict, scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.ft_lr)
    
    n_down = model.text_aligner.n_down
    criterion = nn.L1Loss() # F0 loss (regression)
    torch.cuda.empty_cache()
    
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    print(f"PEFT initialization complete. Mode: {mode}. Trainable keys: {list(trainable_params_dict.keys())}")
    
    iters = 0
    for epoch in range(epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].eval() for key in model]
        
        # Un-freeze train mode for modules holding our adapters
        for key in trainable_params_dict:
            model[key].train()

        # Modules containing LSTMs/RNNs must be in train mode for backward pass to work in cuDNN
        model.text_aligner.train()
        model.text_encoder.train()
        model.predictor.train()
        model.bert_encoder.train()
        model.bert.train()

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
            
            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)
                
            try:
                ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)
                s2s_attn = s2s_attn[..., 1:]
                s2s_attn = s2s_attn.transpose(-1, -2)
            except:
                continue

            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode text
            t_en = model.text_encoder(texts, input_lengths, text_mask)
            asr = (t_en @ s2s_attn_mono)
            d_gt = s2s_attn_mono.sum(axis=-1).detach()

            # compute style
            ss = []
            for bib in range(len(mel_input_length)):
                mel = mels[bib, :, :mel_input_length[bib]]
                s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)

            s_dur = torch.stack(ss).squeeze()  # global prosodic styles
            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            
            # Forward adapter/model
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
            
            d, p = model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)
                
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            en = []
            gt = []
            p_en = []
            wav = []
            
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start+mel_len])
                p_en.append(p[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                
                y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))
                
            wav = torch.stack(wav).float().detach()
            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            
            if gt.size(-1) < 80:
                continue
            
            s = model.style_encoder(gt.unsqueeze(1))           
            s_dur_clip = model.predictor_encoder(gt.unsqueeze(1))
                
            with torch.no_grad():
                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()
                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
                
            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur_clip)

            loss_F0_rec = (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            # Backward step
            optimizer.zero_grad()

            loss_dur = 0
            loss_ce = 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p_idx in range(_s2s_trg.shape[0]):
                    _s2s_trg[p_idx, :_text_input[p_idx]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], _text_input[1:_text_length-1])
                loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)

            # Total Generator loss for adapter tuning
            g_loss = loss_params.lambda_F0 * loss_F0_rec + \
                     loss_params.lambda_ce * loss_ce + \
                     loss_params.lambda_norm * loss_norm_rec + \
                     loss_params.lambda_dur * loss_dur
            
            running_loss += g_loss.item()
            g_loss.backward()
            
            # Step optimizers for trainable layers
            for key in trainable_params_dict:
                optimizer.step(key)
            
            iters = iters + 1
            
            if (i+1)%log_interval == 0:
                logger.info('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f'
                    %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, loss_dur, loss_ce, loss_norm_rec, loss_F0_rec))
                
                writer.add_scalar('train/loss', running_loss / log_interval, iters)
                running_loss = 0
                
        # Save checkpoints
        if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
            print(f"Saving PEFT checkpoint for epoch {epoch + 1}...")
            state = {
                'net': {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'epoch': epoch,
                'peft_mode': mode
            }
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)

if __name__ == "__main__":
    main()
