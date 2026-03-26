import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import math
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertModel, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, PegasusForConditionalGeneration, PegasusTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderConfig, EncoderDecoderModel, AutoModelForCausalLM, AutoTokenizer
from Data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive, T5Translator, R1Translator, EEGConformer, BrainBERT
from config import get_config

def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path_best = './checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last = './checkpoints/decoding/last/temp_decoding.pt', max_grad_norm=1.0, patience=0, use_scheduled_sampling=False, accumulation_steps=2, plot_save_path=None):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
      
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Scheduled sampling: sigmoid decay with 40% warmup
        if use_scheduled_sampling:
            warmup_frac = 0.4
            min_tf = 0.3
            progress = epoch / max(1, num_epochs - 1)
            if progress < warmup_frac:
                tf_ratio = 1.0
            else:
                adjusted = (progress - warmup_frac) / (1.0 - warmup_frac)
                tf_ratio = min_tf + (1.0 - min_tf) / (1.0 + math.exp(10 * (adjusted - 0.5)))
            print(f'[Scheduled Sampling] tf_ratio = {tf_ratio:.3f}')
        else:
            tf_ratio = 1.0

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for batch_idx, (input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels) in enumerate(tqdm(dataloaders[phase])):
                
                # load in batch
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                """replace padding ids in target_ids with -100"""
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                # zero the parameter gradients only at accumulation boundary
                if phase == 'train' and batch_idx % accumulation_steps == 0:
                    optimizer.zero_grad()

                # forward
    	        # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch, tf_ratio=tf_ratio if phase == 'train' else 1.0)

                    """calculate loss"""
                    loss = seq2seqLMoutput.loss # use the BART language modeling loss
                
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        (loss.sum() / accumulation_steps).backward()
                        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloaders[phase]):
                            # Gradient clipping to prevent exploding gradients
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                # statistics
                running_loss += loss.sum().item() * input_embeddings_batch.size()[0] # batch loss

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

            # deep copy the model
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
            elif phase == 'dev':
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch} (no improvement for {patience} epochs)')
                    break
        else:
            # This else belongs to the for-loop (not if), continues to next epoch
            print()
            continue
        # If inner loop was broken (early stopping), break outer loop too
        break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')

    # plot training and validation loss curves
    if plot_save_path is not None and (train_losses or val_losses):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Train Loss', marker='o', markersize=3)
        ax.plot(val_losses, label='Validation Loss', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plot_dir = os.path.dirname(plot_save_path)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_save_path, dpi=150)
        plt.close()
        print(f'[INFO] Loss curve saved to {plot_save_path}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)


if __name__ == '__main__':
    # Main argument parser (add all required and ablation args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, required=True)
    parser.add_argument('--task_name', '-t', type=str, required=True)
    parser.add_argument('--num_epoch_step1', '-ne1', type=int, required=True)
    parser.add_argument('--num_epoch_step2', '-ne2', type=int, required=True)
    parser.add_argument('--learning_rate_step1', '-lr1', type=float, required=True)
    parser.add_argument('--learning_rate_step2', '-lr2', type=float, required=True)
    parser.add_argument('--batch_size', '-b', type=int, required=True)
    parser.add_argument('--save_path', '-s', type=str, required=True)
    parser.add_argument('--train_input', type=str, required=True)
    # Optional/advanced args (add as needed)
    parser.add_argument('--subjects', type=str, default='ALL', help='use all subjects or specify a particular one (default: ALL)')
    parser.add_argument('--eeg_type', type=str, default='GD', help='choose from {GD, FFD, TRT} (default: GD)')
    parser.add_argument('--eeg_bands', type=str, nargs='+', default=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], help='specify freqency bands (default: all 8 bands)')
    parser.add_argument('--cuda', type=str, default='cuda:0')
    parser.add_argument('--skip_step_one', action='store_true')
    parser.add_argument('--load_step1_checkpoint', action='store_true')
    parser.add_argument('--use_random_init', action='store_true')
    # Ablation flags
    parser.add_argument('--ablate_sba', type=str, default='False', help='Ablate Spectral Band Attention (True, False, or uniform)')
    parser.add_argument('--ablate_multiscale', action='store_true', help='Ablate Multi-Scale Conv1D')
    parser.add_argument('--ablate_cab', action='store_true', help='Ablate Cross-Attention Bridge')
    parser.add_argument('--ablate_label_smoothing', action='store_true', help='Ablate Label Smoothing')
    parser.add_argument('--conv_kernels', type=str, default='1,3,5', help='Comma separated list of conv kernel sizes')
    parser.add_argument('--feature_level', type=str, default='word', help='Level of EEG feature representations: word or sentence')


    cli_args = parser.parse_args()

    # Load config as before (if you want to merge with config file, do so here)
    args = vars(cli_args)

    ''' config param'''
    dataset_setting = 'unique_sent'
    
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    
    batch_size = args['batch_size']
    
    model_name = args['model_name']
    # model_name = 'BrainTranslatorNaive' # with no additional transformers
    # model_name = 'BrainTranslator' 
    
    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    task_name = args['task_name']
    train_input = args['train_input']
    print("train_input is:", train_input)   
    save_path = args['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = args['use_random_init']
    device_ids = [0] # device setting

    # --- Ablation flags (set these to True for ablation runs) ---
    ablate_sba_val = str(args.get('ablate_sba', 'False'))
    if ablate_sba_val.lower() == 'true':
        ablate_sba = True
    elif ablate_sba_val.lower() == 'false':
        ablate_sba = False
    else:
        ablate_sba = ablate_sba_val # e.g. 'uniform'
        
    ablate_multiscale = args.get('ablate_multiscale', False)
    ablate_cab = args.get('ablate_cab', False)
    ablate_label_smoothing = args.get('ablate_label_smoothing', False)
    
    conv_kernels_str = args.get('conv_kernels', '1,3,5')
    conv_kernels = tuple(map(int, conv_kernels_str.split(',')))
    feature_level = args.get('feature_level', 'word')

    args['ablate_sba'] = ablate_sba
    args['conv_kernels'] = conv_kernels_str
    args['feature_level'] = feature_level

    if use_random_init and skip_step_one:
        step2_lr = 5*1e-4

    print(f'[INFO]using model: {model_name}')
    print(f'[INFO]Ablation flags: SBA={ablate_sba}, MultiScale={ablate_multiscale}, CAB={ablate_cab}, LabelSmoothing={ablate_label_smoothing}')

    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}'

    if ablate_sba == 'uniform':
        save_name += '_uniformSBA'
    elif ablate_sba:
        save_name += '_noSBA'
        
    if ablate_multiscale:
        save_name += '_noMSC'
    elif conv_kernels_str != '1,3,5':
        save_name += f'_MSCkernels{conv_kernels_str.replace(",","")}'
        
    if ablate_cab:
        save_name += '_noCAB'
    if ablate_label_smoothing:
        save_name += '_noLS'
        
    if feature_level == 'sentence':
        save_name += '_sentEEG'

    if use_random_init:
        save_name = 'randinit_' + save_name

    # Use save_name for unique checkpoint paths
    output_checkpoint_name_best = os.path.join('checkpoints', 'decoding', 'best', f'{save_name}.pt')
    output_checkpoint_name_last = os.path.join('checkpoints', 'decoding', 'last', f'{save_name}.pt')

    save_path_best = os.path.join(save_path, 'best')
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)

    output_checkpoint_name_best = os.path.join(save_path_best, f'{save_name}.pt')

    save_path_last = os.path.join(save_path, 'last')
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)

    output_checkpoint_name_last = os.path.join(save_path_last, f'{save_name}.pt')

    # subject_choice = 'ALL
    subject_choice = args['subjects']
    print(f'![Debug]using {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    # bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')


    
    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        # dev = "cuda:3" 
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    ''' set up dataloader '''
    PICKLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'pickle_file')
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = os.path.join(PICKLE_DIR, 'task1-SR-dataset.pickle')
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = os.path.join(PICKLE_DIR, 'task2-NR-dataset.pickle')
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = os.path.join(PICKLE_DIR, 'task3-TSR-dataset.pickle')
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
       dataset_path_taskNRv2 = os.path.join(PICKLE_DIR, 'task2-TSR-2.0-dataset.pickle')
       with open(dataset_path_taskNRv2, 'rb') as handle:
           whole_dataset_dicts.append(pickle.load(handle))

    print()

    """save config"""
    cfg_dir = './config/decoding/'

    if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)

    with open(os.path.join(cfg_dir,f'{save_name}.json'), 'w') as out_config:
        json.dump(args, out_config, indent = 4)

    if model_name in ['BrainTranslator','BrainTranslatorNaive','R1Translator','EEGConformer']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    elif model_name == 'PegasusTranslator':
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    
    elif model_name == 'T5Translator':
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        #tokenizer.set_prefix_tokens(language='english')
    elif model_name == 'BrainBERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input, feature_level=feature_level)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input, feature_level=feature_level)
    # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, feature_level=feature_level)

    
    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    # print('[INFO]test_set size: ', len(test_set))
    
    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=0)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = 1, shuffle=False, num_workers=0)
    # dataloaders
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader}

    ''' set up model '''
    if model_name == 'BrainTranslator':
        if use_random_init:
            config = BartConfig.from_pretrained('facebook/bart-large')
            config.label_smoothing = 0.1
            pretrained = BartForConditionalGeneration(config)
        else:
            pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
            pretrained.config.label_smoothing = 0.1
    
        model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    
    elif model_name == 'BrainTranslatorNaive':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        pretrained.config.label_smoothing = 0.1
        model = BrainTranslatorNaive(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    elif model_name == 'PegasusTranslator':
        pretrained = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
        model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    
    elif model_name == 'T5Translator':
        pretrained = T5ForConditionalGeneration.from_pretrained("t5-large")
        model = T5Translator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    elif model_name == 'R1Translator':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large', torch_dtype=torch.float32)
        pretrained = pretrained.float()  # Ensure all parameters are float32
        pretrained.config.label_smoothing = 0.1
        model = R1Translator(
                pretrained,
                in_feature = 105*len(bands_choice),
                decoder_embedding_size = pretrained.config.d_model,
                rnn_hidden_size = 256,
                num_rnn_layers=2
            )
    elif model_name == 'EEGConformer':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large', torch_dtype=torch.float32)
        pretrained = pretrained.float()
        # Set label smoothing to 0 if ablation, else 0.1
        pretrained.config.label_smoothing = 0.0 if ablate_label_smoothing else 0.1
        model = EEGConformer(
                pretrained,
                in_feature=105*len(bands_choice),
                decoder_embedding_size=pretrained.config.d_model,
                n_bands=len(bands_choice),
                n_electrodes=105,
                conv_channels=105*len(bands_choice),
                rnn_hidden_size=512,
                num_rnn_layers=3,
                n_bridge_heads=8,
                dropout=0.3,
                ablate_sba=ablate_sba,
                ablate_multiscale=ablate_multiscale,
                ablate_cab=ablate_cab,
                conv_kernels=conv_kernels
            )
    elif model_name == 'BrainBERT':
        bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        bart_decoder = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = BrainBERT(
                bert_encoder,
                bart_decoder,
                in_feature = 105*len(bands_choice)
            )
    
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    ''' training loop '''

    ######################################################
    '''step one trainig: freeze most of BART params'''
    ######################################################

    # closely follow BART paper
    if model_name in ['BrainTranslator','BrainTranslatorNaive', 'PegasusTranslator', 'T5Translator', 'R1Translator', 'EEGConformer']:
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:

                #New Add

                if 'DeepSeekTranslator' in model_name:
                    if any(key in name for key in ['wte', 'wpe', 'h.0']):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

                else: # Old
                    if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                        continue
                    if ('shared' in name) or ('embed' in name) or ('layer.0' in name) or ('wte' in name):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

    elif model_name == 'BertGeneration':
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('embeddings' in name) or ('encoder.layer.0' in name):
                    continue
                else:
                    param.requires_grad = False
 

    if skip_step_one:
        if load_step1_checkpoint:
            stepone_checkpoint = 'path_to_step_1_checkpoint.pt'
            print(f'skip step one, load checkpoint: {stepone_checkpoint}')
            model.load_state_dict(torch.load(stepone_checkpoint))
        else:
            print('skip step one, start from scratch at step two')
    else:

        ''' set up optimizer and scheduler'''
        optimizer_step1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, weight_decay=0.01)

        exp_lr_scheduler_step1 = CosineAnnealingWarmRestarts(optimizer_step1, T_0=5, T_mult=2, eta_min=1e-6)

        ''' set up loss function '''
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        if model_name == 'DeepSeckTranslator':
            model.module.pretrained.gradient_checkpointing_enable()

            scaler = torch.cuda.amp.GradScaler()

        print('=== start Step1 training ... ===')
        # print training layers
        show_require_grad_layers(model)
        # return best loss model from step1 training
        model = train_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last, plot_save_path=os.path.join('figures', f'{save_name}_step1_loss.png'))

    ######################################################
    '''step two trainig: update whole model for a few iterations'''
    ######################################################
    
    # Selectively unfreeze and use discriminative learning rates
    pretrained_params = []
    custom_params = []
    
    # Only unfreeze BART embeddings + first 4 encoder/decoder layers (freeze layers 4-11)
    unfreeze_keys = ['shared', 'embed_positions', 'embed_tokens', 'layernorm_embedding']
    for i in range(4):
        unfreeze_keys.append(f'encoder.layers.{i}.')
        unfreeze_keys.append(f'decoder.layers.{i}.')
    
    for name, param in model.named_parameters():
        if 'pretrained' in name:
            if any(key in name for key in unfreeze_keys):
                param.requires_grad = True
                pretrained_params.append(param)
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True
            custom_params.append(param)

    ''' set up optimizer and scheduler'''
    # Discriminative LR: pretrained BART at 10x lower rate than custom EEG layers
    optimizer_step2 = optim.AdamW([
        {'params': custom_params, 'lr': step2_lr},
        {'params': pretrained_params, 'lr': step2_lr * 0.1}
    ], weight_decay=0.01)

    # Warmup + cosine annealing scheduler (warmup over first 20% of epochs)
    warmup_epochs = max(1, num_epochs_step2 // 5)
    def warmup_cosine_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        progress = (epoch - warmup_epochs) / max(1, num_epochs_step2 - warmup_epochs)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    exp_lr_scheduler_step2 = lr_scheduler.LambdaLR(optimizer_step2, warmup_cosine_lambda)

    ''' set up loss function '''
    # Set label smoothing to 0 if ablation, else 0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0 if ablate_label_smoothing else 0.1)
    
    print()
    print('=== start Step2 training ... ===')
    # print training layers
    show_require_grad_layers(model)
    

    '''main loop'''
    trained_model = train_model(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last, patience=10, use_scheduled_sampling=True, plot_save_path=os.path.join('figures', f'{save_name}_step2_loss.png'))

    # '''save checkpoint'''
    # torch.save(trained_model.state_dict(), os.path.join(save_path,output_checkpoint_name))
