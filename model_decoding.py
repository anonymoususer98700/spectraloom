import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import math
import numpy as np

""" main architecture for open vocabulary EEG-To-Text decoding"""
class BrainTranslator(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(BrainTranslator, self).__init__()
        
        self.pretrained = pretrained_layers
        # additional transformer encoder, following BART paper about 
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        
        # print('[INFO]adding positional embedding')
        # self.positional_embedding = PositionalEncoding(in_feature)

        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def addin_forward(self,input_embeddings_batch,  input_masks_invert):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""

        # input_embeddings_batch = self.positional_embedding(input_embeddings_batch)
        # use src_key_padding_masks
        encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask=input_masks_invert)

        # encoded_embedding = self.additional_encoder(input_embeddings_batch)
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        return encoded_embedding

    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted,
            generation_config = None,
            logits_processor = None,
            stopping_criteria = None,
            prefix_allowed_tokens_fn= None,
            synced_gpus= None,
            assistant_model = None,
            streamer= None,
            negative_prompt_ids= None,
            negative_prompt_attention_mask = None,
            **kwargs,
    ):
        encoded_embedding=self.addin_forward(input_embeddings_batch, input_masks_invert)
        attention_mask = input_masks_batch[:, :encoded_embedding.size(1)]
        output=self.pretrained.generate(
            inputs_embeds = encoded_embedding,
            attention_mask = attention_mask,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,)

        return output

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        encoded_embedding=self.addin_forward(input_embeddings_batch, input_masks_invert)
        # print(f'forward:{input_embeddings_batch.shape,input_masks_batch.shape,input_masks_invert.shape,target_ids_batch_converted.shape,encoded_embedding.shape}')
        out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch,
                              return_dict = True, labels = target_ids_batch_converted)
        
        return out


from transformers import T5Tokenizer
""" main architecture for open vocabulary EEG-To-Text decoding"""

class T5Translator(nn.Module):
    def __init__(self, pretrained_layers, in_feature=840, decoder_embedding_size=1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048):
        super(T5Translator, self).__init__()
        
        self.pretrained = pretrained_layers
        self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        
        # Additional transformer encoder
        self.additional_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature, 
            nhead=additional_encoder_nhead, 
            dim_feedforward=additional_encoder_dim_feedforward, 
            batch_first=True
        )
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def addin_forward(self, input_embeddings_batch, input_masks_invert):
        """Process EEG features through additional encoder"""
        encoded_embedding = self.additional_encoder(
            input_embeddings_batch, 
            src_key_padding_mask=input_masks_invert
        )
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        return encoded_embedding

    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted,
            **kwargs  # Accept variable arguments for generation
    ):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)

        # Add task-specific prefix
        input_ids = self.tokenizer("transcribe in English: ", return_tensors="pt").input_ids.to(encoded_embedding.device)
        self.task_embedding = self.pretrained.shared(input_ids).to(encoded_embedding.device)
        task_embedding = self.task_embedding.repeat(encoded_embedding.size(0), 1, 1).to(encoded_embedding.device)
        encoded_embedding = torch.cat((task_embedding, encoded_embedding), dim=1)
        input_masks_batch = torch.cat((torch.ones(encoded_embedding.size(0), task_embedding.size(1)).to(encoded_embedding.device), input_masks_batch), dim=1)

        # Generate text
        output = self.pretrained.generate(
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch[:, :encoded_embedding.shape[1]],
            **kwargs  # Pass all generation parameters
        )
        return output

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)
        
        # Add task-specific prefix
        input_ids = self.tokenizer("transcribe in English: ", return_tensors="pt").input_ids.to(encoded_embedding.device)
        self.task_embedding = self.pretrained.shared(input_ids).to(encoded_embedding.device)
        task_embedding = self.task_embedding.repeat(encoded_embedding.size(0), 1, 1).to(encoded_embedding.device)
        encoded_embedding = torch.cat((task_embedding, encoded_embedding), dim=1)
        input_masks_batch = torch.cat((torch.ones(encoded_embedding.size(0), task_embedding.size(1)).to(encoded_embedding.device), input_masks_batch), dim=1)

        # Forward pass
        out = self.pretrained(
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch,
            labels=target_ids_batch_converted,
            return_dict=True
        )
        return out


""" crippled open vocabulary EEG-To-Text decoding model w/o additional MTE encoder"""
class BrainTranslatorNaive(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(BrainTranslatorNaive, self).__init__()
        '''no additional transformer encoder version'''
        self.pretrained = pretrained_layers
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        encoded_embedding = F.relu(self.fc1(input_embeddings_batch))
        out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch, return_dict = True, labels = target_ids_batch_converted)                    
        return out


""" helper modules """
# modified from BertPooler
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('[DEBUG] input size:', x.size())
        # print('[DEBUG] positional embedding size:', self.pe.size())
        x = x + self.pe[:x.size(0), :]
        # print('[DEBUG] output x with pe size:', x.size())
        return self.dropout(x)


""" Miscellaneous (not working well) """
class BrainTranslatorBert(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, hidden_size = 768):
        super(BrainTranslatorBert, self).__init__()

        self.pretrained_Bert = pretrained_layers
        self.fc1 = nn.Linear(in_feature, hidden_size)

    def forward(self, input_embeddings_batch, input_masks_batch, target_ids_batch):
        embedding = F.relu(self.fc1(input_embeddings_batch))
        out = self.pretrained_Bert(inputs_embeds = embedding, attention_mask = input_masks_batch, labels = target_ids_batch, return_dict = True)
        return out

class EEG2BertMapping(nn.Module):
    def __init__(self, in_feature = 840, hidden_size = 512, out_feature = 768):
        super(EEG2BertMapping, self).__init__()
        self.fc1 = nn.Linear(in_feature, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_feature)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

class ContrastiveBrainTextEncoder(nn.Module):
    def __init__(self, pretrained_text_encoder, in_feature = 840, eeg_encoder_nhead=8, eeg_encoder_dim_feedforward = 2048, embed_dim = 768):
        super(ContrastiveBrainTextEncoder, self).__init__()
        # EEG Encoder
        self.positional_embedding = PositionalEncoding(in_feature)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=eeg_encoder_nhead,  dim_feedforward = eeg_encoder_dim_feedforward, batch_first=True)
        self.EEG_Encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.EEG_pooler = Pooler(in_feature)
        self.ln_final = nn.LayerNorm(in_feature) # to be considered
        
        # project to text embedding
        self.EEG_projection = nn.Parameter(torch.empty(in_feature, embed_dim))
        
        # Text Encoder
        self.TextEncoder = pretrained_text_encoder
        
        # learned temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_EEG_features, input_EEG_attn_mask, input_ids, input_text_attention_masks):
        # add positional embedding
        input_EEG_features = self.positional_embedding(input_EEG_features)
        # get EEG feature embedding
        EEG_hiddenstates = self.EEG_Encoder(input_EEG_features,  src_key_padding_mask = input_EEG_attn_mask)
        EEG_hiddenstates = self.ln_final(EEG_hiddenstates)
        EEG_features = self.EEG_pooler(EEG_hiddenstates) # [N, 840]

        # project to text embed size
        EEG_features = EEG_features @ self.EEG_projection # [N, 768]

        # get text feature embedding
        Text_features = self.TextEncoder(input_ids = input_ids, attention_mask = input_text_attention_masks, return_dict = True).pooler_output # [N, 768]
        
        # normalized features
        EEG_features = EEG_features / EEG_features.norm(dim=-1, keepdim=True) # [N, 768]
        Text_features = Text_features / Text_features.norm(dim=-1, keepdim=True) # [N, 768]

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp() 
        logits_per_EEG = logit_scale * EEG_features @ Text_features.t() # [N, N]
        logits_per_text = logit_scale * Text_features @ EEG_features.t() # [N, N]

        return logits_per_EEG, logits_per_text
 


from transformers import AutoModelForCausalLM, AutoTokenizer

class R1Translator(nn.Module):
    def __init__(
        self,
        pretrained_layers,
        in_feature=840,
        decoder_embedding_size=1024,
        rnn_hidden_size=256,
        bidirectional=True,
        num_rnn_layers=2,
        dropout=0.1
    ):
        super(R1Translator, self).__init__()

        self.pretrained = pretrained_layers

        # Input LayerNorm for EEG feature stabilization
        self.input_norm = nn.LayerNorm(in_feature)

        # Additional RNN encoder (LSTM) with dropout
        self.rnn_layer = nn.LSTM(
            input_size=in_feature,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_rnn_layers > 1 else 0.0
        )

        # Calculate RNN output size for bidirectional processing
        self.rnn_output_size = rnn_hidden_size * (2 if bidirectional else 1)

        # Post-LSTM normalization and regularization
        self.post_rnn_norm = nn.LayerNorm(self.rnn_output_size)
        self.post_rnn_dropout = nn.Dropout(dropout)

        # Projection to decoder embedding size with GELU activation
        self.fc1 = nn.Linear(self.rnn_output_size, decoder_embedding_size)
        self.fc_norm = nn.LayerNorm(decoder_embedding_size)
        self.fc_dropout = nn.Dropout(dropout)

        # Initialize custom layers
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        for name, param in self.rnn_layer.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def addin_forward(self, input_embeddings_batch, input_masks_invert):
        """Process EEG features through LSTM encoder with normalization"""
        # Normalize input EEG features
        x = self.input_norm(input_embeddings_batch)

        # BiLSTM encoding
        rnn_out, _ = self.rnn_layer(x)

        # Post-LSTM normalization and dropout
        rnn_out = self.post_rnn_norm(rnn_out)
        rnn_out = self.post_rnn_dropout(rnn_out)

        # Projection with GELU activation (smoother than ReLU, better for NLP)
        encoded_embedding = F.gelu(self.fc1(rnn_out))
        encoded_embedding = self.fc_norm(encoded_embedding)
        encoded_embedding = self.fc_dropout(encoded_embedding)

        return encoded_embedding

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted, tf_ratio=1.0):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)

        if self.training and tf_ratio < 1.0:
            # Scheduled sampling: add word dropout to decoder inputs to reduce exposure bias
            # Recreate decoder_input_ids by right-shifting labels (what BART does internally)
            pad_id = self.pretrained.config.pad_token_id
            bos_id = self.pretrained.config.decoder_start_token_id

            shifted = target_ids_batch_converted.clone()
            shifted[shifted == -100] = pad_id
            decoder_input_ids = shifted.new_zeros(shifted.shape)
            decoder_input_ids[:, 1:] = shifted[:, :-1]
            decoder_input_ids[:, 0] = bos_id

            # Randomly replace tokens with uniform random vocab tokens
            noise_mask = torch.rand(decoder_input_ids.shape, device=decoder_input_ids.device) > tf_ratio
            noise_mask[:, 0] = False  # Never noise the BOS token
            noise_mask[decoder_input_ids == pad_id] = False  # Never noise padding

            random_tokens = torch.randint(
                4, self.pretrained.config.vocab_size,
                decoder_input_ids.shape, device=decoder_input_ids.device
            )
            decoder_input_ids[noise_mask] = random_tokens[noise_mask]

            out = self.pretrained(
                inputs_embeds=encoded_embedding,
                attention_mask=input_masks_batch,
                decoder_input_ids=decoder_input_ids,
                labels=target_ids_batch_converted,
                return_dict=True
            )
        else:
            out = self.pretrained(
                inputs_embeds=encoded_embedding,
                attention_mask=input_masks_batch,
                labels=target_ids_batch_converted,
                return_dict=True
            )
        return out

    @torch.no_grad()
    def generate(
        self,
        input_embeddings_batch,
        input_masks_batch,
        input_masks_invert,
        target_ids_batch_converted,
        **kwargs
    ):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)
        output = self.pretrained.generate(
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch[:, :encoded_embedding.shape[1]],
            **kwargs
        )
        return output


"""Novel EEG-to-Text architecture with Spectral Band Attention, 
Multi-Scale Conv + BiLSTM encoder, and Cross-Attention Bridge."""

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).
    Injects temporal order information into the encoder output."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (B, L, d_model) -> (B, L, d_model) with positional info added"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SpectralBandAttention(nn.Module):
    """Learns to weight 8 EEG frequency bands per word via attention.
    Input (B, L, 840) is reshaped to (B, L, 8, 105), attention scores
    computed over the band dimension, then collapsed back."""
    def __init__(self, n_bands=8, n_electrodes=105, uniform=False):
        super().__init__()
        self.n_bands = n_bands
        self.n_electrodes = n_electrodes
        self.uniform = uniform
        # Small MLP to produce per-band attention logits
        if not uniform:
            self.band_query = nn.Linear(n_electrodes, 1, bias=False)
            self.band_norm = nn.LayerNorm(n_electrodes)

    def forward(self, x):
        """x: (B, L, 840) -> (B, L, 840) with band-reweighted features"""
        B, L, _ = x.shape
        # Reshape to (B, L, 8, 105)
        x_bands = x.view(B, L, self.n_bands, self.n_electrodes)
        
        if self.uniform:
            attn_weights = torch.ones(B, L, self.n_bands, 1, device=x.device) / self.n_bands
        else:
            # Compute attention scores per band: (B, L, 8, 1)
            attn_logits = self.band_query(self.band_norm(x_bands))
            attn_weights = F.softmax(attn_logits, dim=2)  # softmax over bands
            
        # Reweight bands
        x_weighted = x_bands * attn_weights  # (B, L, 8, 105)
        return x_weighted.reshape(B, L, -1)  # (B, L, 840)


class MultiScaleConv1D(nn.Module):
    """Parallel 1D convolutions with kernel sizes 1,3,5 to capture
    multi-scale local temporal EEG patterns."""
    def __init__(self, in_channels, out_channels, dropout=0.2, kernels=(1, 3, 5)):
        super().__init__()
        self.kernels = kernels
        # Each branch outputs out_channels // len(kernels) features
        branch_channels = out_channels // len(kernels)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, branch_channels, kernel_size=k, padding=k//2)
            for k in kernels
        ])
        
        # Handle remainder if out_channels not divisible
        total = branch_channels * len(kernels)
        self.proj = nn.Linear(total, out_channels) if total != out_channels else nn.Identity()
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, L, C) -> (B, L, out_channels)"""
        # Conv1d expects (B, C, L)
        xt = x.transpose(1, 2)
        
        c_outs = [F.gelu(conv(xt)) for conv in self.convs]
        
        # Concat and back to (B, L, C)
        out = torch.cat(c_outs, dim=1).transpose(1, 2)
        out = self.proj(out)
        out = self.norm(out)
        out = self.dropout(out)
        return out


class CrossAttentionBridge(nn.Module):
    """Multi-head cross-attention where learnable queries attend to encoder
    output, producing a richer representation for the decoder."""
    def __init__(self, d_encoder, d_decoder, n_heads=8, max_len=56, dropout=0.2):
        super().__init__()
        # Learnable query embeddings (one per position)
        self.query_embed = nn.Parameter(torch.randn(1, max_len, d_decoder) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_decoder, num_heads=n_heads,
            kdim=d_encoder, vdim=d_encoder,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_decoder)
        self.ffn = nn.Sequential(
            nn.Linear(d_decoder, d_decoder * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_decoder * 2, d_decoder),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_decoder)

    def forward(self, encoder_out, key_padding_mask=None):
        """encoder_out: (B, L, d_encoder) -> (B, L, d_decoder)"""
        B = encoder_out.size(0)
        queries = self.query_embed.expand(B, -1, -1)  # (B, L, d_decoder)
        # Cross-attention: queries attend to encoder output
        attn_out, _ = self.cross_attn(
            queries, encoder_out, encoder_out,
            key_padding_mask=key_padding_mask
        )
        out = self.norm1(queries + attn_out)
        out = self.norm2(out + self.ffn(out))
        return out


class EEGConformer(nn.Module):
    """Novel EEG-to-Text model with three key components:
    1. Spectral Band Attention — neuroscience-motivated band weighting
    2. Multi-Scale Conv1D + BiLSTM — multi-resolution temporal encoder
    3. Cross-Attention Bridge — rich encoder-decoder interface
    """
    def __init__(
        self,
        pretrained_layers,
        in_feature=840,
        decoder_embedding_size=1024,
        n_bands=8,
        n_electrodes=105,
        conv_channels=840,
        rnn_hidden_size=512,
        num_rnn_layers=3,
        n_bridge_heads=8,
        dropout=0.2,
        ablate_sba=False,
        ablate_multiscale=False,
        ablate_cab=False,
        conv_kernels=(1, 3, 5)
    ):
        super(EEGConformer, self).__init__()

        self.pretrained = pretrained_layers

        # Ablation flags
        self.ablate_sba = ablate_sba # False or 'uniform'
        self.ablate_multiscale = ablate_multiscale
        self.ablate_cab = ablate_cab
        self.conv_kernels = conv_kernels

        # --- Component 1: Spectral Band Attention ---
        self.input_norm = nn.LayerNorm(in_feature)
        if self.ablate_sba == 'uniform':
            self.band_attention = SpectralBandAttention(n_bands, n_electrodes, uniform=True)
        elif not self.ablate_sba:
            self.band_attention = SpectralBandAttention(n_bands, n_electrodes, uniform=False)

        # --- Component 2: Multi-Scale Conv1D + BiLSTM ---
        if not self.ablate_multiscale:
            self.multi_scale_conv = MultiScaleConv1D(in_feature, conv_channels, dropout, kernels=self.conv_kernels)
            conv_out_channels = conv_channels
        else:
            # Normalized linear substitute for Conv1D to be fair
            self.linear_conv = nn.Sequential(
                nn.Linear(in_feature, conv_channels),
                nn.LayerNorm(conv_channels),
                nn.GELU()
            )
            conv_out_channels = conv_channels

        self.rnn = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_rnn_layers > 1 else 0.0
        )
        self.rnn_output_size = rnn_hidden_size * 2  # bidirectional
        self.post_rnn_norm = nn.LayerNorm(self.rnn_output_size)
        self.post_rnn_dropout = nn.Dropout(dropout)

        # --- Positional Encoding (temporal order signal for BART cross-attention) ---
        self.pos_encoding = SinusoidalPositionalEncoding(self.rnn_output_size, dropout=dropout)

        # --- Component 3: Cross-Attention Bridge ---
        if not self.ablate_cab:
            self.bridge = CrossAttentionBridge(
                d_encoder=self.rnn_output_size,
                d_decoder=decoder_embedding_size,
                n_heads=n_bridge_heads,
                dropout=dropout
            )
        else:
            # Use a normalized linear projection to decoder space as a fair simple counterpart
            self.bridge_linear = nn.Sequential(
                nn.Linear(self.rnn_output_size, decoder_embedding_size),
                nn.LayerNorm(decoder_embedding_size),
                nn.Dropout(dropout)
            )

        # Initialize custom layers
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def addin_forward(self, input_embeddings_batch, input_masks_invert):
        """Full encoder pipeline: BandAttn -> Conv+LSTM -> CrossAttnBridge (with ablation)"""
        # Normalize raw EEG
        x = self.input_norm(input_embeddings_batch)

        # Input noise augmentation for regularization
        if self.training:
            x = x + torch.randn_like(x) * 0.02

        # Spectral Band Attention (skip if ablated, but keep uniform support)
        if self.ablate_sba == 'uniform' or not self.ablate_sba:
            x = self.band_attention(x)
        # else: pass through unchanged (raw ablate_sba fallback)

        # Multi-scale convolution (skip if ablated)
        if not self.ablate_multiscale:
            x = self.multi_scale_conv(x)
        else:
            x = self.linear_conv(x)

        # BiLSTM
        rnn_out, _ = self.rnn(x)
        rnn_out = self.post_rnn_norm(rnn_out)
        rnn_out = self.post_rnn_dropout(rnn_out)

        # Inject temporal position information
        rnn_out = self.pos_encoding(rnn_out)

        # Cross-Attention Bridge to decoder space (skip if ablated)
        if not self.ablate_cab:
            encoded = self.bridge(rnn_out, key_padding_mask=input_masks_invert)
        else:
            encoded = self.bridge_linear(rnn_out)
        return encoded

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted, tf_ratio=1.0):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)

        if self.training and tf_ratio < 1.0:
            pad_id = self.pretrained.config.pad_token_id
            bos_id = self.pretrained.config.decoder_start_token_id

            shifted = target_ids_batch_converted.clone()
            shifted[shifted == -100] = pad_id
            decoder_input_ids = shifted.new_zeros(shifted.shape)
            decoder_input_ids[:, 1:] = shifted[:, :-1]
            decoder_input_ids[:, 0] = bos_id

            noise_mask = torch.rand(decoder_input_ids.shape, device=decoder_input_ids.device) > tf_ratio
            noise_mask[:, 0] = False
            noise_mask[decoder_input_ids == pad_id] = False

            random_tokens = torch.randint(
                4, self.pretrained.config.vocab_size,
                decoder_input_ids.shape, device=decoder_input_ids.device
            )
            decoder_input_ids[noise_mask] = random_tokens[noise_mask]

            out = self.pretrained(
                inputs_embeds=encoded_embedding,
                attention_mask=input_masks_batch,
                decoder_input_ids=decoder_input_ids,
                labels=target_ids_batch_converted,
                return_dict=True
            )
        else:
            out = self.pretrained(
                inputs_embeds=encoded_embedding,
                attention_mask=input_masks_batch,
                labels=target_ids_batch_converted,
                return_dict=True
            )
        return out

    @torch.no_grad()
    def generate(
        self,
        input_embeddings_batch,
        input_masks_batch,
        input_masks_invert,
        target_ids_batch_converted,
        **kwargs
    ):
        encoded_embedding = self.addin_forward(input_embeddings_batch, input_masks_invert)
        output = self.pretrained.generate(
            inputs_embeds=encoded_embedding,
            attention_mask=input_masks_batch[:, :encoded_embedding.shape[1]],
            **kwargs
        )
        return output


class BrainBERT(nn.Module):
    def __init__(self, bert_encoder, bart_decoder, in_feature=840):
        super(BrainBERT, self).__init__()
        self.bert_encoder = bert_encoder
        self.bart_decoder = bart_decoder
        
        # Additional EEG encoder
        self.eeg_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_feature, 
            nhead=8, 
            dim_feedforward=2048, 
            batch_first=True
        )
        self.eeg_encoder = nn.TransformerEncoder(self.eeg_encoder_layer, num_layers=6)
        
        # Projection layer to match BERT's hidden size
        self.fc1 = nn.Linear(in_feature, bert_encoder.config.hidden_size)
    
    def forward(self, eeg_input_batch, text_input_ids, text_attention_mask, labels=None):
        # Process EEG data through additional encoder
        eeg_embeddings = self.eeg_encoder(eeg_input_batch)
        eeg_embeddings = F.relu(self.fc1(eeg_embeddings))
        
        # Process text through BERT encoder
        text_embeddings = self.bert_encoder.embeddings(input_ids=text_input_ids)
        
        # Combine EEG and text embeddings
        combined_embeddings = torch.cat([eeg_embeddings, text_embeddings], dim=1)
        combined_attention_mask = torch.cat([
            torch.ones(eeg_embeddings.size(0), eeg_embeddings.size(1), device=eeg_embeddings.device),
            text_attention_mask
        ], dim=1)
        
        # Pass through BERT encoder
        encoder_outputs = self.bert_encoder.encoder(combined_embeddings, attention_mask=combined_attention_mask)
        
        # Use encoder outputs as input to BART decoder
        decoder_outputs = self.bart_decoder(
            inputs_embeds=encoder_outputs.last_hidden_state,
            attention_mask=combined_attention_mask,
            labels=labels,
            return_dict=True
        )
        return decoder_outputs
