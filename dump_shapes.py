import torch
d = torch.load('./checkpoints/decoding/best/task1v1_task2v1_task3v1_finetune_EEGConformer_skipstep1_b32_20_30_0.0001_2e-05_unique_sent_EEG.pt', map_location='cpu', weights_only=True)
for k, v in d.items():
    if "multi_scale_conv" in k:
        print(f"{k}: {v.shape}")
