import os
import subprocess

save_name = 'task1_task2_taskNRv2_finetune_EEGConformer_skipstep1_b32_20_30_5e-05_5e-07_unique_sent_EEG_uniformSBA'
checkpoint_path = f"./checkpoints/decoding/best/{save_name}.pt"
config_path = f"./config/decoding/{save_name}.json"
score_file = f"./Result/ZuCo_dataset_v1_v2/Score/{save_name}_SAFE.txt"

eval_cmd = [
    "python", "eval_decoding.py",
    "-checkpoint", checkpoint_path,
    "-conf", config_path,
    "-test_input", "EEG",
    "-train_input", "EEG"
]

print(f"Starting eval for uniformSBA...")
with open(score_file, 'w', encoding='utf-8') as f:
    subprocess.run(eval_cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
print("Finished!")
