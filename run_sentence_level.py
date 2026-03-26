import os
import subprocess

def run_sentence_level():
    base_save_name = "task1_task2_taskNRv2_finetune_EEGConformer_skipstep1_b32_20_30_5e-05_5e-07_unique_sent_EEG"
    save_name = base_save_name + "_sentence_level"
    
    train_cmd = [
        "python", "train_decoding.py",
        "-model_name", "EEGConformer",
        "-task_name", "task1_task2_taskNRv2",
        "-dataset", "ZuCo",
        "-eeg_type", "GD",
        "-level", "sentence",
        "--batch_size", "32",
        "-learning_rate", "0.00005",
        "-pretrained_lr_ratio", "0.01",
        "-epochs", "29",
        "-skip_step_one",
        "-save_name", save_name,
        "-feature_level", "sentence"
    ]
    
    print("=============================================")
    print(f" Starting Sentence-Level Training: {save_name}")
    print("=============================================")
    subprocess.run(train_cmd, check=True)

if __name__ == '__main__':
    run_sentence_level()
