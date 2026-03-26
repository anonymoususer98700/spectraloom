import os
import subprocess
import json

def run_evaluation():
    base_save_name = 'task1_task2_taskNRv2_finetune_EEGConformer_skipstep1_b32_20_30_5e-05_5e-07_unique_sent_EEG'
    
    # All the suffix variations you tested (Skipping Full Model and uniformSBA as they are done)
    suffixes = [
        "_noMSC",           # w/o MSC
        "_MSCkernels357",   # Alt kernels
        "_noCAB",           # w/o CAB
        "_noLS"             # w/o LS
    ]
    
    for suffix in suffixes:
        save_name = base_save_name + suffix
        checkpoint_path = f"./checkpoints/decoding/best/{save_name}.pt"
        config_path = f"./config/decoding/{save_name}.json"
        
        if not os.path.exists(checkpoint_path):
            print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
            continue
            
        print(f"\n=============================================")
        print(f" Evaluating: {save_name}")
        print(f"=============================================")
        
        eval_cmd = [
            "python", "eval_decoding.py",
            "-checkpoint", checkpoint_path,
            "-conf", config_path,
            "-test_input", "EEG",
            "-train_input", "EEG"
        ]
        
        try:
            score_file = f"./Result/ZuCo_dataset_v1_v2/Score/{save_name}.txt"
            with open(score_file, 'w', encoding='utf-8') as f:
                subprocess.run(eval_cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
            print(f"[INFO] Evaluated {save_name} successfully, metrics saved to {score_file}.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed evaluating {save_name}.")

    print("\n---------------- RESULTS EXTRACTION ----------------\n")
    print(f"{'Condition':<30} | BLEU-1 | BLEU-4 | MeanS BLEU | ROUGE-LF1 | SacreBLEU")
    print("-" * 85)
    
    # We add back the full model to the parser array so it extracts it if the background job finished!
    all_suffixes = [""] + suffixes
    for suffix in all_suffixes:
        save_name = base_save_name + suffix
        score_file = f"./Result/ZuCo_dataset_v1_v2/Score/{save_name}.txt"
        
        if not os.path.exists(score_file):
            continue
            
        with open(score_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        bleu1 = bleu4 = rougeLF = sacre = meanS = "N/A"
        
        for i, line in enumerate(lines):
            if "Without teacher forcing:" in line:
                for j in range(i, min(i+20, len(lines))):
                    if "corpusBLEU" in lines[j] and bleu1 == "N/A":
                        b_str = lines[j].split()
                        try:
                            b_vals = lines[j].split("[")[1].split("]")[0].split(",")
                            bleu1 = f"{float(b_vals[0])*100:.2f}"
                            bleu4 = f"{float(b_vals[3])*100:.2f}"
                        except:
                            pass
                    if "mean_sentence_bleu_score" in lines[j]:
                        try:
                            m_vals = lines[j].split("[")[1].split("]")[0].split(",")
                            meanS = f"{float(m_vals[0])*100:.2f}"
                        except:
                            pass
                    if "'rouge-l': {'" in lines[j]:
                        try:
                            val_str = lines[j].split("'f': ")[1].split('}')[0]
                            rougeLF = f"{float(val_str)*100:.2f}"
                        except:
                            pass
                    if "'score': " in lines[j] and "counts" in lines[j]:
                        try:
                            val_str = lines[j].split("'score': ")[1].split(',')[0]
                            sacre = f"{float(val_str):.2f}"
                        except:
                            pass
                            
        print(f"{suffix if suffix else 'Full Model':<30} | {bleu1:>6} | {bleu4:>6} | {meanS:>10} | {rougeLF:>9} | {sacre:>9}")

if __name__ == '__main__':
    run_evaluation()
