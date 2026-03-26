import os
import subprocess
import sys

def main():
    # Base command matching your previous training runs
    base_cmd = [
        "python", "train_decoding.py", 
        "-m", "EEGConformer", 
        "-t", "task1_task2_taskNRv2", 
        "-ne1", "20", 
        "-ne2", "30", 
        "-lr1", "0.00005", 
        "-lr2", "0.0000005", 
        "-b", "32", 
        "-s", "./checkpoints/decoding", 
        "--train_input", "EEG",
        "--skip_step_one"
    ]

    # The ablation configurations we want to test
    ablation_flags = [
        [],                                 # Full Model
        ["--ablate_sba", "uniform"],        # w/o SBA (Uniform Band Weighting)
        ["--ablate_multiscale"],            # w/o Multi-Scale Conv1D (Linear Projection instead)
        ["--conv_kernels", "3,5,7"],        # Alternative Conv Kernels
        ["--ablate_cab"],                   # w/o Cross-Attention Bridge (Linear Projection instead)
        ["--ablate_label_smoothing"],       # w/o Label Smoothing
        ["--feature_level", "sentence"]     # Test with Sentence-level EEG Features
    ]

    print("=========================================================")
    print("Starting Sequential Ablation Study for EEGConformer")
    print(f"Total Runs: {len(ablation_flags)}")
    print("=========================================================\n")

    for i, flags in enumerate(ablation_flags, 1):
        print(f"---------------------------------------------------------")
        print(f"[{i}/{len(ablation_flags)}] Running Setup with flags: {' '.join(flags) if flags else 'Full Model'}")
        print(f"---------------------------------------------------------")
        
        # Construct the specific command for this run
        cmd = base_cmd + flags
        
        try:
            # Run the command and wait for it to finish
            # stdout/stderr will stream directly to the terminal
            result = subprocess.run(cmd, check=True)
            print(f"\n[INFO] Successfully completed run with {' '.join(flags) if flags else 'Full Model'} training.\n")
            
            # Now run evaluation
            # Reconstruct the expected save name
            base_save_name = 'task1_task2_taskNRv2_finetune_EEGConformer_skipstep1_b32_20_30_5e-05_5e-07_unique_sent_EEG'
            save_name = base_save_name
            
            if "--ablate_sba" in flags and "uniform" in flags:
                save_name += '_uniformSBA'
            if "--ablate_multiscale" in flags:
                save_name += '_noMSC'
            if "--conv_kernels" in flags and "3,5,7" in flags:
                save_name += '_MSCkernels357'
            if "--ablate_cab" in flags:
                save_name += '_noCAB'
            if "--ablate_label_smoothing" in flags:
                save_name += '_noLS'
                
            checkpoint_path = f"./checkpoints/decoding/best/{save_name}.pt"
            config_path = f"./config/decoding/{save_name}.json"
            
            print(f"--- Running Evaluation for {' '.join(flags) if flags else 'Full Model'} ---")
            eval_cmd = [
                "python", "eval_decoding.py",
                "-checkpoint", checkpoint_path,
                "-conf", config_path,
                "-test_input", "EEG",
                "-train_input", "EEG"
            ]
            subprocess.run(eval_cmd, check=True)
            print(f"[INFO] Successfully evaluated {' '.join(flags) if flags else 'Full Model'}\n")
            
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Run failed with {' '.join(flags) if flags else 'Full Model'}. Exit code: {e.returncode}")
            print("Stopping execution of remaining ablations.")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n[WARNING] Process interrupted by user during {' '.join(flags) if flags else 'Full Model'}.")
            print("Stopping execution of remaining ablations.")
            sys.exit(1)

    print("=========================================================")
    print("All Ablation Runs Completed Successfully!")
    print("=========================================================")

if __name__ == "__main__":
    main()
