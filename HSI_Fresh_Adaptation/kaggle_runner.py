"""
Kaggle Runner for HSI WST Pipeline
===================================
Designed for Kaggle's 12-hour time limit.
Run ONE preprocessing combo at a time across multiple sessions.

Usage (run each in a separate Kaggle session):
  Session 1: python kaggle_runner.py --dataset indian_pines --step 1  # SG+SVN
  Session 2: python kaggle_runner.py --dataset indian_pines --step 2  # SG+MSC
  Session 3: python kaggle_runner.py --dataset indian_pines --step 3  # SG1+SVN
  Session 4: python kaggle_runner.py --dataset indian_pines --step 4  # SG1+MSC

Each step saves checkpoints. If interrupted, restart the same step to resume.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

# Define the 4 preprocessing combinations
PREPROCESSING_STEPS = {
    1: ["SG", "SVN"],   # Step 1: SG + SVN (estimated ~3-4 hours)
    2: ["SG", "MSC"],   # Step 2: SG + MSC (estimated ~3-4 hours)
    3: ["SG1", "SVN"],  # Step 3: SG1 + SVN (estimated ~3-4 hours)
    4: ["SG1", "MSC"],  # Step 4: SG1 + MSC (estimated ~3-4 hours)
}

def get_checkpoint_file(dataset, step):
    """Get the checkpoint file path for a given step."""
    return f"checkpoint_{dataset}_step{step}.json"

def save_checkpoint(dataset, step, status, details=""):
    """Save checkpoint after completing a step."""
    checkpoint = {
        "dataset": dataset,
        "step": step,
        "preprocessing": PREPROCESSING_STEPS[step],
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "details": details
    }
    with open(get_checkpoint_file(dataset, step), 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"Checkpoint saved: Step {step} - {status}")

def load_checkpoint(dataset, step):
    """Load checkpoint if exists."""
    checkpoint_file = get_checkpoint_file(dataset, step)
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def print_status(dataset):
    """Print status of all steps for a dataset."""
    print(f"\n{'='*60}")
    print(f"HSI WST Pipeline Status - {dataset.upper()}")
    print(f"{'='*60}")
    
    for step, preprocessing in PREPROCESSING_STEPS.items():
        checkpoint = load_checkpoint(dataset, step)
        if checkpoint:
            status = checkpoint.get("status", "Unknown")
            timestamp = checkpoint.get("timestamp", "")
            print(f"  Step {step}: {preprocessing[0]}+{preprocessing[1]:3} - {status:12} ({timestamp[:19]})")
        else:
            print(f"  Step {step}: {preprocessing[0]}+{preprocessing[1]:3} - NOT STARTED")
    print(f"{'='*60}\n")

def run_step(dataset, step, resume=False):
    """Run a single preprocessing step."""
    if step not in PREPROCESSING_STEPS:
        print(f"Invalid step: {step}. Choose 1-4.")
        return
    
    preprocessing = PREPROCESSING_STEPS[step]
    pipeline_name = f"10_{preprocessing[0]}_{preprocessing[1]}"
    
    print(f"\n{'='*60}")
    print(f"STEP {step}: Running {preprocessing[0]} + {preprocessing[1]}")
    print(f"Dataset: {dataset}")
    print(f"Pipeline: {pipeline_name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint(dataset, step)
    if checkpoint and checkpoint.get("status") == "COMPLETED":
        print(f"Step {step} already completed. Use --force to re-run.")
        return
    
    # Save "in progress" checkpoint
    save_checkpoint(dataset, step, "IN_PROGRESS", "Started execution")
    
    start_time = time.time()
    
    try:
        # Import and run the pipeline
        from WST_script import run_pipeline
        
        n_components_max = 10
        run_pipeline(n_components_max, preprocessing, dataset, plots_only=False, skip_to=None)
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_str = f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"
        
        # Save completed checkpoint
        save_checkpoint(dataset, step, "COMPLETED", f"Finished in {elapsed_str}")
        
        print(f"\n{'='*60}")
        print(f"STEP {step} COMPLETED in {elapsed_str}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        # Save error checkpoint
        save_checkpoint(dataset, step, "ERROR", str(e))
        print(f"\nERROR in Step {step}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description='Kaggle Runner for HSI WST Pipeline (12-hour time limit support)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kaggle_runner.py --dataset indian_pines --status     # Check progress
  python kaggle_runner.py --dataset indian_pines --step 1     # Run SG+SVN
  python kaggle_runner.py --dataset indian_pines --step 2     # Run SG+MSC
  python kaggle_runner.py --dataset salinas --step 1          # Run Salinas SG+SVN
        """
    )
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['indian_pines', 'salinas'],
                        help='Dataset to process')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4],
                        help='Which preprocessing step to run (1-4)')
    parser.add_argument('--status', action='store_true',
                        help='Show status of all steps')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if step completed')
    
    args = parser.parse_args()
    
    if args.status:
        print_status(args.dataset)
        return
    
    if args.step is None:
        print("Please specify --step (1-4) or --status")
        print_status(args.dataset)
        return
    
    # Check if already completed (unless force)
    checkpoint = load_checkpoint(args.dataset, args.step)
    if checkpoint and checkpoint.get("status") == "COMPLETED" and not args.force:
        print(f"Step {args.step} already completed on {checkpoint.get('timestamp', 'unknown')}")
        print("Use --force to re-run")
        print_status(args.dataset)
        return
    
    run_step(args.dataset, args.step)
    print_status(args.dataset)

if __name__ == "__main__":
    main()
