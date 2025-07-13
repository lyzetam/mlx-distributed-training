#!/usr/bin/env python3
"""
Train Gemma model on Home Assistant voice commands.
"""

import argparse
import yaml
import time
import json
from pathlib import Path

from utils.model_cache import ModelCacheManager
from utils.logging import setup_logging
from utils.training_core import train_model
from utils.data_preparation import validate_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Gemma on Home Assistant data")
    parser.add_argument("--config", type=str, 
                       default="config/homeassistant_training_config.yaml",
                       help="Path to training configuration")
    parser.add_argument("--distributed", action="store_true",
                       help="Enable distributed training")
    parser.add_argument("--test", action="store_true",
                       help="Run a quick test with fewer iterations")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging(
        log_level=config['logging']['level'],
        log_file=config['logging'].get('log_file')
    )
    
    print("=" * 60)
    print("Home Assistant Voice Training")
    print("=" * 60)
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']['path']}")
    print(f"Training type: {config['training']['type']}")
    
    # Validate dataset
    if not validate_dataset(config['dataset']['output_dir']):
        print("❌ Dataset validation failed!")
        print("Please run: python3 convert_homeassistant_data.py first")
        return
    
    # Adjust for test run
    if args.test:
        print("\n🧪 Test mode: Reducing iterations to 10")
        config['hyperparameters']['num_iterations'] = 10
        config['hyperparameters']['steps_per_eval'] = 5
        config['checkpointing']['save_every'] = 10
    
    # Prepare training configuration
    training_config = {
        "fine_tune_type": config['training']['type'],
        "num_layers": config['training']['lora']['num_layers'],
        "lora_parameters": config['training']['lora'],
        "batch_size": config['hyperparameters']['batch_size'],
        "iters": config['hyperparameters']['num_iterations'],
        "learning_rate": config['hyperparameters']['learning_rate'],
        "val_batches": config['hyperparameters']['val_batches'],
        "steps_per_report": config['hyperparameters']['steps_per_report'],
        "steps_per_eval": config['hyperparameters']['steps_per_eval'],
        "save_every": config['checkpointing']['save_every'],
        "adapter_path": config['checkpointing']['output_dir'],
        "max_seq_length": config['dataset']['max_seq_length'],
        "grad_checkpoint": config['optimization']['gradient_checkpointing']
    }
    
    print("\nTraining Configuration:")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Iterations: {training_config['iters']}")
    print(f"  LoRA rank: {training_config['lora_parameters']['rank']}")
    print(f"  Gradient checkpointing: {training_config['grad_checkpoint']}")
    
    # Warning for 9B model
    print("\n⚠️  Note: Gemma 9B requires significant memory (~20GB)")
    print("   Consider using distributed training or the 4-bit version")
    print("   if you encounter memory issues.")
    
    # Skip input prompt in distributed mode
    if not (args.distributed or config['distributed']['enabled']):
        input("\nPress Enter to start training...")
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run training
    results = train_model(
        model_name=config['model']['name'],
        data_path=config['dataset']['output_dir'],
        training_config=training_config,
        distributed=args.distributed or config['distributed']['enabled']
    )
    
    training_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 60)
    print(f"Training completed in {training_time:.2f} seconds")
    
    if results.get('success'):
        print("✅ Training successful!")
        print(f"Adapter saved to: {results.get('adapter_path')}")
        
        # Save training summary
        summary = {
            "model": config['model']['name'],
            "dataset": config['dataset']['path'],
            "training_time": training_time,
            "final_train_loss": results.get('final_train_loss'),
            "final_val_loss": results.get('final_val_loss'),
            "adapter_path": str(results.get('adapter_path')),
            "config": config
        }
        
        summary_path = Path(results.get('adapter_path')) / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining summary saved to: {summary_path}")
        
        # Print example usage
        print("\n" + "=" * 60)
        print("To use your trained model:")
        print(f"mlx_lm.generate --model {config['model']['name']} \\")
        print(f"  --adapter-path {results.get('adapter_path')} \\")
        print("  --max-tokens 150 \\")
        print("  --prompt \"turn on the living room lights\"")
        
        # Test with evaluation prompts
        if config.get('evaluation', {}).get('test_prompts'):
            print("\n" + "=" * 60)
            print("Test your model with these prompts:")
            for prompt in config['evaluation']['test_prompts'][:3]:
                print(f"  - {prompt}")
    else:
        print(f"❌ Training failed: {results.get('error')}")
        print("\nTroubleshooting:")
        print("1. Check memory usage - Gemma 9B needs ~20GB")
        print("2. Try the 4-bit version: mlx-community/gemma-2-9b-it-4bit")
        print("3. Reduce batch_size to 1 in the config")
        print("4. Enable distributed training with --distributed")


if __name__ == "__main__":
    main()
