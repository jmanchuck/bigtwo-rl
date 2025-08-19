#!/usr/bin/env python3
"""Quick test of training pipeline."""

from bigtwo_rl.training import Trainer

if __name__ == "__main__":
    print("Testing training pipeline with short run...")
    trainer = Trainer()
    model, model_dir = trainer.train(total_timesteps=1000)
    print("✓ Training pipeline works!")
