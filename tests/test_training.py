#!/usr/bin/env python3
"""Quick test of training pipeline."""

from train import train_agent

if __name__ == "__main__":
    print("Testing training pipeline with short run...")
    model = train_agent(total_timesteps=1000, eval_freq=500)
    print("✓ Training pipeline works!")