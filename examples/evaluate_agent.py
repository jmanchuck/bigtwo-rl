#!/usr/bin/env python3
"""Simple example: Evaluate a trained Big Two agent."""

import sys
from bigtwo_rl.evaluation import Evaluator

def main():
    """Evaluate a trained agent against baselines."""
    if len(sys.argv) < 2:
        print("Usage: python examples/evaluate_agent.py <model_path>")
        print("Example: python examples/evaluate_agent.py ./models/example_agent/best_model")
        return
        
    model_path = sys.argv[1]
    num_games = 100 if len(sys.argv) < 3 else int(sys.argv[2])
    
    print(f"Evaluating model: {model_path}")
    print(f"Running {num_games} games against each baseline...")
    
    # Create evaluator
    evaluator = Evaluator(num_games=num_games)
    
    # Evaluate the model
    results = evaluator.evaluate_model(model_path)
    
    # Print results
    print("\n" + results["tournament_summary"])

if __name__ == "__main__":
    main()