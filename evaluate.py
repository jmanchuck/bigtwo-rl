#!/usr/bin/env python3
"""Evaluate trained agent against baselines."""

import numpy as np
from stable_baselines3 import PPO
from rl_wrapper import BigTwoRLWrapper
from agents import RandomAgent, GreedyAgent

def evaluate_agent(model_path, num_games=100):
    """Evaluate agent against baselines."""
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    env = BigTwoRLWrapper()
    random_agent = RandomAgent("Random")
    greedy_agent = GreedyAgent("Greedy")
    greedy_agent.set_env_reference(env)
    
    results = {"vs_random": 0, "vs_greedy": 0}
    
    # Test vs random policy
    print("Evaluating vs random policy...")
    wins = 0
    for game in range(num_games):
        obs, _ = env.reset()
        done = False
        
        while not done:
            if env.env.current_player == 0:  # Agent's turn
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:  # Random opponent
                action_mask = env.get_action_mask()
                action = random_agent.get_action(obs, action_mask)
            
            obs, reward, done, _, _ = env.step(action)
            
            if done and reward > 0:  # Agent won
                wins += 1
                break
    
    results["vs_random"] = wins / num_games
    print(f"Win rate vs random: {results['vs_random']:.2%}")
    
    # Test vs greedy policy
    print("Evaluating vs greedy policy...")
    wins = 0
    for game in range(num_games):
        obs, _ = env.reset()
        done = False
        
        while not done:
            if env.env.current_player == 0:  # Agent's turn
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:  # Greedy opponent
                action_mask = env.get_action_mask()
                action = greedy_agent.get_action(obs, action_mask)
            
            obs, reward, done, _, _ = env.step(action)
            
            if done and reward > 0:  # Agent won
                wins += 1
                break
    
    results["vs_greedy"] = wins / num_games
    print(f"Win rate vs greedy: {results['vs_greedy']:.2%}")
    
    return results

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./models/best_model"
    results = evaluate_agent(model_path)