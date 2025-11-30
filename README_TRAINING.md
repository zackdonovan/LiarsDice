# How to Train and Test Your Q-Learning Agent

## Quick Start

### Option 1: Run the test script
```bash
julia test_training.jl
```
This runs a quick test with 5 training episodes.

### Option 2: Train in Julia REPL
```julia
include("rl.jl")

# Create game
game = create_game(2, 2, true, 1)  # 2 players, 2 dice each, ones wild, ego is player 1

# Create Q-learning model
model = QLearning_for_game(game, 10000, γ=0.99, α=0.1)
encoder = StateEncoder(10000)

# Train for 1000 episodes
rewards = train!(model, encoder, game, 1000, 
                 epsilon_start=1.0, 
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 verbose=true,
                 eval_interval=100)

# Evaluate trained agent
win_rate, avg_reward = evaluate(model, encoder, game, 100, epsilon=0.0)
println("Win rate: $(round(win_rate*100, digits=1))%")
println("Average reward: $(round(avg_reward, digits=3))")
```

## Parameters Explained

- **`num_episodes`**: How many games to train on (more = better learning, but slower)
- **`epsilon_start`**: Initial exploration rate (1.0 = 100% random, 0.0 = 100% greedy)
- **`epsilon_end`**: Final exploration rate (keep some exploration)
- **`epsilon_decay`**: How fast to reduce exploration (0.995 = slow decay)
- **`γ` (gamma)**: Discount factor (0.99 = care about future rewards)
- **`α` (alpha)**: Learning rate (0.1 = moderate learning speed)

## What to Expect

- **Early training**: Agent plays randomly, win rate ~50%
- **After training**: Win rate should improve (ideally >50% vs random opponents)
- **Rewards**: Should increase over time as agent learns

## Tips

- Start with small number of episodes (100-1000) to test
- Increase to 10,000+ episodes for serious training
- Monitor the average reward - it should trend upward
- Win rate > 50% means the agent is learning!

