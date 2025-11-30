include("rl.jl")

# Simple test: train a Q-learning agent
println("=" ^ 60)
println("Testing Q-Learning Training for Liar's Dice")
println("=" ^ 60)

# Create a simple 2-player game
game = create_game(2, 2, true, 1)  # 2 players, 2 dice each, ones wild, ego is player 1

# Create Q-learning model and state encoder
println("\nInitializing Q-learning model...")
model = QLearning_for_game(game, 10000, γ=0.99, α=0.1)
encoder = StateEncoder(10000)

# Test 1: Can we encode states?
println("\nTest 1: State encoding...")
state, obs = reset(game)
s = encode_state(encoder, obs)
println("✓ State encoded to index: $s")

# Test 2: Can we select actions?
println("\nTest 2: Action selection...")
a = select_action(model, encoder, obs, game, 1.0)  # Full exploration (epsilon=1.0)
println("✓ Selected action index: $a (action: $(game.actions[a]))")

# Test 3: Run a few training episodes
println("\nTest 3: Training (5 episodes)...")
rewards = train!(model, encoder, game, 5, 
                 epsilon_start=1.0, 
                 epsilon_end=0.1,
                 epsilon_decay=0.9,
                 verbose=true,
                 eval_interval=1)

println("\n✓ Training completed!")
println("Episode rewards: $rewards")

# Test 4: Evaluate the trained agent
println("\nTest 4: Evaluating trained agent (10 games)...")
win_rate, avg_reward = evaluate(model, encoder, game, 10, epsilon=0.0)
println("✓ Evaluation completed!")
println("Win rate: $(round(win_rate*100, digits=1))%")
println("Average reward: $(round(avg_reward, digits=3))")

println("\n" * "=" ^ 60)
println("All tests passed! ✓")
println("=" ^ 60)

