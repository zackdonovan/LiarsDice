include("rl.jl")

println("=" ^ 70)
println("Training Q-Learning Agent for Liar's Dice")
println("=" ^ 70)

# Create game: 2 players, 2 dice each, ones wild, ego is player 1
game = create_game(2, 2, true, 1)

# Create Q-learning model and encoder
println("\nInitializing Q-learning model...")
model = QLearning_for_game(game, 10000, γ=0.99, α=0.1)
encoder = StateEncoder(10000)
println("✓ Model initialized")

# Training parameters
num_episodes = 1000
println("\nTraining for $num_episodes episodes...")
println("(This may take a few minutes...)")

# Train!
rewards = train!(model, encoder, game, num_episodes,
                 epsilon_start=1.0,      # Start with full exploration
                 epsilon_end=0.01,       # End with mostly exploitation
                 epsilon_decay=0.995,    # Slow decay
                 verbose=true,
                 eval_interval=100)

println("\n" * "=" ^ 70)
println("Training Complete!")
println("=" ^ 70)

# Evaluate the trained agent
println("\nEvaluating trained agent against random opponents (100 games)...")
win_rate, avg_reward = evaluate(model, encoder, game, 100, epsilon=0.0)

println("\nResults:")
println("  Win rate: $(round(win_rate*100, digits=1))%")
println("  Average reward: $(round(avg_reward, digits=3))")

if win_rate > 0.5
    println("\n🎉 Success! Agent learned to play better than random!")
else
    println("\n⚠️  Agent may need more training. Try increasing num_episodes.")
end

println("\n" * "=" ^ 70)

