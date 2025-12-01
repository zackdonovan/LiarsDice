include("pomdp_training.jl")
using Statistics

println("=" ^ 70)
println("Training POMDP-based Q-Learning Agent for Liar's Dice")
println("=" ^ 70)

# Create POMDP
pomdp = LiarsDicePOMDP(2, 2, true, 1, 0.99)
println("\nPOMDP Configuration:")
println("  Players: $(pomdp.num_players)")
println("  Dice per player: $(pomdp.dice_per_player)")
println("  Ones wild: $(pomdp.ones_wild)")
println("  Ego (agent) ID: $(pomdp.ego_id)")
println("  Total actions: $(length(actions(pomdp)))")
println("  Discount factor: $(discount(pomdp))")

# Training parameters
num_episodes = 1000
println("\nTraining Parameters:")
println("  Episodes: $num_episodes")
println("  Learning rate (alpha): 0.1")
println("  Discount (gamma): 0.99")
println("  Epsilon start: 1.0 (full exploration)")
println("  Epsilon end: 0.01 (mostly exploitation)")
println("  Epsilon decay: 0.995")

println("\n" * "=" ^ 70)
println("Starting Training...")
println("=" ^ 70)
println("(This may take a few minutes)\n")

# Train!
rewards, Q = train_pomdp_qlearning(pomdp, num_episodes,
                                   alpha=0.1,
                                   gamma=0.99,
                                   epsilon_start=1.0,
                                   epsilon_end=0.01,
                                   epsilon_decay=0.995,
                                   verbose=true)

println("\n" * "=" ^ 70)
println("Training Complete!")
println("=" ^ 70)

# Results
println("\nResults:")
println("  Total episodes: $num_episodes")
println("  Final 100 episodes average reward: $(round(mean(rewards[end-99:end]), digits=3))")
println("  Q-table entries: $(length(Q))")
println("  Average reward (all episodes): $(round(mean(rewards), digits=3))")
println("  Best episode reward: $(round(maximum(rewards), digits=3))")
println("  Worst episode reward: $(round(minimum(rewards), digits=3))")

if mean(rewards[end-99:end]) > 0
    println("\n🎉 Agent appears to be learning! (positive average reward)")
else
    println("\n⚠️  Agent may need more training or parameter tuning")
end

println("\n" * "=" ^ 70)

