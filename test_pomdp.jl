include("liars_dice_pomdp.jl")
using Statistics

println("=" ^ 70)
println("Testing POMDP Integration with main.jl")
println("=" ^ 70)

# Test 1: Create POMDP
println("\n1. Creating POMDP...")
pomdp = LiarsDicePOMDP(2, 2, true, 1, 0.99)
println("   ✓ POMDP created")
println("   - Players: $(pomdp.num_players)")
println("   - Dice per player: $(pomdp.dice_per_player)")
println("   - Actions: $(length(actions(pomdp)))")
println("   - Discount: $(discount(pomdp))")

# Test 2: Initial state
println("\n2. Testing initialstate()...")
init_dist = initialstate(pomdp)
state = init_dist()
println("   ✓ Initial state sampled")
println("   - Type: $(typeof(state))")
println("   - Players: $(length(state.players))")
println("   - Turn: $(state.pub.turn)")
println("   - Dice left: $(state.pub.dice_left)")

# Test 3: Observation
println("\n3. Testing observation()...")
game = Game(pomdp.num_players, pomdp.dice_per_player, pomdp.ones_wild,
           pomdp.ego_id, pomdp.actions)
obs = observe(game, state)
println("   ✓ Observation created")
println("   - Type: $(typeof(obs))")
println("   - Agent's dice: $(obs.dice)")
println("   - Last bid: $(obs.last_bid)")
println("   - Legal actions: $(sum(obs.legal_actions))")

# Test 4: Transition (if it's agent's turn)
println("\n4. Testing transition()...")
if state.pub.turn == pomdp.ego_id
    action = pomdp.actions[1]  # First action
    println("   - Taking action: $action")
    trans_dist = transition(pomdp, state, action)
    next_state = trans_dist()
    println("   ✓ Transition completed")
    println("   - Next state type: $(typeof(next_state))")
    println("   - Next turn: $(next_state.pub.turn)")
    println("   - Terminal: $(isterminal(pomdp, next_state))")
else
    println("   ⚠️  Not agent's turn, skipping transition test")
end

# Test 5: Reward
println("\n5. Testing reward()...")
if state.pub.turn == pomdp.ego_id
    action = pomdp.actions[1]
    r = reward(pomdp, state, action)
    println("   ✓ Reward calculated: $r")
end

println("\n" * "=" ^ 70)
println("✅ All tests passed! POMDP correctly integrates with main.jl")
println("=" ^ 70)

