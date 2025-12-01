include("liars_dice_pomdp.jl")
include("main.jl")
using POMDPs
using POMDPSimulators
using POMDPPolicies
using Random

"""
Simple random policy for POMDP - for testing
"""
struct RandomPOMDPPolicy <: Policy
    pomdp::LiarsDicePOMDP
end

function POMDPs.action(policy::RandomPOMDPPolicy, b)
    # b is a belief (or in our case, an observation)
    # For now, treat b as an Observation and pick random legal action
    if b isa Observation
        legal_actions = findall(b.legal_actions)
        if isempty(legal_actions)
            error("No legal actions!")
        end
        return policy.pomdp.actions[rand(legal_actions)]
    else
        error("Unexpected belief type: $(typeof(b))")
    end
end

"""
Train using POMDP structure with Q-learning on observations
This bridges the gap: use POMDP structure but Q-learning solver
"""
function train_pomdp_qlearning(pomdp::LiarsDicePOMDP, num_episodes::Int;
                                alpha::Float64=0.1, gamma::Float64=0.99,
                                epsilon_start::Float64=1.0, epsilon_end::Float64=0.01,
                                epsilon_decay::Float64=0.995, verbose::Bool=true)
    
    # Q-table: maps (observation_hash, action_index) -> Q-value
    Q = Dict{Tuple{UInt64, Int}, Float64}()
    epsilon = epsilon_start
    episode_rewards = Float64[]
    
    # Helper to hash observation
    function obs_hash(obs::Observation)::UInt64
        if obs.last_bid === nothing
            bid_qty = 0
            bid_face = 0
        else
            bid_qty = obs.last_bid.qty
            bid_face = obs.last_bid.face
        end
        total_dice = sum(obs.dice_left)
        state_tuple = (obs.dice..., bid_qty, bid_face, total_dice)
        return hash(state_tuple)  # hash() returns UInt64
    end
    
    # Get Q-value
    function get_q(obs::Observation, a_idx::Int)::Float64
        key = (obs_hash(obs), a_idx)
        return get(Q, key, 0.0)
    end
    
    # Set Q-value
    function set_q(obs::Observation, a_idx::Int, value::Float64)
        key = (obs_hash(obs), a_idx)
        Q[key] = value
    end
    
    # Epsilon-greedy action selection
    function select_action(obs::Observation, epsilon::Float64)::Int
        legal_actions = findall(obs.legal_actions)
        if isempty(legal_actions)
            error("No legal actions!")
        end
        
        if rand() < epsilon
            return rand(legal_actions)
        else
            best_action = legal_actions[1]
            best_q = get_q(obs, best_action)
            for a in legal_actions
                q_val = get_q(obs, a)
                if q_val > best_q
                    best_q = q_val
                    best_action = a
                end
            end
            return best_action
        end
    end
    
    for episode in 1:num_episodes
        # Sample initial state
        init_dist = initialstate(pomdp)
        state = init_dist()
        game = Game(pomdp.num_players, pomdp.dice_per_player, pomdp.ones_wild,
                   pomdp.ego_id, pomdp.actions)
        obs = observe(game, state)
        
        total_reward = 0.0
        done = false
        
        while !done && !isterminal(pomdp, state)
            # Only act when it's the agent's turn
            # The transition function handles opponent actions
            if state.pub.turn != pomdp.ego_id
                # Skip to agent's turn (transition handles this)
                break
            end
            
            # Agent's turn
            a_idx = select_action(obs, epsilon)
            action = pomdp.actions[a_idx]
            
            # Take action - transition function handles opponent actions too
            trans_dist = transition(pomdp, state, action)
            next_state = trans_dist()
            reward_val = reward(pomdp, state, action)
            total_reward += reward_val
            
            if isterminal(pomdp, next_state) || next_state === nothing
                # Terminal: Q(s,a) = r
                set_q(obs, a_idx, get_q(obs, a_idx) + alpha * (reward_val - get_q(obs, a_idx)))
                done = true
            else
                # Get next observation (when it's agent's turn again)
                if next_state.pub.turn == pomdp.ego_id
                    obs_dist = observation(pomdp, state, action, next_state)
                    next_obs = rand(obs_dist)
                    
                    # Q-learning update
                    if !isempty(findall(next_obs.legal_actions))
                        max_next_q = maximum([get_q(next_obs, a) for a in findall(next_obs.legal_actions)])
                        target = reward_val + gamma * max_next_q
                        current_q = get_q(obs, a_idx)
                        set_q(obs, a_idx, current_q + alpha * (target - current_q))
                    end
                    
                    state = next_state
                    obs = next_obs
                else
                    # Next state is opponent's turn - transition should have handled this
                    # But if we're here, we need to wait
                    state = next_state
                    obs = observe(game, state)
                end
            end
        end
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        push!(episode_rewards, total_reward)
        
        if verbose && episode % 100 == 0
            avg_reward = mean(episode_rewards[max(1, end-99):end])
            println("Episode $episode | Avg Reward: $(round(avg_reward, digits=3)) | Epsilon: $(round(epsilon, digits=3))")
        end
    end
    
    return episode_rewards, Q
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 70)
    println("Training POMDP-based Q-Learning Agent")
    println("=" ^ 70)
    
    pomdp = LiarsDicePOMDP(2, 2, true, 1, 0.99)
    println("\nPOMDP created:")
    println("  Players: $(pomdp.num_players)")
    println("  Dice per player: $(pomdp.dice_per_player)")
    println("  Actions: $(length(actions(pomdp)))")
    println("  Discount: $(discount(pomdp))")
    
    println("\nTraining for 1000 episodes...")
    rewards, Q = train_pomdp_qlearning(pomdp, 1000, verbose=true)
    
    println("\nTraining complete!")
    println("Final average reward: $(round(mean(rewards[end-99:end]), digits=3))")
    println("Q-table size: $(length(Q)) entries")
end

