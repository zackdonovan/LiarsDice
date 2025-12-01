include("main.jl")
using Statistics

# IncrementalEstimate
mutable struct IncrementalEstimate
    μ # mean estimate
    α # learning rate function
    m # number of updates
end

# update
function update!(model::IncrementalEstimate, x)
    model.m += 1
    model.μ += model.α(model.m) * (x - model.μ)
    return model
end

# Qlearning struct
mutable struct QLearning
    𝒮 # state space
    𝒜 # action space
    γ # discount
    Q # action value function
    α # learning rate
end

# lookahead function
lookahead(model::QLearning, s, a) = model.Q[s,a]

function update!(model::QLearning, s, a, r, s′)
    γ, Q, α = model.γ, model.Q, model.α
    Q[s,a] += α*(r + γ*maximum(Q[s′,:]) - Q[s,a])
    return model
end

# Construct QLearning model with a large pre-allocated Q-table
function QLearning_for_game(game::Game, max_states::Int=10000; γ=0.99, α=0.1)
    𝒮 = 1:max_states
    𝒜 = 1:length(game.actions)
    Q = zeros(Float64, max_states, length(𝒜))
    return QLearning(𝒮, 𝒜, γ, Q, α)
end

# State encoder: maps Observations to discrete state indices
# Uses a hash-based approach to map to a fixed range of state IDs
mutable struct StateEncoder
    max_states::Int
    StateEncoder(max_states::Int=10000) = new(max_states)
end

# Epsilon-greedy action selection
function select_action(model::QLearning, encoder::StateEncoder, obs::Observation, 
                      game::Game, epsilon::Float64)::Int
    s = encode_state(encoder, obs)
    legal_actions = findall(obs.legal_actions)
    
    if isempty(legal_actions)
        error("No legal actions available!")
    end
    
    # Epsilon-greedy: explore with probability epsilon
    if rand() < epsilon
        # Explore: random legal action
        return rand(legal_actions)
    else
        # Exploit: best legal action according to Q-table
        best_action = legal_actions[1]
        best_q = model.Q[s, best_action]
        
        for a in legal_actions
            q_val = model.Q[s, a]
            if q_val > best_q
                best_q = q_val
                best_action = a
            end
        end
        
        return best_action
    end
end

function encode_state(encoder::StateEncoder, obs::Observation)::Int
    # Create a tuple representation of the state
    # Format: (dice[1], dice[2], ..., dice[6], bid_qty, bid_face, total_dice_left)
    if obs.last_bid === nothing
        bid_qty = 0
        bid_face = 0
    else
        bid_qty = obs.last_bid.qty
        bid_face = obs.last_bid.face
    end
    
    total_dice = sum(obs.dice_left)
    state_tuple = (obs.dice..., bid_qty, bid_face, total_dice)
    
    # Hash the tuple to get a state index in [1, max_states]
    # Use Julia's hash function and mod to map to our range
    state_id = (hash(state_tuple) % encoder.max_states) + 1
    return state_id
end

# Training loop: train Q-learning agent through self-play
function train!(model::QLearning, encoder::StateEncoder, game::Game, 
               num_episodes::Int; 
               epsilon_start::Float64=1.0, 
               epsilon_end::Float64=0.01,
               epsilon_decay::Float64=0.995,
               verbose::Bool=true,
               eval_interval::Int=100)
    
    epsilon = epsilon_start
    episode_rewards = Float64[]
    
    for episode in 1:num_episodes
        state, obs = reset(game)
        total_reward = 0.0
        done = false
        
        # Store transitions for this episode (for potential future use)
        transitions = Tuple{Int, Int, Float64, Int}[]  # (s, a, r, s')
        
        while !done
            if state.pub.turn == game.ego_id
                # Agent's turn: use Q-learning
                s = encode_state(encoder, obs)
                a = select_action(model, encoder, obs, game, epsilon)
                action = game.actions[a]
                
                next_state, next_obs, reward, done = step(game, state, action)
                total_reward += reward
                
                if done
                    # Terminal state: Q(s,a) = r (no future rewards)
                    model.Q[s, a] += model.α * (reward - model.Q[s, a])
                else
                    # standard Q-learning update
                    s′ = encode_state(encoder, next_obs)
                    update!(model, s, a, reward, s′)
                    push!(transitions, (s, a, reward, s′))
                end
                
                state = next_state
                obs = next_obs
            else
                # Opponent's turn: use random policy
                action, _ = random_legal_action(game, obs)
                next_state, next_obs, reward, done = step(game, state, action)
                
                # Agent gets reward even on opponent's turn (if game ends)
                if done
                    total_reward += reward
                end
                
                state = next_state
                obs = next_obs
            end
        end
        
        # decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        push!(episode_rewards, total_reward)
        
        if verbose && episode % eval_interval == 0
            avg_reward = mean(episode_rewards[max(1, end-99):end])
            println("Episode $episode | Avg Reward (last 100): $(round(avg_reward, digits=3)) | " *
                   "Epsilon: $(round(epsilon, digits=3))")
        end
    end
    
    return episode_rewards
end

# Evaluation: test trained agent against random opponents
function evaluate(model::QLearning, encoder::StateEncoder, game::Game, 
                 num_games::Int; epsilon::Float64=0.0)
    wins = 0
    total_rewards = Float64[]
    
    for game_num in 1:num_games
        state, obs = reset(game)
        total_reward = 0.0
        done = false
        
        while !done
            if state.pub.turn == game.ego_id
                # Agent's turn
                a = select_action(model, encoder, obs, game, epsilon)
                action = game.actions[a]
                next_state, next_obs, reward, done = step(game, state, action)
                total_reward += reward
                state = next_state
                obs = next_obs
            else
                # Opponent's turn: random
                action, _ = random_legal_action(game, obs)
                next_state, next_obs, reward, done = step(game, state, action)
                if done
                    total_reward += reward
                end
                state = next_state
                obs = next_obs
            end
        end
        
        # Check if agent won (has dice left when game ends)
        if state !== nothing && state.pub.dice_left[game.ego_id] > 0
            wins += 1
        end
        push!(total_rewards, total_reward)
    end
    
    win_rate = wins / num_games
    avg_reward = mean(total_rewards)
    return win_rate, avg_reward
end
