include("main.jl")
using POMDPs
using POMDPModels
using POMDPSimulators
using POMDPPolicies
using POMDPTools: Deterministic

"""
Liar's Dice as a POMDP

State: FullState (includes all players' hidden dice)
Action: Action (BidAction or LiarAction)  
Observation: Observation (what the agent sees - own dice, last bid, dice left)
"""
mutable struct LiarsDicePOMDP <: POMDP{FullState, Action, Observation}
    num_players::Int
    dice_per_player::Int
    ones_wild::Bool
    ego_id::Int
    actions::Vector{Action}
    discount::Float64
end

function LiarsDicePOMDP(num_players::Int=2, dice_per_player::Int=2, 
                        ones_wild::Bool=true, ego_id::Int=1, discount::Float64=0.99)
    max_bid = num_players * dice_per_player
    actions = all_actions(max_bid)
    return LiarsDicePOMDP(num_players, dice_per_player, ones_wild, ego_id, actions, discount)
end

# POMDPs.jl interface functions

# State space
POMDPs.states(pomdp::LiarsDicePOMDP) = error("State space too large to enumerate")
POMDPs.stateindex(pomdp::LiarsDicePOMDP, s::FullState) = error("State indexing not implemented (state space too large)")

# Action space
POMDPs.actions(pomdp::LiarsDicePOMDP) = pomdp.actions
POMDPs.actionindex(pomdp::LiarsDicePOMDP, a::Action) = findfirst(==(a), pomdp.actions)

# Observation space
POMDPs.observations(pomdp::LiarsDicePOMDP) = error("Observation space too large to enumerate")
POMDPs.obsindex(pomdp::LiarsDicePOMDP, o::Observation) = error("Observation indexing not implemented")

# Discount factor
POMDPs.discount(pomdp::LiarsDicePOMDP) = pomdp.discount

# Initial state distribution
# Returns a function that samples from the initial state distribution
function POMDPs.initialstate(pomdp::LiarsDicePOMDP)
    return function()
        game = Game(pomdp.num_players, pomdp.dice_per_player, pomdp.ones_wild, 
                   pomdp.ego_id, pomdp.actions)
        state, _ = reset(game)
        return state
    end
end

# Transition function: T(s'|s,a)
# Returns a function that samples from the transition distribution
# Note: This handles the agent's action, but opponent actions happen between agent turns
function POMDPs.transition(pomdp::LiarsDicePOMDP, s::FullState, a::Action)
    # Create a game instance for stepping
    game = Game(pomdp.num_players, pomdp.dice_per_player, pomdp.ones_wild,
               pomdp.ego_id, pomdp.actions)
    
    # If it's not the agent's turn, this shouldn't be called
    if s.pub.turn != pomdp.ego_id
        error("Transition called when it's not the agent's turn")
    end
    
    # Return a sampling function
    return function()
        # Take the agent's action
        next_state, _, _, done = step(game, s, a)
        
        # If game ended, return terminal state
        if done || next_state === nothing
            return s  # Stay in terminal state
        end
        
        # Now handle opponent actions until it's agent's turn again
        # This makes the transition stochastic (opponents act randomly)
        current_state = next_state
        while current_state.pub.turn != pomdp.ego_id && !isterminal(pomdp, current_state)
            # Opponent's turn - sample random legal action
            obs = observe(game, current_state)
            opp_action, _ = random_legal_action(game, obs)
            current_state, _, _, done = step(game, current_state, opp_action)
            
            if done || current_state === nothing
                return s  # Terminal
            end
        end
        
        return current_state
    end
end

# Observation function: O(o|s',a)
function POMDPs.observation(pomdp::LiarsDicePOMDP, s::FullState, a::Action, sp::FullState)
    # Return the observation the agent would see in state sp
    game = Game(pomdp.num_players, pomdp.dice_per_player, pomdp.ones_wild,
               pomdp.ego_id, pomdp.actions)
    obs = observe(game, sp)
    return Deterministic(obs)
end

# Reward function: R(s,a) or R(s,a,s')
# POMDPs.jl supports both signatures
function POMDPs.reward(pomdp::LiarsDicePOMDP, s::FullState, a::Action, sp::Union{FullState, Nothing}=nothing)
    game = Game(pomdp.num_players, pomdp.dice_per_player, pomdp.ones_wild,
               pomdp.ego_id, pomdp.actions)
    
    # Calculate reward for taking action a in state s
    # The reward comes from the step function
    _, _, r, _ = step(game, s, a)
    return r
end

# Check if state is terminal
function POMDPs.isterminal(pomdp::LiarsDicePOMDP, s::FullState)
    # Game is terminal if any player has 0 dice
    return any(d == 0 for d in s.pub.dice_left)
end

# Convert from existing Game/FullState to POMDP
function create_pomdp_from_game(game::Game; discount::Float64=0.99)
    return LiarsDicePOMDP(game.num_players, game.dice_per_player, 
                         game.ones_wild, game.ego_id, game.actions, discount)
end

