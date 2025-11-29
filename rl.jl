include("main.jl")

# Algorithm 17.1: IncrementalEstimate
mutable struct IncrementalEstimate
    μ # mean estimate
    α # learning rate function
    m # number of updates
end

function update!(model::IncrementalEstimate, x)
    model.m += 1
    model.μ += model.α(model.m) * (x - model.μ)
    return model
end

# Algorithm 17.2: QLearning
mutable struct QLearning
    𝒮 # state space
    𝒜 # action space
    γ # discount
    Q # action value function
    α # learning rate
end

lookahead(model::QLearning, s, a) = model.Q[s,a]

function update!(model::QLearning, s, a, r, s′)
    γ, Q, α = model.γ, model.Q, model.α
    Q[s,a] += α*(r + γ*maximum(Q[s′,:]) - Q[s,a])
    return model
end

# construct QLearning model 
function QLearning_for_game(game::Game, nstates::Int; γ=0.99, α=0.1)
    𝒮 = 1:nstates
    𝒜 = 1:length(game.actions)
    Q = zeros(length(𝒮), length(𝒜))
    return QLearning(𝒮, 𝒜, γ, Q, α)
end

#stub
function encode_state(obs::Observation)::Int
    return 1
end
