using Lux, Random, ForwardDiff, OptimizationOptimisers, ComponentArrays
using PositPR
using Plots, Printf

T = Posit{8,2}
rng = MersenneTwister(1234)
epochs = 200

# Rede
chain = Chain(
    Dense(1, 5, tanh; init_weight=randnP8_2, init_bias=zerosP8_2),
    Dense(5, 5, tanh; init_weight=randnP8_2, init_bias=zerosP8_2),
    Dense(5, 1; init_weight=randnP8_2, init_bias=zerosP8_2)
)

θ, st = Lux.setup(rng, chain)
θ = ComponentArray(θ)

global N = 50
global ONE_T = one(T)
global ZERO_T = zero(T)
global TWO_PI = T(2.0) * pi_ramanujan(T)

function make_loss(T, chain, st)
    f(u, t) = cos(t * TWO_PI)

    function trial_solution(input, θ)
        out, _ = chain(input, θ, st)
        t = input[1]
        return t * only(out)
    end

    function loss(θ)
        ts = range(ZERO_T, ONE_T; length=N)
        s = ZERO_T
        for t in ts
            g(t_val) = trial_solution([t_val], θ)
            û = g(t)
            dû = ForwardDiff.derivative(g, t)
            r = dû - f(û, t)
            s += r^2
        end
        return s / length(ts)
    end

    return loss, trial_solution
end

loss, trial = make_loss(T, chain, st)
melhor_loss = typemax(T)
melhor_θ = deepcopy(θ)
melhor_epoch = 0

η = T(0.09)

for epoch in 1:epochs
    global melhor_loss
    global melhor_θ
    global melhor_epoch

    grad = ForwardDiff.gradient(loss, θ)
    θ .= θ .- η .* grad

    current_loss = loss(θ)
    if current_loss < melhor_loss
        melhor_loss = current_loss
        melhor_θ = deepcopy(θ)
        melhor_epoch = epoch
    end

    @printf("Epoch %3d | Loss = %12.9f | Melhor Epoch = %3d | Melhor Loss = %12.9f\n",
        epoch, Posit2x(current_loss), melhor_epoch, Posit2x(melhor_loss))
end

# Visualização
ts_plot = Float64.(range(0, 1; length=1000))
u_pred = [Float64(Posit2x(trial([T(t)], melhor_θ))) for t in ts_plot]
us_best = [Float64(Posit2x(trial([T(t)], melhor_θ))) for t in ts_plot]
plt = plot(ts_plot, u_pred, lw=2, label="Rede Posit{8,2} ótimo Epoch $melhor_epoch", xlabel="t", ylabel="u(t)", color=:red)

function exact_eq1_float64_to_posit(T::Type{<:Posit})
    return t -> T(sin(2π * Float64(t)) / (2π))
end
u_conv = [Float64(Posit2x(exact_eq1_float64_to_posit(T)(T(t)))) for t in ts_plot]

plot!(plt, ts_plot, u_conv, label="Posit{8,2} convertido", c=:red, lw=1, ls=:dash)

# Solução analítica
u_exact = [sin(2π * t) / (2π) for t in ts_plot]
plot!(plt, ts_plot, u_exact, lw=2, ls=:dash, label="Solução exata (Float64)", color=:black)
