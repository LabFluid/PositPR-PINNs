using NeuralPDE, Lux, Random, ComponentArrays
using OptimizationOptimisers, ForwardDiff, Optimization
using PositPR
using Plots, Printf

T_hi = Posit{16,2}
T_lo = Posit{8,2}
rng = MersenneTwister(1234)
epochs_hi = 50
epochs_lo = 100

global N = 50

#NN definition
chain = Chain(
    Dense(1, 5, tanh; init_weight=randnP16_2, init_bias=zerosP16_2),
    Dense(5, 5, tanh; init_weight=randnP16_2, init_bias=zerosP16_2),
    Dense(5, 1; init_weight=randnP16_2, init_bias=zerosP16_2)
)

θ, st = Lux.setup(rng, chain)
θ = ComponentArray(θ)

function make_loss(T, chain, st)
    ZERO_T = zero(T)
    ONE_T = one(T)
    TWO_PI = T(2.0) * pi_ramanujan(T)
    f(u, t) = cos(t * TWO_PI)

    function trial_solution(input, θ)
        out, _ = chain(input, θ, st)
        t = input[1]
        return t*(1 - t) * only(out)
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

loss_hi, trial_hi = make_loss(T_hi, chain, st)

global best_loss_hi = typemax(T_hi)
global best_θ_hi = deepcopy(θ)
global best_epoch_hi = 0

η_hi = T_hi(0.01)
β1_hi = T_hi(0.9)
β2_hi = T_hi(0.999)
eps_hi = T_hi(1e-8)

m = zero.(θ)
v = zero.(θ)

println("Training using $(T_hi) ...")

for epoch in 1:epochs_hi
    global best_loss_hi
    global best_θ_hi
    global best_epoch_hi

    grad = ForwardDiff.gradient(loss_hi, θ)
    m .= β1_hi .* m .+ (one(T_hi) - β1_hi) .* grad
    v .= β2_hi .* v .+ (one(T_hi) - β2_hi) .* grad.^2
    m̂ = m ./ (one(T_hi) - β1_hi^epoch)
    v̂ = v ./ (one(T_hi) - β2_hi^epoch)
    θ .= θ .- η_hi .* m̂ ./ (sqrt.(v̂) .+ eps_hi)

    current_loss = loss_hi(θ)
    if current_loss < best_loss_hi
        best_loss_hi = current_loss
        best_θ_hi = deepcopy(θ)
        best_epoch_hi = epoch
    end

    @printf("Epoch %3d | Loss = %12.9f | Melhor Epoch = %3d | Melhor Loss = %12.9f\n",
        epoch, Posit2x(current_loss), best_epoch_hi, Posit2x(best_loss_hi))

end

function to_posit8(x)
    x isa Number && return Posit{8,2}(x)
    x isa AbstractArray && return map(to_posit8, x)
    x isa NamedTuple && return NamedTuple{keys(x)}(map(to_posit8, values(x)))
    return x
end

θ_lo, st_lo = Lux.setup(rng, chain)
θ_lo = to_posit8(best_θ_hi)
loss_lo, trial_lo = make_loss(T_lo, chain, st)

global best_loss_lo = typemax(T_lo)
global best_θ_lo = deepcopy(θ_lo)
global best_epoch_lo = 0

# SGD T_lo
η_lo = T_lo(0.01)

println("Training using $(T_lo) ...")

for epoch in epochs_hi+1:epochs_lo
    grad = ForwardDiff.gradient(loss_lo, θ_lo)
    θ_lo .= θ_lo .- η_lo .* grad

    current_loss = loss_lo(θ_lo)
    if current_loss < best_loss_lo
        best_loss_lo = current_loss
        best_θ_lo = deepcopy(θ_lo)
        best_epoch_lo = epoch
    end

    #println("Epoch $epoch | Loss = $(Posit2x(current_loss)) | Melhor Epoch = $best_epoch_lo | Melhor Loss = $(Posit2x(best_loss_lo))")
    @printf("Epoch %3d | Loss = %12.9f | Melhor Epoch = %3d | Melhor Loss = %12.9f\n",
        epoch, Posit2x(current_loss), best_epoch_lo, Posit2x(best_loss_lo))

end

ts_plot = Float64.(range(0, 1; length=1000))
plt = plot(title="", xlabel="t", ylabel="u(t)")

us_best_hi = [Float64(Posit2x(trial_hi([T_hi(t)], best_θ_hi))) for t in ts_plot]
plot!(plt, ts_plot, us_best_hi, lw=2, label="Posit{16,2} best Epoch $best_epoch_hi ", color=:blue)

us_best_lo = [Float64(Posit2x(trial_lo([T_lo(t)], best_θ_lo))) for t in ts_plot]
plot!(plt, ts_plot, us_best_lo, lw=2, label="Posit{8,2} best Epoch $best_epoch_lo ", color=:red)

function exact_eq1_float64_to_posit(T::Type{<:Posit})
    return t -> T(sin(2π * Float64(t)) / (2π))
end

u_hi_conv = [Float64(Posit2x(exact_eq1_float64_to_posit(T_hi)(T_hi(t)))) for t in ts_plot]
u_lo_conv = [Float64(Posit2x(exact_eq1_float64_to_posit(T_lo)(T_lo(t)))) for t in ts_plot]

plot!(plt, ts_plot, u_hi_conv, label="Posit{16,2} converted", c=:blue, lw=1, ls=:dash, xlabel="t", ylabel="u(t)")
plot!(plt, ts_plot, u_lo_conv, label="Posit{8,2} converted", c=:red, lw=1, ls=:dash)
plot!(plt, ts_plot, t -> sin(2π * t) / (2π), lw=2, ls=:dash, lc=:black, label="Float64")
