using NeuralPDE, Lux, Random, ForwardDiff, ComponentArrays
using Optimization, OptimizationOptimisers, ModelingToolkit
using PositPR
using Plots

T = Posit{16,2}
rng = MersenneTwister(1234)
maxiters = 500 #epochs

global TWO_PI = T(2) * pi_ramanujan(T)

#NN definition
chain = Chain(
    Dense(1, 5, tanh; init_weight=randnP16_2, init_bias=zerosP16_2),
    Dense(5, 5, tanh; init_weight=randnP16_2, init_bias=zerosP16_2),
    Dense(5, 1; init_weight=randnP16_2, init_bias=zerosP16_2)
)
θ, st = Lux.setup(rng, chain)
θ = ComponentArray(θ)

opt = Optimisers.Adam(0.01)
ts = range(zero(T), one(T), length=50)

#symbolic ODE
@parameters t
@variables u(..)
Dt = Differential(t)

expr = cos(TWO_PI * t)

eq = Dt(u(t)) ~ expr
bcs = [u(0) ~ zero(T),
u(1) ~ zero(T)]
domain = [t ∈ Interval(zero(T), one(T))]
@named pde_system = PDESystem(eq, bcs, domain, [t], [u(t)])


#strategy and training
strategy = StochasticTraining(100, bcs_points = 2)
discretization = PhysicsInformedNN(chain, strategy, init_params=θ)

Problem = symbolic_discretize(pde_system, discretization)

pde_loss_functions = Problem.loss_functions.pde_loss_functions
bc_loss_functions = Problem.loss_functions.bc_loss_functions
loss_functions = [pde_loss_functions; bc_loss_functions]


function loss_function(θ, p)
    return sum(map(l -> l(θ), loss_functions))
end

function callback(p, l)
    println("Iteracao: ", p.iter)
    println("loss: ", float(l))
    println("pde_losses: ", map(l_ -> float.(l_(p.u)), pde_loss_functions))
    println("bcs_losses: ", map(l_ -> float.(l_(p.u)), bc_loss_functions))
	return false
end

f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())
prob = Optimization.OptimizationProblem(f_, Problem.init_params)
phi = Problem.phi
println("phi(t, θ): ", float(phi(T(0.5), θ)))
alg = Optimisers.Adam(0.01)

println("Treinando com Posit{16,2}")
sol = solve(prob, alg, maxiters=maxiters, saveat=ts, callback = callback)

function to_posit8(x)
    x isa Number && return Posit{8,2}(x)
    x isa AbstractArray && return map(to_posit8, x)
    x isa NamedTuple && return NamedTuple{keys(x)}(map(to_posit8, values(x)))
    return x
end

ts_plot = Float64.(range(0, 1; length=1000))
u_predict_hi = [first(phi(T(t), sol.minimizer)) for t in ts_plot]
exact_u(t) = sin(2π * t) / (2π)

function exact_eq1_float64_to_posit(T::Type{<:Posit})
    return t -> T(sin(2π * Float64(t)) / (2π))
end

plt = plot(ts_plot, u_predict_hi; lw=2, label="Posit{16,2}", xlabel="t", ylabel="u(t)", color=:blue)
u_hi_conv = [Float64(Posit2x(exact_eq1_float64_to_posit(T)(T(t)))) for t in ts_plot]

plot!(plt, ts_plot, u_hi_conv, label="Posit{16,2} convertido", c=:blue, lw=1, ls=:dash, xlabel="t", ylabel="u(t)")
plot!(plt, ts_plot, [exact_u(t) for t in ts_plot]; lw=2, ls=:dash, lc=:black, label="Solução Exata")
