using Lux, Random, ForwardDiff, ComponentArrays, Plots, Printf
using PositPR
using Base.Threads

T = Posit{16,2}
rng = MersenneTwister(1234)
epochs = 2000

global PI_T = pi_ramanujan(T)
global ONE_T = one(T)
global ZERO_T = zero(T)
global N = 20 #domain points
global pts_x = range(zero(T) + T(1/N), one(T) - T(1/N); length=N-2)
global pts_t = range(zero(T) + T(1/N), one(T); length=N-1)

exact(x, t) = exp(-PI_T^2 * t) * sin(PI_T * x)

#NN definition
chain = Chain(
    Dense(2, 8, tanh; init_weight=randnP16_2, init_bias=zerosP16_2),
    Dense(8, 1; init_weight=randnP16_2, init_bias=zerosP16_2)
)

θ, st = Lux.setup(rng, chain)
θ = ComponentArray(θ)

#trial_solution : u(0,t)=u(1,t)=0, u(x,0)=sin(πx)
function trial(p::Vector, θ)
    x, t = p
    û, _ = chain(p, θ, st)
    return (ONE_T - x) * x * t * only(û) + sin(PI_T * x)
end

function residual_eq(x, t, θ)
    p = [x, t]
    g(p) = trial(p, θ)
    ∂t = ForwardDiff.derivative(z -> g([x, z]), t)
    ∂²x = ForwardDiff.derivative(z -> ForwardDiff.derivative(w -> g([w, t]), z), x)
    return ∂t - ∂²x
end

# loss function: interior points only
function loss(θ)

    chunk_size_x = ceil(Int, length(pts_x) / Threads.nthreads())
    chunks = Iterators.partition(pts_x, chunk_size_x)

    tasks = map(chunks) do chunk_x
        Threads.@spawn begin
            local_sum = ZERO_T
            for x in chunk_x, t in pts_t
                r = residual_eq(x,t,θ)
                local_sum += r^2
            end
            return local_sum
        end
    end
    
    total = sum(fetch.(tasks))
    return total / (length(pts_x) * length(pts_t))
end

# training "manual Adam"
η = T(0.01)
β1, β2 = T(0.9), T(0.999)
eps = T(1e-8)
m = zero.(θ)
v = zero.(θ)

global best_loss = typemax(T)
global best_θ = deepcopy(θ)
global best_epoch = 0

for epoch in 1:epochs
    grad = ForwardDiff.gradient(loss, θ)
    m .= β1 .* m .+ (ONE_T - β1) .* grad
    v .= β2 .* v .+ (ONE_T - β2) .* grad.^2
    m̂ = m ./ (ONE_T - β1^epoch)
    v̂ = v ./ (ONE_T - β2^epoch)
    θ .= θ .- η .* m̂ ./ (sqrt.(v̂) .+ eps)

    l = loss(θ)
    if l < best_loss
        best_loss = l
        best_θ = deepcopy(θ)
        best_epoch = epoch
    end

    if (epoch % 10 == 0)
        #plots
        xs = ts = range(0, 1; length=100)
        u_pred = [Float64(Posit2x(trial([T(x), T(t)], best_θ))) for x in xs, t in ts]
        u_exact = [Float64(Posit2x(exact(T(x), T(t)))) for x in xs, t in ts]
        erro = [abs(a - b) for (a, b) in zip(u_pred, u_exact)]

        p1 = heatmap(xs, ts, reshape(u_pred, length(xs), length(ts))', title="Posit{16,2} NN Solution", xlabel="x", ylabel="t", clim=(0, 1))
        p2 = heatmap(xs, ts, reshape(u_exact, length(xs), length(ts))', title="Posit{16,2} Analytical Solution", xlabel="x", ylabel="t", clim=(0, 1))
        p3 = heatmap(xs, ts, reshape(erro, length(xs), length(ts))', title="Absolute Error", xlabel="x", ylabel="t", clim=(0, 1))

        plt = plot(p1, p2, p3, layout=(1,3), size=(1200, 500))
        savefig(plt, "$(T)_$(epoch)_$(best_epoch).png")
    end

    @printf("Epoch %3d | Loss = %.9f\n", epoch, Posit2x(l))
end

