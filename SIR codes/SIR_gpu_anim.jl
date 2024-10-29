#This code stores plot as animation
using Lux, Optimization, OptimizationOptimisers, OptimizationOptimJL, Zygote, 
OrdinaryDiffEq, Plots, LuxCUDA, SciMLSensitivity, Random, ComponentArrays
import DiffEqFlux: NeuralODE
println("Use NN to solve SIR ODE model")
CUDA.allowscalar(false)
gdev = gpu_device()
# u = [s(t), I(t), R(t)]
function trueSirModel!(du, u, p, t)
    beta, gamma, N = p
    du[1] = -(beta * u[1] * u[2]) / N
    du[2] = ((beta * u[1] * u[2]) / N) - (gamma * u[2])
    du[3] = gamma * u[2]
end

# Boundary conditions
N = 1000
i0 = 1
r0 = 0
s0 = (N - i0 - r0)
u0 = Float32[s0, i0, r0]

# constants
beta = 0.3
gamma = 0.1
p = Float32[beta, gamma, N]
# time duration
tspan = Float32[0.0, 160.0]
datasize = 160
tsteps = range(tspan[1], tspan[2]; length=datasize)

# Solving the ODE solution

trueOdeProblem = ODEProblem(trueSirModel!, u0, tspan, p)
trueOdeData = solve(trueOdeProblem, Tsit5(); saveat=tsteps)
trueOdeData = Array(trueOdeData) |> gdev
# Defining the Nueral Network
#rng for Lux.setup
rng = Xoshiro(0)

# After multiple iterations, the layer with 3x150 fit the true data very well.

sirNN = Lux.Chain(Lux.Dense(3, 150, tanh), Lux.Dense(150, 150, tanh), Lux.Dense(150, 3))
u0 = Float32[s0, i0, r0] |> gdev
p, st = Lux.setup(rng, sirNN)
p = p |> ComponentArray |> gdev
st = st |> gdev
sirNNOde = NeuralODE(sirNN, tspan, Tsit5(); saveat=tsteps)

function prediction(p)
    Array(sirNNOde(u0, p, st)[1])
end

# Loss represents the difference between the original data and the predicted output
function loss(p)
    pred = prediction(p)
    l2oss = sum(abs2, trueOdeData .- pred)
    return l2oss
end

# A Callback function to plot the output of the true dat and predicted output for suspectible, infected and recvered data points
anim = Animation()
lossrecord=Float32[] #initialize vector of loss data
callback = function (state, l; doplot = true)
    if doplot
        pred = prediction(state.u)
        plt = scatter(tsteps, Array(trueOdeData[1, :]); label="true suspectible")
        scatter!(plt, tsteps, Array(pred[1, :]); label="prediction suspectible")
        iplt = scatter(tsteps, Array(trueOdeData[2, :]); label="true infected")
        scatter!(iplt, tsteps, Array(pred[2, :]); label="prediction infected")
        rplt = scatter(tsteps, Array(trueOdeData[3, :]); label="true recovered")
        scatter!(rplt, tsteps, Array(pred[3, :]); label="prediction recovered")
        frame(anim)
        push!(lossrecord, l)
    else
        println(l)
    end
    return false
end
# Try the callback function to see if it works.
pinit = ComponentArray(p)
callback((; u = pinit), loss(pinit); doplot=false)
# Defining optimization techniques

adtype = Optimization.AutoZygote()
optimizeFunction = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)

# Defining the problem to be optimized
neuralProblem = Optimization.OptimizationProblem(optimizeFunction, p)

# NN solver that iterates over 3000 using ADAM optimizer
it_n = 3000 # iteration numbers
result_neuralProblem = Optimization.solve(
    neuralProblem, 
    OptimizationOptimisers.Adam(0.001),
    callback=callback, 
    maxiters=it_n
)
println("Loss is: ", loss(result_neuralProblem.u))


# Visualize the fitting process
mp4(anim, fps=15)
lossrecord
plot(lossrecord[1:it_n], xlabel="Iters", ylabel="Loss", lab="Adam", yscale=:log10)
using CSV, Tables
CSV.write("LossData.csv",  Tables.table(lossrecord), writeheader=false)
