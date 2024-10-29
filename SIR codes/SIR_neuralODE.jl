#This code does not use GPU
println("Use NN to solve SIR ODE model")
using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq 
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random, Plots
function sir_ode(du,u,p,t)
    S,I,R = u
    b,g,N = p
    du[1] = -b*S*I/N
    du[2] = b*S*I/N-g*I
    du[3] = g*I
end
N=1000.0 #total population
I0=1.0 #initial infected
R0=0.0 #initial recovered
b0=0.3 #transmission rate
g0=0.1 #recovery rate
S0=N-I0-R0 #initial susceptible
parms = [b0,g0,N] #initial parms
u0 = [S0,I0,R0] #initial variables
tspan = [0.0,160.0] #timespan
datasize=160
tsteps = range(tspan[1], tspan[2]; length=datasize)
#Use ODE to solve the SIR problem
sir_prob = ODEProblem(sir_ode,u0,tspan,parms) #define the ODE
sir_sol = solve(sir_prob,Tsit5(),saveat = tsteps)

plot(sir_sol,xlabel="Days",ylabel="Number")

# make an Array for ode solution
ode_data = Array(sir_sol)

#define Neural Network (NN)
rng = Xoshiro(0) #seeding
dudt = Lux.Chain(Lux.Dense(3, 100, tanh), Lux.Dense(100, 100, tanh), Lux.Dense(100,3)) # model name is du/dt
reltol = 1e-7
abstol = 1e-9
# Parameter and State Variables
p, st = Lux.setup(rng, dudt) |> f64
sir_neuralode = NeuralODE(dudt, tspan, Tsit5(), saveat = tsteps,reltol=reltol,abstol=abstol)

# create the neural ODE output data
function predict_neuralode(p)
    (Array(sir_neuralode(u0, p, st)[1]))
end
#check the loss ; difference between original & NN prediction 
function loss_neuralode(p)
    pred = predict_neuralode(p)
    lossl2 = sum(abs2, ode_data .- pred)
    return lossl2
end

# A Callback function to plot the comparison between original ODE vs NN solution for suspectible, infected and recovered individuals
lossrecord=Float64[]
callback = function (state, l; doplot=true)
    if doplot
        pred = predict_neuralode(state.u)
        splt = scatter(tsteps, ode_data[1, :]; label="true suspectible")
        scatter!(splt, tsteps, pred[1, :]; label="prediction suspectible")

        iplt = scatter(tsteps, ode_data[2, :]; label="true infected")
        scatter!(iplt, tsteps, pred[2, :]; label="prediction infected")

        rplt = scatter(tsteps, ode_data[3, :]; label="true recovered")
        scatter!(rplt, tsteps, pred[3, :]; label="prediction recovered")

        display(plot(splt, iplt, rplt))
        push!(lossrecord, l)
    else
        println(l)
    end 
    return false
end

# Try the callback function to see if it works.
pinit = ComponentArray(p)
callback((; u = pinit), loss_neuralode(pinit); doplot=false)

# Use https://github.com/SciML/Optimization.jl to solve the problem and https://github.com/FluxML/Zygote.jl for automatic differentiation (AD).
adtype = Optimization.AutoZygote()

# Define a [function](https://docs.sciml.ai/Optimization/stable/API/optimization_function/) to optimize with AD.
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

# Define an `OptimizationProblem`
optprob = Optimization.OptimizationProblem(optf, pinit)
it_n = 100 # iteration numbers

# Solve the `OptimizationProblem` using the ADAM optimizer first to get a rough estimate.
result_neuralode = Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.001),
    callback = callback,
    maxiters = it_n
)

println("Loss is: ", loss_neuralode(result_neuralode.u))

# Use another optimizer (BFGS) to refine the solution.
optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(
    optprob2,
    Optim.BFGS(; initial_stepnorm = 0.001),
    callback = callback,
    maxiters = it_n,
    allow_f_increases = false
)

println("Loss is: ", loss_neuralode(result_neuralode2.u))
#plot loss data
lossrecord

plot(lossrecord[1:it_n], xlabel="Iters", ylabel="Loss", lab="Adam", yscale=:log10)
plot!(it_n:length(lossrecord), lossrecord[it_n:end], lab="BFGS")
using CSV, Tables
CSV.write("LossData.csv",  Tables.table(lossrecord), writeheader=false)
