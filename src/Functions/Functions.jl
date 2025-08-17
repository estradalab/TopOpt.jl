module Functions

using ..TopOpt: dim, TopOpt, PENALTY_BEFORE_INTERPOLATION, PseudoDensities
using ..TopOptProblems, ..TrussTopOptProblems
using ..TopOptProblems: initialize_K, getdh
using ..FEA, ..CheqFilters
using ..Utilities, ForwardDiff, LinearAlgebra
using ..TrussTopOptProblems: getA, compute_local_axes

using Parameters: @unpack
using TimerOutputs, Ferrite, StaticArrays, StatsFuns
using SparseArrays, Statistics, ChainRulesCore, Zygote
using Nonconvex: Nonconvex
using Flux
using AbstractDifferentiation: AbstractDifferentiation
using StatsBase, Statistics 
using Symbolics  
using Plots:heatmap  
using ThreadsX
using ImplicitDifferentiation
using Krylov
using IterativeSolvers
const AD = AbstractDifferentiation

export Volume,
    Compliance,
    Displacement,
    MeanCompliance,
    BlockCompliance,
    AbstractFunction,
    getfevals,
    getmaxfevals,
    maxedfevals,
    von_mises_stress_function,
    project,
    generate_scenarios,
    hutch_rand!,
    hadamard!,
    TrussStress,
    AssembleK,
    Assemblef,
    TrussElementKσ,
    ElementK,
    Elementg,
    apply_boundary_with_zerodiag!,
    apply_boundary_with_meandiag!,
    NeuralNetwork,
    TrainFunction,
    PredictFunction,
    NNParams,
    Coordinates,
    StressTensor,
    ElementStressTensor,
    MaterialInterpolation,
    MultiMaterialVariables,
    element_densities,
    tounit,
    DefGradTensor,
    ElementDefGradTensor, 
    FToK123,
    orth_decomp,
    sensitivityFieldFncs,
    sensitivityPVW,
    Entropy_Calc,
    Entropy,
    SMu_gen, 
    SAlpha_gen

const to = TimerOutput()

abstract type AbstractFunction{T} <: Nonconvex.NonconvexCore.AbstractFunction end
struct DisplacementResult{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    u::A
end

# buckling / hyperelasticity
include("apply_boundary.jl")
include("assemble_K.jl")
include("assemble_f.jl")
include("element_ksigma.jl")
include("element_k.jl")
include("element_ges.jl")

include("function_utils.jl")
include("compliance.jl")
include("displacement.jl")
include("volume.jl")
include("trace.jl")
include("mean_compliance.jl")
include("block_compliance.jl")

# stress
include("stress_tensor.jl")

# goodness related
include("defgrad.jl")
include("goodness.jl")

# TODO no rrules yet
include("truss_stress.jl")

include("neural.jl")

include("interpolation.jl")

end