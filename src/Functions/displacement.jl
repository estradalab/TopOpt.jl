mutable struct Displacement{
    T,
    Tu<:AbstractVector{T},
    Td<:AbstractVector,
    Ts<:AbstractDisplacementSolver,
    Tg<:AbstractVector{<:Integer},
} <: AbstractFunction{T}
    u::Tu # displacement vector
    dudx_tmp::Td # directional derivative
    solver::Ts
    global_dofs::Tg
    fevals::Int
    maxfevals::Int
end

@params mutable struct HyperelasticDisplacement{T} <: AbstractFunction{T}
    u::AbstractVector{T} # displacement vector
    F::AbstractVector # deformation gradient tensor
    dudx_tmp::AbstractVector # directional derivative
    solver::AbstractHyperelasticDisplacementSolver
    global_dofs::AbstractVector{<:Integer}
    fevals::Int
    maxfevals::Int
    ek::ElementK
    eg::TopOpt.Functions.Elementg
    Assemble_K::AssembleK
    Assemble_g::TopOpt.Functions.Assemblef
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::Displacement)
    return println("TopOpt displacement function")
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::HyperelasticDisplacement)
    return println("TopOpt displacement function for hyperelastic strain regimes")
end

Base.length(u::DisplacementResult) = length(u.u)
Base.size(u::DisplacementResult, i...) = size(u.u, i...)
Base.getindex(u::DisplacementResult, i...) = u.u[i...]
Base.sum(u::DisplacementResult) = sum(u.u)
LinearAlgebra.dot(u::DisplacementResult, weights::AbstractArray) = dot(u.u, weights)

"""
    Displacement()

Construct the Displacement function struct.
"""
function Displacement(solver::AbstractFEASolver; maxfevals=10^8)
    T = eltype(solver.u)
    dh = solver.problem.ch.dh
    k = ndofs_per_cell(dh)
    global_dofs = zeros(Int, k)
    total_ndof = ndofs(dh)
    u = zeros(T, total_ndof)
    dudx_tmp = zeros(T, length(solver.vars))
    return Displacement(u, dudx_tmp, solver, global_dofs, 0, maxfevals)
end

function Displacement(solver::AbstractHyperelasticDisplacementSolver; maxfevals=10^8)
    dim = TopOptProblems.getdim(solver.problem)
    dim == 3 || throw("2D hyperelastic FEA is not supported yet.")
    T = eltype(solver.u)
    dh = solver.problem.ch.dh
    k = ndofs_per_cell(dh)
    global_dofs = zeros(Int, k)
    total_ndof = ndofs(dh)
    u = zeros(T, total_ndof)
    F = [zeros(typeof(solver.elementinfo.Fes[1])) for _ in 1:total_ndof/dim]
    dudx_tmp = zeros(T, length(solver.vars))
    ek = ElementK(solver) 
    eg = TopOpt.Functions.Elementg(solver)
    Assemble_K = AssembleK(solver.problem) 
    Assemble_g = TopOpt.Functions.Assemblef(solver.problem) 
    return HyperelasticDisplacement(u, F, dudx_tmp, solver, global_dofs, 0, maxfevals, ek, eg, Assemble_K, Assemble_g)
end

"""
# Arguments
`x` = design variables

# Returns
displacement vector `u`
"""
function (dp::Displacement{T})(x::PseudoDensities) where {T}
    @unpack solver, global_dofs = dp
    @unpack penalty, problem, xmin = solver
    dp.fevals += 1
    @assert length(global_dofs) == ndofs_per_cell(solver.problem.ch.dh)
    solver.vars .= x.x
    solver()
    return DisplacementResult(copy(solver.u))
end

function (dp::HyperelasticDisplacement{T})(x::PseudoDensities) where {T}
    @unpack solver, global_dofs = dp
    @unpack penalty, problem, xmin = solver
    dp.fevals += 1
    @assert length(global_dofs) == ndofs_per_cell(solver.problem.ch.dh)
    solver.vars = x.x
    solver()
    return DisplacementResult(copy(solver.u)),copy(solver.F) #, copy(solver.F) # I need to add F support
end

"""
rrule for linear elastic solver autodiff.
    
du/dxe = -K^-1 * dK/dxe * u
d(u)/d(x_e) = - K^-1 * d(K)/d(x_e) * u
            = - K^-1 * (Σ_ei d(ρ_ei)/d(x_e) * K_ei) * u
            = - K^-1 * [d(ρ_e)/d(x_e) * K_e * u]
d(u)/d(x_e)' * Δ = -d(ρ_e)/d(x_e) * u' * K_e * (K^-1 * Δ)

where d(u)/d(x) ∈ (nDof x nCell); d(u)/d(x)^T * Δ = (nCell x nDof) * (nDof x 1) -> nCell x 1
"""
function ChainRulesCore.rrule(dp::Displacement, x::PseudoDensities)
    @unpack dudx_tmp, solver, global_dofs = dp
    @unpack penalty, problem, u, xmin = solver
    dh = getdh(problem)
    @unpack Kes = solver.elementinfo
    # Forward-pass
    # Cholesky factorisation
    u = dp(x)
    return u, Δ -> begin # v
        if hasproperty(Δ, :u)
            solver.rhs .= Δ.u
        else
            solver.rhs .= Δ
        end
        solver(; reuse_fact=true, assemble_f=false)
        dudx_tmp .= 0
        for e in 1:length(x.x)
            _, dρe = get_ρ_dρ(x.x[e], penalty, xmin)
            celldofs!(global_dofs, dh, e)
            Keu = bcmatrix(Kes[e]) * u.u[global_dofs]
            dudx_tmp[e] = -dρe * dot(Keu, solver.lhs[global_dofs])
        end
        return nothing, Tangent{typeof(x)}(; x=dudx_tmp) # J1' * v, J2' * v
    end
end

#TODO ensure that this works with Neumann boundary conditions

function ChainRulesCore.rrule(dp::HyperelasticDisplacement, x::PseudoDensities)
    @unpack ek, eg, Assemble_K, Assemble_g, solver = dp
    full_output = dp(x)
    u = full_output[1]
    forward(x) = dp(PseudoDensities(x))[1]
    function conditions(x::PseudoDensities, u::TopOpt.Functions.DisplacementResult)
        K_ = Assemble_K(ek(x,u))
        g_ = Assemble_g(eg(x,u))
        _, g = apply_boundary_with_meandiag!(copy(K_), solver.problem.ch, copy(g_), true)
        return g
    end                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    function solver_wrapper(A, b)
        A_mat = Matrix(A)
        for pdof in solver.problem.ch.prescribed_dofs
            A_mat[pdof,pdof] = 1.0
        end
        return Krylov.gmres(A_mat, b; itmax=10000) 
    end
    implicit = ImplicitDifferentiation.ImplicitFunction(forward, conditions, solver_wrapper)
    function pullback(Δ)
        _, back = Zygote.rrule_via_ad(Zygote.ZygoteRuleConfig(),implicit, x)
        _, x̄ = back(Δ[1])
        return NoTangent(), Tangent{typeof(x)}(; x=x̄), NoTangent() # F does have a tangent, this is just a placeholder
    end
    return full_output, pullback
end