mutable struct Assemblef{T,Tp<:StiffnessTopOptProblem,Tf<:AbstractVector{T},Tg<:AbstractVector{<:Integer}} <: AbstractFunction{T}
    problem::Tp
    f::Tf
    global_dofs::Tg # preallocated dof vector for a cell
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::Assemblef)
    return println("TopOpt global forcing vector assembly function")
end

function Assemblef(problem::StiffnessTopOptProblem)
    dh = problem.ch.dh
    global_dofs = zeros(Int, ndofs_per_cell(dh))
    return Assemblef(problem, TopOpt.TopOptProblems.initialize_f(problem), global_dofs)
end

"""
    assemblef(fes) = (Σ_ei x_e * f_ei)

Forward-pass function call.
"""
function (ak::Assemblef)(fes::AbstractVector{<:Vector{T}}) where {T}
    @unpack problem, f, global_dofs = ak
    dh = problem.ch.dh
    f .= 0
    fe = zeros(T, size(fes[1]))
    for (i, _) in enumerate(CellIterator(dh))
        celldofs!(global_dofs, dh, i)
        fe = fes[i]
        Ferrite.assemble!(f, global_dofs, fe)
    end
    return copy(f)
end

"""
    ChainRulesCore.rrule(af::Assemblef{T}, fes)

`rrule` for autodiff. 
    
Let's consider composite function `g(F(...))`, where
`F` can be a struct-valued, vector-valued, or matrix-valued function.
In the case here, `F = Assemblef`. Then `rrule` wants us to find `g`'s derivative
w.r.t each *input* of `F`, given `g`'s derivative w.r.t. each *output* of `F`.
Here, `F: f_e -> f_i = sum_e f_e_i. Then `df_i/df_e_i = 1`.
And we know `Delta_i = dg/df_i`.

And our goal for `rrule` is to go from `dg/df_i` to `dg/df_e_i`, 
which has the same structure as the input `f_e`.

    dg/df_e_i = sum_i' dg/df_i' * df_i'/df_e_i
    		   = dg/df_i df_i/df_e_i
    		   # i above is a global indice
    		   # i below is a local indice
    		   = Delta[global_dofs[i]]

    (df_i'/df_e_i = 0 unless i' == i, i in e)

which can be shortened as:

    dg/df_e = Delta[global_dofs]
"""
function ChainRulesCore.rrule(
    af::Assemblef{T}, fes::AbstractVector{<:AbstractVector{T}}
) where {T}
    @unpack problem, f, global_dofs = af
    dh = problem.ch.dh
    # * forward-pass
    f = af(fes)
    n_dofs = length(global_dofs)
    function assemblef_pullback(Δ)
        Δfes = [zeros(T, n_dofs) for _ in 1:getncells(dh.grid)]
        for (ci, _) in enumerate(CellIterator(dh))
            celldofs!(global_dofs, dh, ci)
            Δfes[ci] = Δ[global_dofs]
        end
        return Tangent{typeof(af)}(; problem=NoTangent(), f=Δ, global_dofs=NoTangent()),
        Δfes
    end
    return f, assemblef_pullback
end
