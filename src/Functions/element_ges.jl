mutable struct Elementg{T,Ts<:HyperelasticDisplacementSolver,Tg<:AbstractVector{<:AbstractVector{T}},
    Tg0<:AbstractVector{<:AbstractVector{T}},Tp<:AbstractPenalty{T},Tb<:AbstractVector{T},Tc<:CellScalarValues,Tcv<:CellVectorValues} <: AbstractFunction{T}
    solver::Ts
    ges::Tg
    ges_0::Tg0
    penalty::Tp
    xmin::T
    gesize::Int
    body_force::Tb
    cellvalues::Tc
    cellvaluesV::Tcv
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::Elementg)
    return println("TopOpt element hyperelastic force residual function")
end

function Elementg(solver::HyperelasticDisplacementSolver)
    T = eltype(solver.elementinfo.ges[1])
    penalty = getpenalty(solver)
    xmin = solver.xmin
    quad_order=FEA.default_quad_order(solver.problem)
    dims = TopOptProblems.getdim(solver.problem)
    geom_order = TopOptProblems.getgeomorder(solver.problem)
    refshape = Ferrite.getrefshape(solver.problem.ch.dh.field_interpolations[1])
    interpolation_space = Ferrite.Lagrange{dims,refshape,geom_order}()
    quadrature_rule = Ferrite.QuadratureRule{dims,refshape}(quad_order) 
    cellvalues = CellScalarValues(quadrature_rule, interpolation_space)
    cellvaluesV = Ferrite.CellVectorValues(quadrature_rule, interpolation_space) 
    n_basefuncs = getnbasefunctions(cellvalues)
    gesize = dims * n_basefuncs
    ρ = TopOptProblems.getdensity(solver.problem)
    g = [0.0, 9.81, 0.0]
    body_force = ρ .* g
    ges_0 = [zeros(T, ndofs_per_cell(solver.problem.ch.dh)) for i in 1:length(solver.problem.varind)]
    ges = [similar(x) for x in ges_0]
    return  Elementg(solver, ges, ges_0, penalty, xmin, gesize, body_force, cellvalues, cellvaluesV) 
end

function (eg::Elementg)(xe::Number,ue::AbstractVector{<:Number})
    @unpack solver, gesize, body_force, cellvalues, cellvaluesV, xmin, penalty = eg
    ge = zeros(eltype(ue),(ndofs_per_cell(solver.problem.ch.dh)))
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        ∇u = function_gradient(cellvaluesV, q_point, ue) # JGB add (NEEDS TO BE CHECKED!!)
        F = one(∇u) + ∇u # JGB add 
        C = tdot(F) # JGB add 
        S, ∂S∂C = TopOptProblems.constitutive_driver(C, solver.mp) # JGB add 
        P = F ⋅ S # JGB add 
        for b in 1:gesize
            ∇ϕb = shape_gradient(cellvaluesV, q_point, b) # JGB: like ∇δui
            ϕb = shape_value(cellvaluesV, q_point, b)
            ge[b] += ( ∇ϕb ⊡ P - ϕb ⋅ body_force ) * dΩ
        end
    end
    if PENALTY_BEFORE_INTERPOLATION
        px = density(penalty(xe), xmin)
    else
        px = penalty(density(xe), xmin)
    end
    return px*ge
end

function (eg::Elementg{Teg})(x::PseudoDensities,u::TopOpt.Functions.DisplacementResult{T, N, V}) where {Teg, T, N, V}
@unpack solver, ges, ges_0, cellvalues, cellvaluesV = eg  
    ges = copy(ges_0)  
    celliterator = CellIterator(solver.problem.ch.dh)
    for (ci, cell) in enumerate(celliterator)
        dofs=celldofs(cell)
        reinit!(cellvalues, cell)
        reinit!(cellvaluesV, cell)
        cell_g = eg(x.x[ci],u.u[dofs])
        ges[ci] += cell_g
    end
    return ges
end

function ChainRulesCore.rrule(eg::Elementg, x::PseudoDensities, u::DisplacementResult)
    @unpack solver, cellvalues, cellvaluesV = eg    
    ges = eg(x,u)
    function pullback_fn(Δ)
        celliterator = CellIterator(solver.problem.ch.dh)
        Δu = zeros(length(u.u))
        Δu_threaded = [zeros(length(u.u)) for _ in 1:Threads.nthreads()]
        Δx = zeros(length(x.x))
        Δx_threaded = [zeros(length(x.x)) for _ in 1:Threads.nthreads()]
        sample_dofs = celldofs(first(CellIterator(solver.problem.ch.dh)))
        ndof = length(sample_dofs) 
        nel = length(x.x)
        jac_cell_ue = zeros(ndof, ndof)
        der_cell_xe = zeros(nel)
        for (ci, cell) in enumerate(celliterator)  
            dofs=celldofs(cell)
            reinit!(cellvalues, cell)
            reinit!(cellvaluesV, cell)
            ges_cell_fn_ue = ue -> vec(eg(x.x[ci],ue))
            jacobian_options = ForwardDiff.JacobianConfig(ges_cell_fn_ue,u.u[sample_dofs])
            ForwardDiff.jacobian!(jac_cell_ue, ges_cell_fn_ue, u.u[dofs], jacobian_options)
            ges_cell_fn_xe = xe -> vec(eg(xe,u.u[dofs]))
            der_cell_xe = ForwardDiff.derivative(ges_cell_fn_xe, x.x[ci])

            tid = Threads.threadid()
            Δu_threaded[tid][dofs] .+= jac_cell_ue' * vec(Δ[ci])
            Δx_threaded[tid][ci] +=  dot(der_cell_xe,vec(Δ[ci]))
        end
        Δu = mapreduce(identity, +, Δu_threaded) 
        Δx = mapreduce(identity, +, Δx_threaded)
        return  Tangent{typeof(eg)}(;
            solver=NoTangent(),
            ges=Δ,
            ges_0=NoTangent(),
            penalty=NoTangent(),
            xmin=NoTangent(),
            gesize=NoTangent(),
            body_force=NoTangent(),
            cellvalues=NoTangent(),
            cellvaluesV=NoTangent()),
            Δx,
            Δu
    end
    return ges, pullback_fn
end 