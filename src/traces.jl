import Base: length, append!, sizehint!

struct TopOptTrace{T,TI<:Integer}
	c_hist::Vector{T}
    v_hist::Vector{T}
	x_hist::Vector{Vector{T}}
	add_hist::Vector{TI}
	rem_hist::Vector{TI}
end
TopOptTrace{T,TI}() where {T,TI<:Integer} = TopOptTrace{T,TI}(Vector{T}(), Vector{T}(), Vector{Vector{T}}(), Vector{TI}(), Vector{TI}())
length(t::TopOptTrace) = length(t.v_hist)

topopt_trace_fields = fieldnames(TopOptTrace)

macro append_fields_t1_t2()
    return esc(Expr(:block, [Expr(:call, :append!, :(t1.$f), :(t2.$f)) for f in topopt_trace_fields]...))
end
function append!(t1::TopOptTrace, t2::TopOptTrace)
    @append_fields_t1_t2()
end

macro sizehint!_fields_t()
    return esc(Expr(:block, [Expr(:call, :sizehint!, :(t.$f), :n) for f in topopt_trace_fields]...))
end
function sizehint!(t::TopOptTrace, n)
    @sizehint!_fields_t()
    return
end

function append!(ts::Vector{<:TopOptTrace})
    sizehint!(ts[1], sum(length.(ts)))
    for i in 2:length(ts)
        append!(ts[1], ts[i])
    end
    return ts[1]
end
