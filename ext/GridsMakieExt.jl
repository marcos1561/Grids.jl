module GridsMakieExt
using Grids
import Makie

function Grids.plotgrid!(axis::Makie.Axis, g::AbstractGrid;
                         color=:black, adjust_limits::Bool=true, arcpoints::Integer=200, kwargs...)
    ndims(g) == 2 || throw(ArgumentError("plotting requires a 2D grid"))
    vertices = Makie.Point2d[]
    for line in gridlines(g; arcpoints)
        append!(vertices, (Makie.Point2d(p[1],p[2]) for p in line))
        push!(vertices, Makie.Point2d(NaN, NaN))
    end
    plot = Makie.lines!(axis, vertices; color, kwargs...)
    if adjust_limits
        if g isa PolarGrid
            x,y = gridcenter(g)
            r = last(edges(g)[1])
            Makie.limits!(axis, x-r, x+r, y-r, y+r)
        else
            (xmin,xmax),(ymin,ymax) = bounds(g)
            dx,dy = 0.02*(xmax-xmin),0.02*(ymax-ymin)
            Makie.limits!(axis, xmin-dx, xmax+dx, ymin-dy, ymax+dy)
        end
    end
    return plot
end

function Grids.plotcenters!(axis::Makie.Axis, g::AbstractGrid; color=:black, kwargs...)
    ndims(g) == 2 || throw(ArgumentError("plotting requires a 2D grid"))
    points = cellcenters(g)
    return Makie.scatter!(axis, [Makie.Point2d(p[1],p[2]) for p in points]; color, kwargs...)
end
function Grids.plotcorners!(axis::Makie.Axis, g::AbstractRectangularGrid{T,2}; color=:black, kwargs...) where T
    points = cellcorners(g)
    return Makie.scatter!(axis, [Makie.Point2d(p[1],p[2]) for p in points]; color, kwargs...)
end
function Grids.plotindices!(axis::Makie.Axis, g::AbstractGrid, points::AbstractArray;
                            color=:black, textcolor=:red, fontsize=12, kwargs...)
    ndims(g) == 2 || throw(ArgumentError("plotting requires a 2D grid"))
    indices = cellindices(g, points)
    ndims(indices) == 1 || throw(DimensionMismatch("plot one collection of points at a time"))
    positions = points isa AbstractVector ? [Makie.Point2d(p[1],p[2]) for p in points] :
                                          [Makie.Point2d(points[1,p],points[2,p]) for p in axes(points,2)]
    plot = Makie.scatter!(axis, positions; color, kwargs...)
    labels = [string(Tuple(i)) for i in indices]
    Makie.text!(axis, positions; text=labels, color=textcolor, fontsize, align=(:left,:bottom))
    return plot
end
end
