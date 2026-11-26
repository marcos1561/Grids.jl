# Run with `julia --project=test examples/plotting.jl [output.png]`.
using Grids, GLMakie, Random

grid = RegularGrid((8,4);extent=(4,2))
polar = PolarGrid([0,0.5,1,1.5],range(0,2π;length=9))
figure = Figure(size=(1000,450))
for (i,g) in enumerate((grid,polar))
    axis = Axis(figure[1,i];aspect=DataAspect(),title=string(nameof(typeof(g))))
    plotgrid!(axis,g)
    plotindices!(axis,g,rand(MersenneTwister(i),g,12))
end

# path = isempty(ARGS) ? joinpath(tempdir(),"grids-example.png") : first(ARGS)
# save(path,figure)
# println("Saved ",path)

figure
