using Grids

grid = RegularGrid((10,10);extent=(1,1))
mktempdir() do directory
    path = joinpath(directory,"grid.toml")
    savegrid(path,grid)
    restored = loadgrid(path)
    @assert edges(grid) == edges(restored)
    println("Grid round trip succeeded: ",restored)
end
