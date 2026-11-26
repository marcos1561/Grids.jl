function hot_path_allocations(g,p,indices,values,counts,sums,means)
    cellindices!(indices,g,p)
    cellcounts!(counts,g,indices)
    cellsum!(sums,g,values,indices)
    cellmean!(means,g,values,indices;count_buffer=counts)
    return (
        @allocated(cellindices!(indices,g,p)),
        @allocated(cellcounts!(counts,g,indices)),
        @allocated(cellsum!(sums,g,values,indices)),
        @allocated(cellmean!(means,g,values,indices;count_buffer=counts)),
    )
end

@testset "Reusable buffers do not allocate per point" begin
    g = RegularGrid((40,30);extent=(4,3))
    measurements = map((100,10_000)) do n
        p = rand(MersenneTwister(1),g,n)
        indices = cellindices(g,p)
        values = ones(3,n)
        counts = zeros(Int,size(g))
        sums = zeros(3,size(g)...)
        hot_path_allocations(g,p,indices,values,counts,sums,similar(sums))
    end
    for operation in 1:4
        @test measurements[2][operation] <= measurements[1][operation]+256
        @test measurements[2][operation] <= 1024
    end
end
