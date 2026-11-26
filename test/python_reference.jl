@testset "Original numerical fixtures, converted to Julia axis order" begin
    data = TOML.parsefile(joinpath(@__DIR__,"fixtures","python_reference.toml"))
    array(d,key) = reshape(d[key],Tuple(d[key*"_shape"]))
    points = permutedims(array(data,"points"),(3,2,1))
    values = permutedims(array(data,"values"),(3,4,2,1))
    lengths = data["end_id"]
    grids = (
        "regular" => RegularGrid((4,3);extent=(4,3),center=(0.25,-0.5)),
        "rectangular" => RectangularGrid(([-2,-1.75,0,1,3],[-3,-0.5,0,2])),
        "polar" => PolarGrid([0.25,0.75,1.5,2,3],[0.2,1,3.5,5.5];center=(0.25,-0.5)),
    )
    for (name,g) in grids
        ref = data[name]
        c = array(ref,"coords")
        expected_indices = [CartesianIndex(c[b,p,1],c[b,p,2]) for p in axes(c,2),b in axes(c,1)]
        indices = cellindices(g,points)
        @test indices == expected_indices
        for outside in (:ignore,:include)
            suffix = outside === :ignore ? "interior" : "full"
            counts = permutedims(array(ref,"count_"*suffix),(3,2,1))
            sums = permutedims(array(ref,"sum_"*suffix),(4,5,3,2,1))
            means = permutedims(array(ref,"mean_"*suffix),(4,5,3,2,1))
            if outside === :include
                # Source bins: interior, upper, lower. Native bins: lower,
                # interior, upper. This conversion stays entirely in the test.
                xorder = vcat(size(g,1)+2,collect(1:size(g,1)),size(g,1)+1)
                yorder = vcat(size(g,2)+2,collect(1:size(g,2)),size(g,2)+1)
                counts = counts[xorder,yorder,:]
                sums = sums[:,:,xorder,yorder,:]
                means = means[:,:,xorder,yorder,:]
            end
            @test cellcounts(g,indices;lengths,outside) == counts
            @test cellsum(g,values,indices;lengths,outside) ≈ sums
            @test cellmean(g,values,indices;lengths,outside) ≈ means
        end
    end
end
