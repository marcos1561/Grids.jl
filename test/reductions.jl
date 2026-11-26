struct AdditiveValue
    x::Float64
end
Base.:+(a::AdditiveValue,b::AdditiveValue) = AdditiveValue(a.x+b.x)

@testset "Native reduction layout and exterior policy" begin
    g = RegularGrid((2,2);extent=(2,2))
    indices = CartesianIndex.([(1,1),(1,1),(2,1),(2,2),(0,1),(3,2),(1,0),(2,3)])
    values = collect(1:8)
    expected_count = [2 0;1 1]
    expected_sum = [3 0;3 4]
    expected_mean = [1.5 0;3 4]
    @test cellcounts(g,indices) == expected_count
    @test cellsum(g,values,indices) == expected_sum
    @test cellmean(g,values,indices) == expected_mean
    @test eltype(cellmean(g,values,indices)) === Float64
    @test cellcounts(g,indices)[indices[1]] == 2
    full_count = cellcounts(g,indices;outside=:include)
    full_sum = cellsum(g,values,indices;outside=:include)
    @test full_count[2:3,2:3] == expected_count
    @test full_sum[1,2] == 5 && full_sum[4,3] == 6
    @test full_sum[2,1] == 7 && full_sum[3,4] == 8
    @test sum(full_count) == 8 && sum(full_sum) == sum(values)
    @test_throws BoundsError cellcounts(g,indices;outside=:error)
    @test_throws BoundsError cellsum(g,values,indices;outside=:error)
    @test cellcounts!(zeros(Int,2,2),g,indices) == expected_count
    @test cellsum!(zeros(Int,2,2),g,values,indices) == expected_sum
    scratch = zeros(Int,2,2)
    out = zeros(2,2)
    @test cellmean!(out,g,values,indices;count_buffer=scratch) == expected_mean
    @test scratch == expected_count
    @test cellmean(g,values,indices;counts=scratch) == expected_mean
    @test scratch == expected_count
    @test cellcounts(g,indices;lengths=2) == [2 0;0 0]
    @test cellsum(g,values,indices;lengths=[2]) == [3 0;0 0]
    @test all(iszero,cellmean(g,values,indices;lengths=0))
    @test sum(cellsum(g,trues(8),indices;outside=:include)) == 8
    @test eltype(cellmean(g,Float32.(values),indices)) === Float32
    @test cellmean(g,complex.(values,values),indices) ≈ complex.(expected_mean,expected_mean)
    @test map(v -> v.x,cellsum(g,AdditiveValue.(values),indices;init=AdditiveValue(0))) == expected_sum

    # Tensor component axes precede spatial axes, including non-contiguous views.
    tensors = vcat(transpose(values),transpose(2values),transpose(3values))
    expected = cellsum(g,tensors,indices)
    storage = zeros(6,4,4)
    dest = @view storage[1:2:end,1:2:end,1:2:end]
    @test cellsum!(dest,g,tensors,indices) == expected
    @test cellsum(g,@view(tensors[:,1:2:end]),@view(indices[1:2:end])) ==
          cellsum(g,tensors[:,1:2:end],indices[1:2:end])

    @test size(cellcounts(g,CartesianIndex{2}[])) == (2,2)
    @test size(cellcounts(g,fill(CartesianIndex(1,1),3,0))) == (2,2,0)
    @test all(iszero,cellmean(g,Float64[],CartesianIndex{2}[]))
    @test size(cellsum(g,zeros(0,8),indices)) == (0,2,2)
    @test size(cellmean!(zeros(0,2,2),g,zeros(0,8),indices;count_buffer=scratch)) == (0,2,2)
    @test scratch == expected_count
    @test size(cellcounts(g,reshape(indices,8,1))) == (2,2,1)
    @test_throws DimensionMismatch cellsum(g,values[1:3],indices)
    @test_throws DimensionMismatch cellsum!(zeros(3,3),g,values,indices)
    @test_throws DimensionMismatch cellcounts(g,indices;lengths=[1,2])
    @test_throws ArgumentError cellcounts(g,indices;lengths=-1)
    @test_throws ArgumentError cellcounts(g,indices;lengths=9)
    @test_throws ArgumentError cellcounts(g,indices;outside=:invalid)
    @test_throws ArgumentError cellmean!(out,g,values,indices;counts=scratch,count_buffer=scratch)
    @test_throws ArgumentError cellmean(g,values,indices;counts=zeros(2,2))
    @test_throws ArgumentError cellmean(g,values,indices;counts=fill(-1,2,2))
    alias = ones(4)
    @test_throws ArgumentError cellsum!(reshape(alias,2,2),g,alias,fill(CartesianIndex(1,1),4))
end

# Independent reference uses direct Cartesian indexing and explicit tensor views,
# rather than the implementation's flattened offsets.
function reference_reduction(g, values, indices, lengths, outside, value_shape)
    spatial = outside === :include ? map(n -> n+2,size(g)) : size(g)
    counts = zeros(Int,spatial...,size(indices,2))
    sums = zeros(value_shape...,spatial...,size(indices,2))
    for b in axes(indices,2), p in 1:lengths[b]
        i = indices[p,b]
        inside = all(d -> 1 <= i[d] <= size(g,d),1:ndims(g))
        outside === :ignore && !inside && continue
        cell = outside === :include ? ntuple(d -> clamp(i[d]+1,1,size(g,d)+2),ndims(g)) : Tuple(i)
        counts[cell...,b] += 1
        for c in CartesianIndices(value_shape)
            sums[Tuple(c)...,cell...,b] += values[Tuple(c)...,p,b]
        end
    end
    means = copy(sums)
    for b in axes(indices,2), cell in CartesianIndices(spatial), c in CartesianIndices(value_shape)
        n = counts[Tuple(cell)...,b]
        n > 0 && (means[Tuple(c)...,Tuple(cell)...,b] /= n)
    end
    return counts,sums,means
end

@testset "Independent batched tensor reductions in 1D–4D" begin
    rng = MersenneTwister(12)
    for n in 1:4, value_shape in ((),(2,),(2,3))
        g = RegularGrid(ntuple(d -> d+1,n);extent=ntuple(_ -> 3.0,n))
        indices = [CartesianIndex(ntuple(d -> rand(rng,0:size(g,d)+1),n)) for p in 1:25,b in 1:3]
        values = randn(rng,value_shape...,25,3)
        values[ntuple(_ -> Colon(),length(value_shape))...,14:25,2] .= NaN
        values[ntuple(_ -> Colon(),length(value_shape))...,:,3] .= NaN
        lengths = [25,13,0]
        for outside in (:ignore,:include)
            counts,sums,means = reference_reduction(g,values,indices,lengths,outside,value_shape)
            @test cellcounts(g,indices;lengths,outside) == counts
            @test cellsum(g,values,indices;lengths,outside) ≈ sums
            @test cellmean(g,values,indices;lengths,outside) ≈ means
            single_values = copy(selectdim(values,ndims(values),1))
            @test cellsum(g,single_values,indices[:,1];outside) ≈ selectdim(sums,ndims(sums),1)
        end
    end
end
