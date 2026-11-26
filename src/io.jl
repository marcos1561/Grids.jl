function _grid_dict(g::AbstractGrid)
    T = eltype(g)
    T in (Float16, Float32, Float64) ||
        throw(ArgumentError("portable files support Float16, Float32, and Float64 coordinates"))
    data = Dict{String,Any}("format_version" => 2, "coordinate_type" => string(T))
    if g isa RegularGrid
        merge!(data, Dict("type" => "regular", "dims" => collect(size(g)), "spacing" => collect(g.spacing), "center" => gridcenter(g)))
    elseif g isa RectangularGrid
        merge!(data, Dict("type" => "rectangular", "edges" => collect(edges(g))))
    elseif g isa PolarGrid
        merge!(data, Dict("type" => "polar", "r_edges" => g.edges[1],
                         "theta_edges" => g.edges[2], "center" => collect(g.center)))
    else
        throw(ArgumentError("unsupported grid type $(typeof(g))"))
    end
    return data
end

function _dict_grid(data::AbstractDict)
    version = get(data, "format_version", 1)
    version in (1,2) || throw(ArgumentError("unsupported grid file version: $version"))
    name = get(data, "coordinate_type", "Float64")
    types = Dict("Float16"=>Float16, "Float32"=>Float32, "Float64"=>Float64)
    haskey(types, name) || throw(ArgumentError("unsupported coordinate type: $name"))
    T = types[name]
    kind = get(data, "type", nothing)
    if kind == "regular"
        return RegularGrid(Tuple(Int.(data["dims"])); spacing=Tuple(T.(data["spacing"])), center=SVector(T.(data["center"])...))
    elseif kind == "rectangular"
        return RectangularGrid(Tuple(T.(e) for e in data["edges"]))
    elseif kind == "polar"
        return PolarGrid(T.(data["r_edges"]), T.(data["theta_edges"]); center=Tuple(T.(data["center"])))
    end
    throw(ArgumentError("unknown grid type: $kind"))
end

function _file_format(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    ext == ".toml" && return :toml
    ext in (".yaml", ".yml") && return :yaml
    throw(ArgumentError("file extension must be .toml, .yaml, or .yml"))
end
function _check_format(format)
    format in (:toml,:yaml) || throw(ArgumentError("format must be :toml or :yaml"))
    format === :yaml && Base.get_extension(@__MODULE__, :GridsYAMLExt) === nothing &&
        throw(ArgumentError("load YAML.jl with `using YAML` to use YAML grid files"))
    return nothing
end
_writegrid(io, data, ::Val{:toml}) = TOML.print(io, data; sorted=true)
_readgrid(io, ::Val{:toml}) = TOML.parse(read(io, String))
_writegrid(io, data, ::Val{:yaml}) = throw(ArgumentError("load YAML.jl first"))
_readgrid(io, ::Val{:yaml}) = throw(ArgumentError("load YAML.jl first"))

"""
    savegrid(path, grid)
    savegrid(io, grid; format=:toml)

Write a versioned, plain-data grid description. TOML uses the standard library;
YAML is enabled by loading YAML.jl. Existing parent directories are required.
"""
function savegrid(io::IO, g::AbstractGrid; format::Symbol=:toml)
    _check_format(format)
    _writegrid(io, _grid_dict(g), Val(format))
    return nothing
end
function savegrid(path::AbstractString, g::AbstractGrid)
    format = _file_format(path)
    _check_format(format)
    data = _grid_dict(g) # validate before opening/truncating an existing file
    open(path, "w") do io
        _writegrid(io, data, Val(format))
    end
    return nothing
end

"""Read a grid from TOML or YAML, including version-1 files from the initial port."""
function loadgrid(io::IO; format::Symbol=:toml)
    _check_format(format)
    data = _readgrid(io, Val(format))
    data isa AbstractDict || throw(ArgumentError("grid file must contain a mapping"))
    return _dict_grid(data)
end
function loadgrid(path::AbstractString)
    format = _file_format(path)
    open(path) do io
        loadgrid(io; format)
    end
end
