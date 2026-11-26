module GridsYAMLExt
using Grids
import YAML

Grids._writegrid(io::IO, data::AbstractDict, ::Val{:yaml}) = YAML.write(io, data)
Grids._readgrid(io::IO, ::Val{:yaml}) = YAML.load(io)
end
