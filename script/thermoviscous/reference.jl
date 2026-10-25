# Run in the isolated environment created by script/ValidateThermoviscous.
using BoundaryIntegralEquations, LinearAlgebra, SparseArrays, JSON
const BIE = BoundaryIntegralEquations
out = abspath(ARGS[1])
mkpath(out)
meshname = length(ARGS) > 1 ? ARGS[2] : "sphere_1m_extremely_coarse"
freq = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 100.0
exterior = length(ARGS) <= 3 || ARGS[4] != "interior"
mesh = BIE.load3dTriangularComsolMesh(joinpath(@__DIR__, "../../external/thermoviscous-bem/examples/meshes", meshname))
n = size(mesh.sources, 2)
println("Reference: $n nodes, $(size(mesh.topology,2)) elements, $freq Hz")
rho,c,_,gamma,mu,_,cp,cv,lambda,beta = BIE.ambient_to_properties(freq)
config = Dict("vertices" => collect(eachcol(mesh.coordinates)), "triangles" => collect(eachcol(mesh.topology .- 1)),
    "frequency" => freq, "quadrature" => 3, "exterior" => exterior, "velocity" => [0,0,0.01], "loss_neighborhood" => "incident",
    "material" => Dict("density"=>rho, "sound_speed"=>c, "viscosity"=>mu, "bulk_viscosity"=>0.6mu,
        "heat_capacity_p"=>cp, "heat_capacity_v"=>cv, "conductivity"=>lambda))
open(io -> JSON.print(io, config), joinpath(out,"scene.json"), "w")
A = BIE.LossyGlobalOuter(mesh,freq;progress=false,exterior=exterior)
# The upstream free terms are interchanged for this outward mesh winding.
# Enforce the constant interior Laplace identity and orient the gradient into the fluid.
_,_,C = BIE.assemble_parallel!(mesh,0.0,mesh.sources;fOn=false,gOn=false,progress=false,m=3,n=3)
correction = exterior ? 1 .- 2C : 2C .- 1
for i in 1:n
    A.Ha[i,i] += correction[i]
    A.Hh[i,i] += correction[i]
    for axis in 0:2; A.Hv[axis*n+i,axis*n+i] += correction[i]; end
end
if !exterior; A.Nd .*= -1; end
# A.inner shares Hv and Nd with A.
data = Dict()
function save(name, value)
    value = ComplexF64.(Array(value))
    open(io -> write(io,value), joinpath(out,name*".bin"),"w")
    data[name] = Dict("rows"=>size(value,1),"cols"=>size(value,2))
end
for name in (:Ha,:Ga,:Hh,:Gh)
    save(string(name),getproperty(A,name))
end
save("Hv",A.Hv[1:n,1:n]); save("Gv",A.Gv[1:n,1:n])
save("Dc",A.Dc); save("Nd",A.Nd)
vs = vcat(zeros(2n),fill(0.01,n))
# Keep the dense 10n oracle bounded; the production solver uses the nested Schur system.
n <= 500 || error("Full-system reference is limited to 500 nodes")
rhs = vcat(zeros(7n),vs)
F = BIE._full10(A)
x = F \ rhs
save("fields",x)
println("Full-system relative residual: ", norm(F*x-rhs)/norm(rhs))
open(io -> JSON.print(io,data),joinpath(out,"operators.json"),"w")
println("Wrote ",out)
