# DG regression inputs

Two curved rigid-cylinder meshes and independent FP64 WADG outputs support the
current solver's regression checks: the 70-element degree-6 unit cylinder and
560-element degree-4 radius-0.5 cylinder. Mesh inputs include geometry, connectivity,
and receiver interpolation. Initial pressure is generated from the analytic
cylinder mode at time zero, with zero velocity. Receiver positions and element
IDs are stored in `mesh.json`. `xyz.bin` stores FP64 coordinates as
`[element][node][x,y,z]`. `vmapP.bin` stores neighboring trace-node indices;
local face indices come from the reference nodes, and self-mapped faces are rigid
boundaries. The CPU records
retain terminal fields, complete receiver traces, clocks, and their original
Python generator provenance in [cpu/provenance.json](cpu/provenance.json), with
shared software/method information and case-specific input hashes. C++ reference
regeneration is checked against these frozen records.

`nodes/` contains the exact r/s/t coordinates extracted from DTU's node tables at
`f08c22f83fd64605a634b94326f07cb88f00b0ef`, with original/subset hashes and its MIT
license. Each row stores one node’s `r s t` coordinates; the files omit other libParanumal tables.
`sha256.json` verifies the retained data before operator preparation.

Each mesh has one `mesh.json` containing dimensions, material properties, array
layout, and generation provenance. Gmsh is not needed to use these fixtures.
Prepared operators and run outputs are generated under ignored `build/` directories.
See [solver instructions](../../../src/Dg/README.md) for preparation, validation,
CPU reference regeneration, and the matched CUDA comparison.
