#pragma once

#include "PorousMaterial.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

struct RoomScene;

namespace Porous {

struct InteriorNode {
    int Node, Material;
    uint32_t Adjacency;
};

struct InterfaceNode {
    int Node, Material;
    // Zero means a blocked rigid-wall link. Otherwise abs(index)-1 addresses
    // Prepared::Links; positive is outward from A and negative outward from B.
    std::array<int, 12> Links{};
};

struct VelocityLink {
    int A, B, Material; // Material indexes LinkInverseDensity, not Materials.
};

struct Prepared {
    // Air is material zero. Input porous labels are shifted by one.
    std::vector<Material> Materials;
    std::vector<InteriorNode> Interior;
    std::vector<InterfaceNode> Interface;
    std::vector<VelocityLink> Links;
    std::vector<Rational> LinkInverseDensity;
};

bool IsInterior(const RoomScene &, int node);
int UnfoldedY(const RoomScene &, int node);
std::array<int, 12> FccNeighbours(const RoomScene &, int node);

// Labels are -1 for air and material indices otherwise. Only retained adjacency
// links connect materials; a wall separating two labels is not a fluid interface.
std::shared_ptr<Prepared> Prepare(const RoomScene &, const std::vector<int> &labels, const std::vector<Material> &materials);

// Small FP64 diagnostic reference. The scene must outlive this object. Receiver
// output is step-major corner pressure, matching RoomGpu::Output's sampling time.
class Cpu {
public:
    void Init(const RoomScene &);
    void RunSteps(int count);
    const std::vector<double> &Pressure() const { return Current; }
    const std::vector<double> &Output() const { return ReceiverOutput; }

private:
    const RoomScene *Scene{};
    std::shared_ptr<const Prepared> Data;
    std::vector<uint16_t> Adjacency;
    std::vector<uint8_t> Boundary;
    std::vector<double> Current, Previous, Velocity, ReceiverOutput;
    std::vector<double> InteriorState, PressureState, VelocityState;
    std::vector<size_t> InteriorOffset, PressureOffset, VelocityOffset;
    int Step{};
};

} // namespace Porous
