#pragma once

#include "RoomScene.h"
#include "json.hpp"

#include <array>
#include <filesystem>

namespace Porous {

// Physical coordinates of a stored FCC node, with origin at unfolded (0,0,0).
std::array<double, 3> Position(const RoomScene &, int node, const std::array<double, 3> &origin);

// Native FCC shoebox, cell-centred geometry, pressure source and receiver output.
RoomScene MakeRoom(const nlohmann::json &, const std::filesystem::path &directory = {});
void AddVolumes(RoomScene &, const nlohmann::json &, const std::filesystem::path &directory, const std::array<double, 3> &origin);

} // namespace Porous
