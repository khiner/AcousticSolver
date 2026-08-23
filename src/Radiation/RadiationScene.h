#pragma once

// Scene runner for the SonicRadiation solver [Jin et al. 2025]: parses the same JSON
// config as the WaveBlender path and renders its listeners through the radiation solver
// instead of the ghost-cell FDTD stack.
//
// Scope is what the method covers: closed, rigid, modal bodies. Water, speakers, point
// impulses, occluders, and thin shells stay on the WaveBlender path.

#include <string>

// `seconds`, when positive, renders only that much of the scene rather than to its `tf`.
void RunRadiationScene(const std::string &config_file, double seconds = 0.);
