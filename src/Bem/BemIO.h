#pragma once
#include "Bem.h"
#include "json.hpp"

namespace bem {
Problem ReadProblem(const nlohmann::json &);
nlohmann::json WriteResult(const Result &, bool includeFields = true);
} // namespace bem
