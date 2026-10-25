#pragma once
#include "NC_TypeDefinition.h"
#include <fstream>

void NumCalcNativeSolve(std::ofstream &);
void NumCalcBeginAssembly();
void NumCalcResetOperator();
void NumCalcRequestDirectSolve();
void NumCalcMultiply(Vector<Complex> const &, Vector<Complex> const &);
