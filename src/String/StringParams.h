#pragma once
#ifndef __METAL_VERSION__
#include "StringReal.h"
#include <cstdint>
using StringUInt = std::uint32_t;
#else
using StringUInt = uint;
#endif
struct StringParams {
    StringUInt Points, Components, Model, Split, Nonlinear, Reference;
    StringUInt Steps, Offset, RecordStates, ReceiverU, ReceiverV;
    StringReal H, InvH, InvH2, Dt, Dt2, InvDt, Mass, Rigidity, EA;
    StringReal Tension, LongitudinalTension, NonlinearModulus;
    StringReal Sigma0, Sigma1, InverseMass, Shift, Length;
};
