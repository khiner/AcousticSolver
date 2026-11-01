#pragma once
#ifndef __METAL_VERSION__
#include "Metal/FloatPair.h"
#include <cstdint>
using StringUInt = std::uint32_t;
#else
using StringUInt = uint;
#endif
struct StringParams {
    StringUInt Points, Components, Model, Split, Nonlinear, Reference;
    StringUInt Steps, Offset, RecordStates, ReceiverU, ReceiverV;
    FloatPair H, InvH, InvH2, Dt, Dt2, InvDt, Mass, Rigidity, EA;
    FloatPair Tension, LongitudinalTension, NonlinearModulus;
    FloatPair Sigma0, Sigma1, InverseMass, Shift, Length;
};
