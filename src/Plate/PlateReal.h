#pragma once

// Binary floating point with a 256-bit significand and a separate exponent.
// Integer limbs avoid FP32 underflow of the low components. Operations truncate
// below the significand; preparation uses MPFR and never passes through double.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#define PLATE_THREAD thread
using PlateWord = uint;
using PlateWide = ulong;
inline uint PlateClz(uint v) { return metal::clz(v); }
inline uint PlateBits(float v) { return as_type<uint>(v); }
#else
#include <bit>
#include <cmath>
#include <cstdint>
#define PLATE_THREAD
using PlateWord = std::uint32_t;
using PlateWide = std::uint64_t;
inline unsigned PlateClz(PlateWord v) { return std::countl_zero(v); }
inline PlateWord PlateBits(float v) { return std::bit_cast<PlateWord>(v); }
#endif

struct PlateReal {
    enum { Limbs = 8 };
    PlateWord Word[Limbs]{}; // Little endian; nonzero values have bit 255 set.
    int Exponent{}, Sign{}; // value = Sign * significand * 2^(Exponent-255).
    PlateReal() = default;
    PlateReal(float value) {
        const PlateWord bits = PlateBits(value), fraction = bits & 0x7fffffu;
        const int exponent = int((bits >> 23) & 255u);
        if (!exponent && !fraction) return;
        Sign = (bits >> 31) ? -1 : 1;
        if (exponent == 255) {
            Exponent = 0x7fffffff;
            return;
        }
        PlateWord const mantissa = fraction | (exponent ? 0x800000u : 0u);
        const unsigned shift = PlateClz(mantissa);
        Word[Limbs - 1] = mantissa << shift;
        Exponent = (exponent ? exponent - 127 : -126) + 8 - int(shift);
    }
};

inline PlateReal PlateNormalize(PlateReal a) {
    int top = PlateReal::Limbs - 1;
    while (top >= 0 && !a.Word[top]) --top;
    if (top < 0) return {};
    const unsigned shift = PlateClz(a.Word[top]);
    PlateReal out;
    out.Sign = a.Sign;
    out.Exponent = a.Exponent - 32 * (PlateReal::Limbs - 1 - top) - int(shift);
    for (int i = PlateReal::Limbs - 1; i >= 0; --i) {
        int const source = i - (PlateReal::Limbs - 1 - top);
        if (source >= 0) {
            out.Word[i] = a.Word[source] << shift;
            if (shift && source > 0) out.Word[i] |= a.Word[source - 1] >> (32 - shift);
        }
    }
    return out;
}
inline PlateReal PlateShift(PlateReal a, unsigned bits) {
    PlateReal out;
    if (bits >= 32 * PlateReal::Limbs) return out;
    unsigned words = bits / 32, shift = bits % 32;
    for (unsigned i = 0; i < PlateReal::Limbs - words; ++i) {
        out.Word[i] = a.Word[i + words] >> shift;
        if (shift && i + words + 1 < PlateReal::Limbs) out.Word[i] |= a.Word[i + words + 1] << (32 - shift);
    }
    return out;
}
inline int PlateCompareMagnitude(PlateReal a, PlateReal b) {
    if (a.Exponent != b.Exponent) return a.Exponent > b.Exponent ? 1 : -1;
    for (int i = PlateReal::Limbs - 1; i >= 0; --i)
        if (a.Word[i] != b.Word[i]) return a.Word[i] > b.Word[i] ? 1 : -1;
    return 0;
}
inline PlateReal operator-(PlateReal a) {
    a.Sign = -a.Sign;
    return a;
}
inline PlateReal operator+(PlateReal a, PlateReal b) {
    if (!a.Sign) return b;
    if (!b.Sign) return a;
    if (PlateCompareMagnitude(a, b) < 0) {
        PlateReal const t = a;
        a = b;
        b = t;
    }
    const PlateReal aligned = PlateShift(b, unsigned(a.Exponent - b.Exponent));
    PlateReal out;
    out.Sign = a.Sign;
    out.Exponent = a.Exponent;
    PlateWide carry = 0;
    if (a.Sign == b.Sign) {
        for (int i = 0; i < PlateReal::Limbs; ++i) {
            const PlateWide sum = PlateWide(a.Word[i]) + aligned.Word[i] + carry;
            out.Word[i] = PlateWord(sum);
            carry = sum >> 32;
        }
        if (carry) {
            PlateReal shifted = PlateShift(out, 1);
            shifted.Word[PlateReal::Limbs - 1] |= 0x80000000u;
            shifted.Sign = out.Sign;
            shifted.Exponent = out.Exponent + 1;
            return shifted;
        }
        return out;
    }
    for (int i = 0; i < PlateReal::Limbs; ++i) {
        const PlateWide sub = PlateWide(aligned.Word[i]) + carry;
        out.Word[i] = PlateWord(PlateWide(a.Word[i]) - sub);
        carry = PlateWide(a.Word[i]) < sub;
    }
    return PlateNormalize(out);
}
inline PlateReal operator-(PlateReal a, PlateReal b) { return a + (-b); }
inline PlateReal operator*(PlateReal a, PlateReal b) {
    if (!a.Sign || !b.Sign) return {};
    PlateWord product[2 * PlateReal::Limbs]{};
    for (int i = 0; i < PlateReal::Limbs; ++i) {
        PlateWide carry = 0;
        for (int j = 0; j < PlateReal::Limbs; ++j) {
            const PlateWide value = PlateWide(a.Word[i]) * b.Word[j] + product[i + j] + carry;
            product[i + j] = PlateWord(value);
            carry = value >> 32;
        }
        product[i + PlateReal::Limbs] = PlateWord(carry);
    }
    PlateReal out;
    out.Sign = a.Sign * b.Sign;
    const bool top = (product[2 * PlateReal::Limbs - 1] & 0x80000000u) != 0;
    out.Exponent = a.Exponent + b.Exponent + int(top);
    for (int i = 0; i < PlateReal::Limbs; ++i)
        out.Word[i] = top ? product[i + PlateReal::Limbs] :
                            (product[i + PlateReal::Limbs] << 1) | (product[i + PlateReal::Limbs - 1] >> 31);
    return out;
}
inline float PlateFloat(PlateReal a) {
#ifdef __METAL_VERSION__
    return a.Sign ? metal::ldexp(float(a.Word[PlateReal::Limbs - 1]) * float(a.Sign), a.Exponent - 31) : 0.f;
#else
    return a.Sign ? std::ldexp(float(a.Word[PlateReal::Limbs - 1]) * float(a.Sign), a.Exponent - 31) : 0.f;
#endif
}
inline PlateReal operator/(PlateReal a, PlateReal b) {
    if (!b.Sign) {
        PlateReal out;
        out.Sign = 1;
        out.Exponent = 0x7fffffff;
        return out;
    }
    if (!a.Sign) return {};
    const int exponent = b.Exponent;
    b.Exponent = 0;
    PlateReal inverse(1.f / PlateFloat(b));
    for (int i = 0; i < 4; ++i) inverse = inverse * (PlateReal(2.f) - b * inverse);
    PlateReal result = a * inverse;
    result.Exponent -= exponent;
    return result;
}
inline PLATE_THREAD PlateReal &operator+=(PLATE_THREAD PlateReal &a, PlateReal b) { return a = a + b; }
inline PLATE_THREAD PlateReal &operator-=(PLATE_THREAD PlateReal &a, PlateReal b) { return a = a - b; }
inline PLATE_THREAD PlateReal &operator*=(PLATE_THREAD PlateReal &a, PlateReal b) { return a = a * b; }
inline PLATE_THREAD PlateReal &operator/=(PLATE_THREAD PlateReal &a, PlateReal b) { return a = a / b; }
inline bool operator==(PlateReal a, PlateReal b) { return a.Sign == b.Sign && (!a.Sign || !PlateCompareMagnitude(a, b)); }
inline bool operator!=(PlateReal a, PlateReal b) { return !(a == b); }
inline bool operator<(PlateReal a, PlateReal b) { return a.Sign != b.Sign ? a.Sign < b.Sign : (a.Sign && a.Sign * PlateCompareMagnitude(a, b) < 0); }
inline bool operator>(PlateReal a, PlateReal b) { return b < a; }
inline bool operator<=(PlateReal a, PlateReal b) { return !(b < a); }
inline bool operator>=(PlateReal a, PlateReal b) { return !(a < b); }
inline PlateReal PlateAbs(PlateReal a) {
    if (a.Sign < 0) a.Sign = 1;
    return a;
}
inline PlateReal PlateSign(PlateReal a) { return {float(a.Sign)}; }
inline PlateReal PlateSqrt(PlateReal a) {
    if (!a.Sign) return {};
    const int exponent = a.Exponent & ~1;
    a.Exponent -= exponent;
#ifdef __METAL_VERSION__
    PlateReal inverse(1.f / metal::sqrt(PlateFloat(a)));
#else
    PlateReal inverse(1.f / std::sqrt(PlateFloat(a)));
#endif
    for (int i = 0; i < 4; ++i) inverse = PlateReal(.5f) * inverse * (PlateReal(3.f) - a * inverse * inverse);
    PlateReal root = a * inverse;
    root.Exponent += exponent / 2;
    return root;
}
#ifndef __METAL_VERSION__
inline PlateReal MakeReal(double value) {
    if (!value) return {};
    const auto bits = std::bit_cast<std::uint64_t>(value);
    int const exponent = int((bits >> 52) & 2047u);
    std::uint64_t mantissa = (bits & 0xfffffffffffffull) | (exponent ? 0x10000000000000ull : 0ull);
    PlateReal out;
    out.Sign = value < 0 ? -1 : 1;
    if (exponent == 2047) {
        out.Exponent = 0x7fffffff;
        return out;
    }
    const unsigned shift = std::countl_zero(mantissa);
    mantissa <<= shift;
    out.Word[PlateReal::Limbs - 1] = PlateWord(mantissa >> 32);
    out.Word[PlateReal::Limbs - 2] = PlateWord(mantissa);
    out.Exponent = (exponent ? exponent - 1023 : -1022) + 11 - int(shift);
    return out;
}
inline double ToDouble(PlateReal value) {
    if (!value.Sign) return 0;
    if (value.Exponent == 0x7fffffff) return INFINITY;
    double result = 0;
    for (int i = 0; i < PlateReal::Limbs; ++i) result = std::ldexp(result, -32) + double(value.Word[i]);
    return double(value.Sign) * std::ldexp(result, value.Exponent - 31);
}
static_assert(sizeof(PlateReal) == 40);
#endif
#undef PLATE_THREAD
