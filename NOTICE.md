# Third-party notices

AcousticSolver as a whole is distributed under the GNU General Public License v3.0 (see
`LICENSE`). Portions derive from the MIT-licensed projects listed below, whose copyright
and permission notices are reproduced here as those licenses require.

## Derived sources under `src/`

Ported from the projects listed, then restructured and optimized. None are
upstream-identical. Files under `src/` not listed here are original to this project.

| Files | Origin |
| --- | --- |
| `WaveBlender.{h,cpp}`, `Shaders.h`, `Shaders/*.cpp`, `FDTD/*` | [WaveBlender](https://github.com/kangruix/WaveBlender) |
| `FluidSound.{h,cpp}`, `Oscillator.{h,cpp}`, `BubbleUtils.{h,cpp}`, `Integrators.{h,cpp}` | [FluidSound](https://github.com/kangruix/FluidSound) |
| `ModalSound.{h,cpp}`, `ModeData.h` | ModalSound, based on [openpbso](https://github.com/jhwang7628/openpbso) |

### WaveBlender and FluidSound

MIT License

Copyright (c) 2024 Kangrui Xue

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### ModalSound

Copyright 2023 Jui-Hsien Wang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Libraries vendored under `src/ThirdParty/`

Each carries its full license text in-tree, at the location given.

| Component | License | Text | State |
| --- | --- | --- | --- |
| Eigen 3.4.0 | MPL-2.0, with some files BSD-3-Clause or Apache-2.0 | `Eigen/COPYING.{MPL2,BSD,APACHE}` | verbatim |
| [AudioFile](https://github.com/adamstark/AudioFile) — Adam Stark | MIT | header of `AudioFile.h` | verbatim |
| [JSON for Modern C++](https://github.com/nlohmann/json) — Niels Lohmann | MIT | header of `json.hpp` | verbatim |
| AABB-triangle overlap — Tomas Akenine-Möller | MIT | header of `tribox.h` | adapted: `float` → `REAL`, internal linkage |
