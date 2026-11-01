# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Tetrahedron clipping tables from the pinned DiffSound release.
TRIANGLE_TABLE = [
    [-1, -1, -1, -1, -1, -1],
    [1, 0, 2, -1, -1, -1],
    [4, 0, 3, -1, -1, -1],
    [1, 4, 2, 1, 3, 4],
    [3, 1, 5, -1, -1, -1],
    [2, 3, 0, 2, 5, 3],
    [1, 4, 0, 1, 5, 4],
    [4, 2, 5, -1, -1, -1],
    [4, 5, 2, -1, -1, -1],
    [4, 1, 0, 4, 5, 1],
    [3, 2, 0, 3, 5, 2],
    [1, 3, 5, -1, -1, -1],
    [4, 1, 2, 4, 3, 1],
    [3, 0, 4, -1, -1, -1],
    [2, 0, 1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1],
]

TET_TABLE = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 1, 8, 6, 5, 1, 7, 6, 5, 0, 1, 6],
    [2, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 0, 6, 7, 9, 0, 7, 6, 7, 0, 9, 2],
    [4, 1, 9, 8, 5, 1, 9, 4, 5, 1, 2, 9],
    [6, 0, 1, 2, 8, 6, 1, 2, 9, 6, 8, 2],
    [3, 6, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 4, 8, 5, 0, 8, 3, 5, 8, 9, 3],
    [1, 4, 7, 3, 4, 7, 6, 3, 9, 6, 7, 3],
    [0, 1, 5, 3, 5, 1, 9, 3, 5, 1, 7, 9],
    [5, 2, 3, 7, 3, 6, 5, 8, 3, 5, 7, 8],
    [0, 4, 7, 8, 0, 3, 8, 7, 0, 3, 7, 2],
    [4, 1, 2, 3, 4, 3, 2, 5, 4, 3, 5, 6],
    [0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1],
]
