//
//  FrameExtrapolationGeometry.swift
//
//  The picture geometry FrameExtrapolation needs to tell head rotation from scene motion: the
//  foveation tables for fxRotationField (FrameExtrapolation.metal) and the relative rotation of
//  two frames. Plain math without app state, so tools/frameflow can test it on the Mac too.
//

import simd

enum FrameExtrapolationGeometry {
    // Entries per foveation table.
    static let lutSize = 2048

    // One axis of the renderer's decompressAxisAlignedCoord (Shaders.metal): an undistorted eye
    // coordinate (0...1) to where the foveated frame holds it, before EYE_SIZE_RATIO. The two axes
    // are independent, which is what makes 1D tables enough.
    static func compressedCoordinate(_ e: Float, centerSize: Float, centerShift: Float, edgeRatio: Float) -> Float {
        let c0 = (1 - centerSize) * 0.5
        let c1 = (edgeRatio - 1) * c0 * (centerShift + 1) / edgeRatio
        let c2 = (edgeRatio - 1) * centerSize + 1
        let loBound = c0 * (centerShift + 1)
        let hiBound = c0 * (centerShift - 1) + 1
        var value: Float = 0
        if e < loBound {
            let loBoundC = c0 * (centerShift + 1) / c2
            let a = (c1 + c2 * loBoundC) / loBoundC
            value = (-a + (a * a + 4 * c2 * (1 - edgeRatio) / (edgeRatio * loBoundC) * e).squareRoot())
                / (2 * c2 * (1 - edgeRatio)) * (edgeRatio * loBoundC)
        } else if e > hiBound {
            let hiBoundC = c0 * (centerShift - 1) / c2 + 1
            let b = (c2 - edgeRatio * c1 - 2 * edgeRatio * c2 + c2 * edgeRatio * (1 - hiBoundC) + edgeRatio)
                / (edgeRatio * (1 - hiBoundC))
            let k = (c2 * edgeRatio - c2) * (c1 - hiBoundC + hiBoundC * c2) / (edgeRatio * (1 - hiBoundC) * (1 - hiBoundC))
                - e * (c2 * edgeRatio - c2) / (edgeRatio * (1 - hiBoundC))
            value = (-b + (b * b - 4 * k).squareRoot()) / (2 * c2 * (edgeRatio - 1)) * (edgeRatio * (1 - hiBoundC))
        } else if e > loBound && e < hiBound {
            value = (e - c1) * edgeRatio / c2
        }
        return value.isFinite ? min(max(value, 0), 1) : 0
    }

    // The four tables fxRotationField reads, each lutSize floats: x and y undistorted -> compressed,
    // then x and y compressed -> undistorted (the inverse, from the same monotone curve).
    static func foveationTables(centerSize: SIMD2<Float>, centerShift: SIMD2<Float>, edgeRatio: SIMD2<Float>) -> [Float] {
        let n = lutSize
        var tables = [Float](repeating: 0, count: 4 * n)
        for axis in 0..<2 {
            // Forward, sampled finely enough to invert by walking it.
            let fine = 8 * n
            var forward = [Float](repeating: 0, count: fine + 1)
            for i in 0...fine {
                forward[i] = compressedCoordinate(Float(i) / Float(fine), centerSize: centerSize[axis],
                                                  centerShift: centerShift[axis], edgeRatio: edgeRatio[axis])
            }
            for i in 0..<n {
                tables[axis * n + i] = compressedCoordinate(Float(i) / Float(n - 1), centerSize: centerSize[axis],
                                                            centerShift: centerShift[axis], edgeRatio: edgeRatio[axis])
            }
            var j = 0
            for i in 0..<n {
                let c = Float(i) / Float(n - 1)
                while j < fine - 1 && forward[j + 1] < c { j += 1 }
                let span = forward[j + 1] - forward[j]
                let f = span > 0 ? min(max((c - forward[j]) / span, 0), 1) : 0
                tables[(2 + axis) * n + i] = (Float(j) + f) / Float(fine)
            }
        }
        return tables
    }

    // Rotation taking directions in the newer frame's eye space into the older frame's: both
    // orientations are world-from-head, the eyes are not canted (the renderer strips the views'
    // rotation), so eye space is head space.
    static func relativeRotation(older: simd_quatf, newer: simd_quatf) -> simd_float3x3 {
        simd_float3x3(older.inverse * newer)
    }
}
