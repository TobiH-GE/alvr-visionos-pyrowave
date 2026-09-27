//
//  IbexModel.swift
//
//  Physical geometry of the Steam Controller (2026), in its own model frame.
//
//  Measured off **Valve's published mechanical CAD** — the `IBEX_SOLID` STL in
//  the SteamController repo, which is the external shell of the real product
//  under a Creative Commons licence. That supersedes the earlier numbers taken
//  from the `ibex_ev1` SteamVR rendermodel, which is a body shell with no
//  sticks, pads or buttons modelled at all.
//
//  The two are in **the same frame**, which is what made the swap free: X spans
//  ±79.4mm against the rendermodel's ±79.1, Z runs -5.8..105.0 against
//  -3.9..105.0. The one real difference is Y, where the CAD reaches +8.16mm and
//  the rendermodel stops dead at 0 — that gap is the thumbsticks.
//
//  Component pose constants still come from `ibex_ev1.json`; the CAD has no
//  named frames.
//
//  Why a separate file: `getTipAdjustments(_:)` in WorldTracker keys every pose
//  offset off `HandAnchor.Chirality`, which cannot describe this device at all.
//  The Steam Controller is *one rigid body held by two hands* — its offsets are
//  properties of the object, not of a hand.
//

import Foundation
import simd

/// Model-space geometry for the Steam Controller (2026).
///
/// # Frame
///
/// Valve's rendermodel frame, which is also ARKit/OpenXR's: **+X right, +Y up
/// out of the button face, +Z toward the user**. The origin sits at the centre
/// of the top face, at its frontmost edge.
///
/// This is deliberately the same frame `SteamControllerDevice` rotates its IMU
/// into (the `(x, z, -y)` permutation), so gyro and accel samples are already
/// expressed in these axes and must not be rotated again. That equivalence
/// assumes the IMU is mounted axis-aligned with the shell, which is the one
/// thing here that is an assumption rather than a measurement — it is also
/// exactly what the wireframe overlay makes visible.
enum IbexModel {

    // MARK: - Extents

    /// CAD bounding box, metres. 158.8mm wide x 64.7mm tall x 110.8mm deep.
    /// The +Y extent is the thumbstick caps, which stand 8.16mm above the top
    /// face at y = 0.
    static let boundsMin = simd_float3(-0.07938, -0.05651, -0.00578)
    static let boundsMax = simd_float3( 0.07938,  0.00816,  0.10498)

    // MARK: - Hand anchors

    /// Where a wrapped palm sits on each handle, metres.
    ///
    /// Mid-point of the handle cross-section over z = 70..94mm, the span a hand
    /// actually covers, taken clear of the body (|x| > 40mm) so the section is
    /// the handle alone.
    ///
    /// **This corrects a real error.** The previous value of ±66mm came from a
    /// vertex *mean* over the rendermodel, and that mesh tessellates its curved
    /// outer wall far more finely than the flat inner one — so the mean was
    /// dragged outboard by about 6mm per side. The y and z terms survived the
    /// move to CAD nearly untouched (-36 → -35.2, 80 → 82); only x was wrong.
    static let leftGripAnchor  = simd_float3(-0.0599, -0.0352, 0.0820)
    static let rightGripAnchor = simd_float3( 0.0599, -0.0352, 0.0820)

    /// Distance between the two palms when both hands are on the device, metres.
    ///
    /// This is the single most useful number for fusion: it is a *known scalar*
    /// relating two independently-tracked hand anchors, so a two-handed grip
    /// pins the device's yaw outright instead of leaving it to drift. Which
    /// also means the old 132mm figure was not a harmless 12mm of slop — yaw is
    /// solved from this, so it was a systematic scale error in the one
    /// measurement the filter leans on hardest.
    static var gripSeparation: Float { rightGripAnchor.x - leftGripAnchor.x }

    // MARK: - Controls
    //
    // All area-weighted centroids off the CAD. These are what `ComponentProbe`
    // now checks itself against, rather than trying to discover from scratch.

    /// Thumbstick cap centres — the dished top surface the thumb rests in.
    /// Cap diameter 16.03mm. The x term is exactly ±23.000mm in the CAD, which
    /// is a designed number rather than a measurement artefact.
    static let leftStickCentre  = simd_float3(-0.0230, 0.00772, 0.03154)
    static let rightStickCentre = simd_float3( 0.0230, 0.00772, 0.03154)

    /// Trackpad centres.
    static let leftPadCentre  = simd_float3(-0.03038, -0.00705, 0.06190)
    static let rightPadCentre = simd_float3( 0.03038, -0.00705, 0.06190)

    /// Outward normal of each pad.
    ///
    /// The pads are **flat**, not domed — an earlier note here had that wrong.
    /// Checked properly: a plane fit over 22250 triangles per pad comes back
    /// with 0.000mm RMS deviation and 0.016mm worst case, which is as planar as
    /// a mesh gets. What this normal encodes is a genuine **15.65° tilt** of
    /// that flat plane — back toward the user and slightly outboard — not
    /// curvature.
    static let leftPadNormal  = simd_float3(-0.0785, 0.9629, 0.2580)
    static let rightPadNormal = simd_float3( 0.0785, 0.9629, 0.2580)

    /// In-plane axes of each pad. `u` runs across the device toward +X, `v` runs
    /// up the pad away from the user, matching `TouchpadSample`'s bottom-left
    /// origin.
    ///
    /// **The pads are toed in by ±9.05°**, mirrored, which is a designed
    /// ergonomic rotation and not noise — the two sides agree to a hundredth of
    /// a degree. An earlier version of this file derived `u` by projecting model
    /// +X into the pad plane, which silently assumes exactly the thing that is
    /// not true. The error is worth ~2.6mm at the pad edge, and since a pad
    /// touch now feeds the orientation solve over a ~36mm palm-to-pad baseline,
    /// it turns straight into several degrees of yaw bias.
    ///
    /// Measured by flood-filling the pad face inside its groove, in the pad's
    /// own plane, and fitting a minimum-area rectangle.
    static let leftPadAxisU  = simd_float3( 0.9845,  0.0342,  0.1720)
    static let rightPadAxisU = simd_float3( 0.9845, -0.0342, -0.1720)
    static let leftPadAxisV  = simd_float3( 0.1568,  0.2675, -0.9507)
    static let rightPadAxisV = simd_float3(-0.1568,  0.2675, -0.9507)

    /// Half-extents of the touch area along `u` and `v`, metres.
    ///
    /// The pads are **square**, 33.5 x 33.5mm. An earlier 37 x 32mm here came
    /// from measuring the outline in the *projected* top view, which the pads'
    /// 15.65° plane tilt distorts; measured in their own plane they come out
    /// square to within 0.02mm.
    ///
    /// 1.0 reaches the very corner — the whole surface is addressable — so no
    /// margin is subtracted here.
    static let padHalfExtents = SIMD2<Float>(0.01673, 0.01673)

    /// Maps a `TouchpadSample.position` onto the pad, in model space.
    ///
    /// Input is 0...1 with the origin at bottom-left, which is the convention
    /// `TouchpadSample` already uses. Worth having because a thumb on a pad is
    /// a *continuously varying* known point rather than a single landmark — a
    /// far richer anchor than the pad centre alone, and unlike the sticks it
    /// comes with the contact coordinate already reported by the device.
    static func padPoint(isLeft: Bool, position: SIMD2<Float>) -> simd_float3 {
        let u = (isLeft ? leftPadAxisU : rightPadAxisU) * ((position.x - 0.5) * 2 * padHalfExtents.x)
        let v = (isLeft ? leftPadAxisV : rightPadAxisV) * ((position.y - 0.5) * 2 * padHalfExtents.y)
        return (isLeft ? leftPadCentre : rightPadCentre) + u + v
    }

    /// How far the stick caps stand above the top face, metres. Worth naming:
    /// this is the number `mountCorrection.y` was originally standing in for.
    static let stickHeightAboveTopFace: Float = 0.00816

    /// Centre of each grip's capacitive sense electrode.
    ///
    /// Located by a construction the user supplied from the physical device:
    /// follow the contour of the back surface out from the centre-bottom screw
    /// toward a grip, and it runs through the lower paddle (L5/R5) and on to
    /// meet the line joining that grip's two screws. Where those cross is the
    /// sensor.
    ///
    /// Worked through against the CAD: the centre-bottom screw is at
    /// (0.0, -21.5, 76.1), the lower paddle pocket centroid at (±37.8, -28.7,
    /// 74.1), and the grip screws at (±60.8, -15.1, 31.3) and (±66.4, -42.5,
    /// 88.7). Intersecting in the back view puts the sensor at x = ±64.84,
    /// z = 72.71.
    ///
    /// The y is the part worth explaining. Interpolating along the *straight*
    /// line between the two grip screws gives -34.9, but that point sits 14mm
    /// inside the shell — the surface bulges well below the chord. Taking the
    /// actual outward-and-downward facing surface at that (x, z) instead gives
    /// **-54.6**, on the belly of the handle. That is where a palm rests, which
    /// is what a grip electrode is there to detect, so the discrepancy is the
    /// construction working rather than failing.
    ///
    /// Nothing in the pose path consumes this yet. It is a *contact* point, and
    /// what the filter needs is where ARKit reports a palm *joint*, which is a
    /// different thing by roughly the thickness of a hand. The debug panel shows
    /// the offset between this and the learned per-hand anchor so that
    /// difference can be measured rather than guessed.
    static let leftGripSensor  = simd_float3(-0.0648, -0.0546, 0.0727)
    static let rightGripSensor = simd_float3( 0.0648, -0.0546, 0.0727)

    static func gripSensor(isLeft: Bool) -> simd_float3 {
        isLeft ? leftGripSensor : rightGripSensor
    }

    // MARK: - Back-surface landmarks
    //
    // Kept because they are the *inputs* to the grip-sensor construction above,
    // and a construction is only as good as the points it was built from. Drawn
    // in the debug overlay so a mislocated screw or paddle shows up as a marker
    // floating off the real thing, rather than as a quietly wrong sensor.

    /// Back paddle pocket centroids. 1 is the upper pair, 2 the lower, matching
    /// `ButtonSet`'s L4/L5 and R4/R5 naming.
    static let leftPaddle1  = simd_float3(-0.03711, -0.03203, 0.05314)
    static let rightPaddle1 = simd_float3( 0.03713, -0.03205, 0.05314)
    static let leftPaddle2  = simd_float3(-0.03772, -0.02878, 0.07416)
    static let rightPaddle2 = simd_float3( 0.03778, -0.02872, 0.07414)

    /// All seven M2x9 screws, found as circular recesses in the back surface
    /// and matching the layout on the drawing: two per grip, two by the pogo
    /// connector, one at the bottom.
    static let screws: [simd_float3] = [
        simd_float3(-0.02109, -0.01963, 0.01092),   // by the pogo connector
        simd_float3( 0.02089, -0.01946, 0.01094),
        simd_float3(-0.06120, -0.01524, 0.03128),   // grip, top
        simd_float3( 0.06077, -0.01514, 0.03128),
        simd_float3(-0.06654, -0.04253, 0.08874),   // grip, bottom
        simd_float3( 0.06642, -0.04253, 0.08874),
        simd_float3(-0.00005, -0.02153, 0.07614),   // bottom centre
    ]

    /// Everything worth drawing, with the marker size to draw it at.
    ///
    /// Size is the only channel available for distinguishing them — the debug
    /// gadget's colours are fixed per axis — so it encodes how much each point
    /// is load-bearing: the anchors that actually solve position are largest,
    /// the raw landmarks that fed a construction are smallest.
    static var landmarks: [(name: String, position: simd_float3, size: Float)] {
        var out: [(String, simd_float3, Float)] = []
        for isLeft in [true, false] {
            let s = isLeft ? "L" : "R"
            out.append(("\(s) grip anchor", mountedGripAnchor(isLeft: isLeft), 0.030))
            out.append(("\(s) grip sensor", gripSensor(isLeft: isLeft), 0.020))
            out.append(("\(s) stick", isLeft ? leftStickCentre : rightStickCentre, 0.012))
            out.append(("\(s) pad", isLeft ? leftPadCentre : rightPadCentre, 0.012))
            // The four corners, so the rectangle is checkable and not just its
            // midpoint. A pad that is the right size in the wrong place and one
            // that is the wrong size look identical from the centre alone.
            for corner in [SIMD2<Float>(0, 0), .init(1, 0), .init(0, 1), .init(1, 1)] {
                out.append(("\(s) pad corner", padPoint(isLeft: isLeft, position: corner), 0.006))
            }
            out.append(("\(s) paddle1", isLeft ? leftPaddle1 : rightPaddle1, 0.008))
            out.append(("\(s) paddle2", isLeft ? leftPaddle2 : rightPaddle2, 0.008))
        }
        for screw in screws { out.append(("screw", screw, 0.005)) }
        return out
    }

    /// Midpoint of the two grip anchors — the natural centre of rotation when
    /// the device is held in both hands.
    static var gripCentre: simd_float3 { (leftGripAnchor + rightGripAnchor) * 0.5 }

    static func gripAnchor(isLeft: Bool) -> simd_float3 {
        isLeft ? leftGripAnchor : rightGripAnchor
    }

    // MARK: - Mount correction

    /// Where the body actually sits relative to the palms, over and above the
    /// measured anchors. **Tuned by eye against hardware, not measured** —
    /// deliberately kept as its own constant so it never gets confused with the
    /// numbers above, which came off the mesh.
    ///
    /// Both terms are about a centimetre, and the CAD has since given them a
    /// better explanation than the one they were tuned under.
    ///
    /// What this is really capturing is that the ARKit palm joint — the
    /// midpoint of middle metacarpal and knuckle — does **not** sit at the
    /// geometric centre of a grasped handle. It sits toward the palm side of
    /// it. That offset is a property of how a hand wraps a 39mm handle, not of
    /// the device, which is exactly why it does not come out of any mesh.
    ///
    /// The reassuring part: carrying the tuned value across to the CAD anchors
    /// moves the resulting body placement by **0.8mm in y and 2mm in z**. Two
    /// independent routes — one measured, one eyeballed — landing that close is
    /// the best evidence available that neither is badly wrong.
    ///
    /// Note the original justification for the `y` term (compensating for the
    /// rendermodel's missing thumbsticks) no longer applies: the CAD wireframe
    /// draws the sticks, and their true height is
    /// `stickHeightAboveTopFace` = 8.16mm. The number stayed because it is doing
    /// the palm-offset job above, not because that reasoning survived.
    ///
    /// Applied as a shift of the *body*, so `mountedGripAnchor` is the anchor
    /// that actually solves position: `position = palm - q.act(mountedAnchor)`.
    static let mountCorrection = simd_float3(0.0, -0.010, 0.010)

    static func mountedGripAnchor(isLeft: Bool) -> simd_float3 {
        gripAnchor(isLeft: isLeft) - mountCorrection
    }

    static var mountedGripCentre: simd_float3 { gripCentre - mountCorrection }

    // MARK: - SteamVR component poses
    //
    // Verbatim from ibex_ev1.json. `rotate_xyz` is degrees, applied XYZ.

    /// `handgrip` — SteamVR's "where the hand is" pose for the whole unit.
    static let handgripOrigin = simd_float3(0.003851, 0.003715, 0.075948)
    static let handgripRotationDegrees = simd_float3(15.392, 2.071, -0.303)

    /// `grip` — the OpenXR grip pose. Note it sits at z = 0.13, *behind* the
    /// mesh (which ends at z = 0.105); that is Valve's data, not a mistake here.
    static let gripOrigin = simd_float3(0.0, -0.015, 0.13)
    static let gripRotationDegrees = simd_float3(15.392, 2.071, -0.303)

    /// `tip` — aim pose. Origin is the model origin; only the pitch differs.
    static let tipOrigin = simd_float3(0.0, 0.0, 0.0)
    static let tipRotationDegrees = simd_float3(-10.0, 0.0, 0.0)

    /// Builds a rotation from a rendermodel `rotate_xyz` triple (degrees, XYZ).
    static func rotation(degrees d: simd_float3) -> simd_quatf {
        let r = d * (.pi / 180.0)
        let qx = simd_quatf(angle: r.x, axis: simd_float3(1, 0, 0))
        let qy = simd_quatf(angle: r.y, axis: simd_float3(0, 1, 0))
        let qz = simd_quatf(angle: r.z, axis: simd_float3(0, 0, 1))
        return qx * qy * qz
    }

    /// A component's transform in model space.
    static func componentTransform(origin: simd_float3, rotationDegrees: simd_float3) -> simd_float4x4 {
        var m = simd_float4x4(rotation(degrees: rotationDegrees))
        m.columns.3 = simd_float4(origin, 1)
        return m
    }

    static var handgripTransform: simd_float4x4 {
        componentTransform(origin: handgripOrigin, rotationDegrees: handgripRotationDegrees)
    }
    static var gripTransform: simd_float4x4 {
        componentTransform(origin: gripOrigin, rotationDegrees: gripRotationDegrees)
    }
    static var tipTransform: simd_float4x4 {
        componentTransform(origin: tipOrigin, rotationDegrees: tipRotationDegrees)
    }

    // MARK: - Wireframe

    /// Decimated body mesh: a non-indexed triangle list, **flat xyz floats**.
    ///
    /// Flat rather than `[simd_float3]` on purpose. The renderer's vertex
    /// descriptor declares position as a packed `float3` with stride 12, while
    /// Swift's `simd_float3` strides 16 — uploading an array of those directly
    /// would misread every vertex after the first, and would do it silently.
    ///
    /// Non-indexed for the same kind of reason: the debug draw path uses
    /// `drawPrimitives` with no index buffer, so this drops in beside the basis
    /// gadgets with no new pipeline state.
    ///
    /// Decimated from the CAD's 1577352 triangles to 10881 by vertex
    /// clustering, at **two cell sizes**: 2mm above y = -14mm and 8mm below.
    /// A single 5mm cell — which is what a uniform decimation to this triangle
    /// count needs — merges the 8mm-tall stick caps straight back into the top
    /// face, losing the one feature the overlay is most useful for. The body
    /// below the control face is a smooth shell and carries no such detail, so
    /// it can be spent coarsely.
    ///
    /// Loaded once, lazily — nothing pays for it unless something asks to draw
    /// the controller.
    static let wireframeVertices: [Float] = loadWireframe()

    /// Vertex count, i.e. `wireframeVertices.count / 3`.
    static var wireframeVertexCount: Int { wireframeVertices.count / 3 }

    /// Deduplicated shell vertices, for nearest-surface queries.
    private static let shellPoints: [simd_float3] = {
        var seen = Set<SIMD3<Int32>>()
        var out: [simd_float3] = []
        let v = wireframeVertices
        for i in stride(from: 0, to: v.count, by: 3) {
            let p = simd_float3(v[i], v[i + 1], v[i + 2])
            // 1mm buckets; the mesh is 6mm-decimated so this only removes the
            // exact duplicates that a non-indexed triangle list is full of.
            let key = SIMD3<Int32>(Int32(p.x * 1000), Int32(p.y * 1000), Int32(p.z * 1000))
            if seen.insert(key).inserted { out.append(p) }
        }
        return out
    }()

    /// Distance from `p` to the nearest point on the shell, metres.
    ///
    /// Nearest *vertex*, not nearest surface — the mesh is decimated at 6mm, so
    /// this reads a few mm high on flat spans. That is fine for what it is used
    /// for: telling apart "this fingertip landed on the controller" from "this
    /// fingertip landed 3cm inside it", which is the check that says whether a
    /// grip anchor is right.
    ///
    /// Unsigned. Distinguishing inside from outside would need the shell to be
    /// watertight and correctly wound, and nothing here depends on the sign.
    static func distanceToShell(_ p: simd_float3) -> Float {
        var best = Float.greatestFiniteMagnitude
        for q in shellPoints {
            let d = simd_distance_squared(p, q)
            if d < best { best = d }
        }
        return best == .greatestFiniteMagnitude ? .nan : sqrt(best)
    }

    private static func loadWireframe() -> [Float] {
        guard let url = Bundle.main.url(forResource: "ibex_ev1", withExtension: "mesh"),
              let data = try? Data(contentsOf: url),
              data.count > 8 else {
            print("IbexModel: ibex_ev1.mesh missing from the bundle — is it in the Resources build phase?")
            return []
        }
        return data.withUnsafeBytes { raw -> [Float] in
            guard raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self) == 0x4D584249 else { // "IBXM", LE
                print("IbexModel: ibex_ev1.mesh has a bad magic")
                return []
            }
            let triangleCount = Int(raw.loadUnaligned(fromByteOffset: 4, as: UInt32.self))
            let floatCount = triangleCount * 9
            guard data.count >= 8 + floatCount * MemoryLayout<Float>.size else {
                print("IbexModel: ibex_ev1.mesh is truncated")
                return []
            }
            var out = [Float]()
            out.reserveCapacity(floatCount)
            for i in 0..<floatCount {
                out.append(raw.loadUnaligned(fromByteOffset: 8 + i * 4, as: Float.self))
            }
            return out
        }
    }
}
