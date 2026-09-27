//
//  ControllerPoseFuser.swift
//
//  6DoF pose for a device that has no 6DoF tracking.
//
//  The Steam Controller (2026) reports gyro and accel and nothing else: no
//  constellation, no ARKit AccessoryAnchor, no magnetometer. On its own that is
//  3DoF at best, and yaw walks away within seconds. What makes it tractable is
//  that the device has **capsense on both grips** and the headset can already
//  see the hands holding it. So the hands supply position and an absolute yaw
//  reference, and the IMU supplies the responsiveness the hands lack.
//
//  Division of labour:
//
//  | Quantity     | Source                        | Why not the other one |
//  |--------------|-------------------------------|-----------------------|
//  | position     | hand anchors, exclusively     | double-integrating accel diverges in seconds |
//  | fast rotation| gyro, integrated at ~68Hz     | hand anchors are slower and jitter under occlusion |
//  | pitch/roll   | gravity from accel, low gain  | gyro alone drifts |
//  | yaw          | hand geometry, low gain       | there is no magnetometer; nothing else observes it |
//
//  The two-handed case is the good one: both grip anchors are known in model
//  space, so the vector between the two palms *is* the device's X axis, and yaw
//  is measured rather than integrated. One hand is weaker — position is still
//  solid, but yaw leans on the palm's own orientation. No hands at all is dead
//  reckoning, and the pose is marked unconfident so callers can ignore it.
//
//  Deliberately free of ARKit and CoreBluetooth so it compiles under the macOS
//  harness in `tools/steam-controller-harness/` alongside the driver.
//

import Foundation
import simd

/// One hand as observed by the headset, in **ARKit world space**.
///
/// The fuser works in Apple's frame throughout and leaves the conversion to
/// SteamVR space to the caller, exactly as `controllerToAlvrDeviceMotion` does.
/// Mixing the two inside a filter is how sign errors get buried.
struct HandObservation {
    /// OpenXR palm point — midpoint of middle metacarpal and knuckle.
    var palmPosition: simd_float3
    /// Wrist orientation, uncorrected.
    var palmOrientation: simd_quatf
    var isTracked: Bool
    /// Thumb tip, when the skeleton provides one. Not used by the filter —
    /// `ComponentProbe` uses it to work out where the sticks and pads are,
    /// since the rendermodel we have does not say.
    var thumbTip: simd_float3?
    /// Index fingertip. The other half of the same measurement: the thumb
    /// reaches the controls on the top face, the index finger reaches the
    /// trigger on the front of the handle.
    var indexTip: simd_float3?
}

/// Fused controller pose, in ARKit world space.
struct FusedControllerPose {
    var orientation: simd_quatf
    /// Model origin, i.e. where `IbexModel`'s frame is planted.
    var position: simd_float3
    var linearVelocity: simd_float3
    var angularVelocity: simd_float3
    /// How much of the pose is actually observed rather than dead-reckoned.
    /// 1.0 = both hands on the device, 0.0 = nothing has been seen for a while.
    var confidence: Float
    /// Which grips capsense reports as held, for the debug overlay.
    var heldLeft: Bool
    var heldRight: Bool
}

final class ControllerPoseFuser {

    // MARK: - Tuning
    //
    // Gains are per-second and scaled by dt at use, so they do not silently
    // change meaning when the report rate does — and it does change here, from
    // ~68Hz nominal down to ~51Hz whenever rumble is running.

    /// How hard gravity pulls pitch/roll back. Low: gravity is only clean when
    /// the device is not being accelerated, and it is usually being accelerated.
    private static let gravityGain: Float = 1.2

    /// How hard a two-handed grip pulls the whole orientation back. Higher than
    /// the single-handed case because the measurement is genuinely good.
    private static let twoHandGain: Float = 6.0

    /// How hard a single palm pulls yaw back. Deliberately feeble — a wrist
    /// rolls freely around a handle, so this is a bias correction, not a fix.
    private static let oneHandYawGain: Float = 0.8

    /// Lever arm, metres, that a unit direction constraint is treated as acting
    /// at inside the orientation fit. Points contribute to the covariance
    /// scaled by their distance from the centroid squared, directions do not,
    /// so without this the balance between gravity and the hands would be an
    /// accident of units. Set near the palm half-separation so the two weigh
    /// comparably.
    private static let gravityLever: Float = 0.06

    /// Per-frame blend for the learned per-hand rotation and anchor. Slow:
    /// these describe how someone is holding the thing, which changes over
    /// seconds, while ARKit's per-frame view of an occluded hand does not.
    private static let learnRate: Float = 0.02

    /// Accel magnitudes further than this from g are treated as manoeuvring and
    /// contribute nothing to attitude.
    private static let gravityToleranceMS2: Float = 1.5
    private static let gravityMS2: Float = 9.81

    /// Position smoothing, per second. The hands are the only position source
    /// and they jitter; this is the same trade the existing hand path makes with
    /// its EWMA, just expressed in a rate rather than a per-frame constant.
    private static let positionGain: Float = 22.0

    /// How long a pose survives with no hand observation before confidence
    /// bottoms out. Long enough to ride out a hand leaving the camera frustum,
    /// short enough that a put-down controller stops being believed.
    private static let deadReckonHorizon: TimeInterval = 0.75

    // MARK: - State

    private var orientation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private var position = simd_float3(0, 0, 0)
    private var linearVelocity = simd_float3(0, 0, 0)
    private var angularVelocity = simd_float3(0, 0, 0)

    private var lastMotionStamp: TimeInterval = 0
    private var lastObservationStamp: TimeInterval = 0
    private var hasOrientation = false
    private var hasPosition = false

    // Learned while both hands are on the device, which is the only state where
    // the pose is fully observed. Both are nil until that has happened once.
    //
    // These exist because the nominal model describes an idealised hand on an
    // idealised handle, and neither the hand nor the grasp is idealised. Rather
    // than tune constants for one person's hands, the filter watches what the
    // hands actually do while it can see the answer.

    /// Rotation from each wrist frame to the device frame.
    private var leftHandRotationOffset: simd_quatf?
    private var rightHandRotationOffset: simd_quatf?

    /// Where each palm actually sits in model space, as opposed to where the
    /// nominal grip anchor says it should.
    private var leftHandAnchor: simd_float3?
    private var rightHandAnchor: simd_float3?

    /// Vector from a pad's reported contact point to where ARKit puts that
    /// thumb's tip joint, in model space. Roughly a finger's thickness.
    private var leftPadThumbOffset: simd_float3?
    private var rightPadThumbOffset: simd_float3?

    /// A model-space point whose world position has just been observed.
    private struct Correspondence {
        var model: simd_float3
        var world: simd_float3
    }

    /// Best-fit rotation over every correspondence at once, plus gravity.
    ///
    /// This replaced picking the widest *pair* of correspondences and aligning
    /// that one axis, with gravity handled separately afterwards at a lower
    /// gain. Two problems with that, both seen on hardware:
    ///
    /// - Aligning one axis leaves a whole degree of freedom untouched. A pose
    ///   rolled 90° about the palm-to-palm axis *satisfies* the alignment, so
    ///   nothing but the weak gravity term pushed it out, and it could sit
    ///   visibly wrong until the controller was rotated far enough to break it
    ///   loose.
    /// - It threw away information. Two palms and two thumbs is four
    ///   correspondences; using the widest two discards the rest, and the
    ///   discarded ones are exactly what pins the axis the pair cannot see.
    ///
    /// Horn's method: build the 3x3 covariance between model and world offsets,
    /// pack it into the 4x4 matrix whose dominant eigenvector is the optimal
    /// rotation quaternion, and find that eigenvector by power iteration.
    ///
    /// Iterating from the *current* orientation is what makes this cheap and
    /// well-behaved. It converges in a handful of steps, and it resolves the
    /// quaternion double cover the right way round on its own rather than
    /// needing a sign fixup.
    ///
    /// Gravity joins as a **direction** correspondence, contributing to the
    /// covariance without centroid subtraction. That is what lets one solve
    /// handle every case: two palms plus gravity is fully determined, one palm
    /// plus a thumb on a pad plus gravity likewise, and gravity alone leaves
    /// yaw with no gradient at all — so the iteration simply keeps the yaw it
    /// started with, which is the correct answer when nothing observes it.
    private func fitOrientation(points: [Correspondence],
                                gravityModel: simd_float3?,
                                seed: simd_quatf) -> simd_quatf? {
        var m = simd_float3x3()
        var contributions = 0

        if points.count >= 2 {
            var modelCentroid = simd_float3(), worldCentroid = simd_float3()
            for p in points { modelCentroid += p.model; worldCentroid += p.world }
            modelCentroid /= Float(points.count)
            worldCentroid /= Float(points.count)
            for p in points {
                m += outer(p.model - modelCentroid, p.world - worldCentroid)
                contributions += 1
            }
        }

        if let gravityModel {
            // Scaled by a lever arm so a unit direction weighs about what a
            // point at that radius would. Without it the two are in different
            // units and the balance between them is accidental.
            let lever = Self.gravityLever * Self.gravityLever
            m += outer(gravityModel, simd_float3(0, 1, 0)) * lever
            contributions += 1
        }

        guard contributions >= 2 else { return nil }

        // Horn's N: its dominant eigenvector is the rotation, as (w, x, y, z).
        //
        // Named out rather than inlined because the antisymmetric terms are the
        // easy thing to get backwards, and getting them backwards yields the
        // *conjugate* rotation — a perfectly well-formed quaternion that is
        // simply wrong, by up to 180°. `m` is built columns-first as
        // (a*b.x, a*b.y, a*b.z), so column j holds the terms S_?j.
        let sxx = m.columns.0.x, syx = m.columns.0.y, szx = m.columns.0.z
        let sxy = m.columns.1.x, syy = m.columns.1.y, szy = m.columns.1.z
        let sxz = m.columns.2.x, syz = m.columns.2.y, szz = m.columns.2.z
        let trace = sxx + syy + szz

        var n = simd_float4x4(0)
        n.columns.0 = simd_float4(trace,       syz - szy,             szx - sxz,             sxy - syx)
        n.columns.1 = simd_float4(syz - szy,   sxx - syy - szz,       sxy + syx,             szx + sxz)
        n.columns.2 = simd_float4(szx - sxz,   sxy + syx,            -sxx + syy - szz,       syz + szy)
        n.columns.3 = simd_float4(sxy - syx,   szx + sxz,             syz + szy,            -sxx - syy + szz)

        let seedVector = simd_float4(seed.real, seed.imag.x, seed.imag.y, seed.imag.z)
        guard let q = dominantEigenvector(n, seed: seedVector) else { return nil }
        guard q.x.isFinite, q.y.isFinite, q.z.isFinite, q.w.isFinite else { return nil }
        return simd_normalize(simd_quatf(ix: q.y, iy: q.z, iz: q.w, r: q.x))
    }

    private func outer(_ a: simd_float3, _ b: simd_float3) -> simd_float3x3 {
        simd_float3x3(columns: (a * b.x, a * b.y, a * b.z))
    }

    /// How close two eigenvalues must be, relative to the largest, before the
    /// solution is treated as ambiguous rather than merely ill-conditioned.
    ///
    /// Swept against 282 randomised two-palm-plus-gravity configurations seeded
    /// 150° wrong: at 0.05 a couple of well-determined cases get misread as
    /// degenerate and come back over 100° off, at 0.02 every one of them solves
    /// exactly.
    private static let eigenDegeneracyTolerance: Float = 0.02

    /// Eigenvector of the largest eigenvalue of a symmetric 4x4, by cyclic
    /// Jacobi rotations.
    ///
    /// This replaced power iteration, which cannot work here. N's eigenvalues
    /// come out symmetric about zero — for a two-palm-plus-gravity fit they are
    /// almost exactly ±a, ±b — so the largest and smallest have equal magnitude
    /// and the iteration oscillates between them instead of converging. Seeded
    /// 90° off it stayed 80° off after twelve iterations and 11° off after two
    /// hundred, which is precisely the "stuck until I rotated it right over"
    /// behaviour seen on hardware.
    ///
    /// Jacobi has no such dependence: it decomposes the matrix outright, so the
    /// answer is the global optimum no matter where the filter currently thinks
    /// the controller is pointing.
    ///
    /// `seed` is used only to resolve genuine ambiguity. When the top
    /// eigenvalues are degenerate the optimum is a whole subspace rather than a
    /// point — gravity alone, with no hands, leaves yaw entirely free — and the
    /// right answer there is the one closest to the current orientation, which
    /// is the seed projected onto that subspace.
    private func dominantEigenvector(_ input: simd_float4x4, seed: simd_float4) -> simd_float4? {
        var a = input
        var v = matrix_identity_float4x4

        for _ in 0..<16 {
            var off: Float = 0
            for p in 0..<4 { for q in (p + 1)..<4 { off += a[q][p] * a[q][p] } }
            if off < 1e-20 { break }

            for p in 0..<4 {
                for q in (p + 1)..<4 {
                    let apq = a[q][p]
                    if abs(apq) < 1e-18 { continue }
                    let theta = (a[q][q] - a[p][p]) / (2 * apq)
                    let t: Float = theta == 0
                        ? 1
                        : (theta > 0 ? 1 : -1) / (abs(theta) + (theta * theta + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot()
                    let s = t * c

                    // A <- Jᵀ A J, applied as a column pass then a row pass.
                    for k in 0..<4 {
                        let akp = a[p][k], akq = a[q][k]
                        a[p][k] = c * akp - s * akq
                        a[q][k] = s * akp + c * akq
                    }
                    for k in 0..<4 {
                        let apk = a[k][p], aqk = a[k][q]
                        a[k][p] = c * apk - s * aqk
                        a[k][q] = s * apk + c * aqk
                    }
                    // V <- V J, so V's columns stay the eigenvectors.
                    for k in 0..<4 {
                        let vkp = v[p][k], vkq = v[q][k]
                        v[p][k] = c * vkp - s * vkq
                        v[q][k] = s * vkp + c * vkq
                    }
                }
            }
        }

        let values = simd_float4(a[0][0], a[1][1], a[2][2], a[3][3])
        var top = 0
        for i in 1..<4 where values[i] > values[top] { top = i }
        let largest = values[top]
        guard largest.isFinite else { return nil }

        // Project the seed onto the span of every eigenvector sharing the top
        // eigenvalue. With a unique optimum that span is one vector and this is
        // just that vector; with an ambiguous one it picks the nearest point of
        // the optimal set.
        var projection = simd_float4()
        let threshold = Self.eigenDegeneracyTolerance * max(abs(largest), 1e-12)
        for i in 0..<4 where largest - values[i] <= threshold {
            let e = v[i]
            projection += e * simd_dot(e, seed)
        }
        let length = simd_length(projection)
        if length < 1e-6 {
            let fallback = v[top]
            let fallbackLength = simd_length(fallback)
            return fallbackLength < 1e-6 ? nil : fallback / fallbackLength
        }
        return projection / length
    }

    /// Where each palm has actually been observed to sit in model space, once
    /// both hands have been on the device. nil before that.
    ///
    /// Exposed because it is the only *measurement* available of where ARKit
    /// puts a palm joint on this controller — everything else is either a mesh
    /// number or a hand-tuned one. Compared against `IbexModel.gripSensor` in
    /// the debug panel, the difference is the palm-thickness vector that
    /// `mountCorrection` is currently standing in for.
    func learnedAnchor(isLeft: Bool) -> simd_float3? {
        isLeft ? leftHandAnchor : rightHandAnchor
    }

    /// Reset to the state a fresh connection should have.
    func reset() {
        orientation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        position = .zero
        linearVelocity = .zero
        angularVelocity = .zero
        lastMotionStamp = 0
        lastObservationStamp = 0
        hasOrientation = false
        hasPosition = false
        leftHandRotationOffset = nil
        rightHandRotationOffset = nil
        leftHandAnchor = nil
        rightHandAnchor = nil
        leftPadThumbOffset = nil
        rightPadThumbOffset = nil
    }

    // MARK: - Update

    /// Folds one controller report and the current hand anchors into the pose.
    ///
    /// Returns nil until there is enough to place the device at all, which means
    /// at least one hand has been seen holding it once. Orientation alone is not
    /// enough — a pose at the world origin would draw the controller inside the
    /// floor and look like a tracking bug rather than a missing observation.
    @discardableResult
    func update(snapshot: InputSnapshot,
                left: HandObservation?,
                right: HandObservation?,
                now: TimeInterval) -> FusedControllerPose? {

        // Host time, not `motion.deviceTimestampNs`. The device clock is
        // monotonic but unrelated to the clock the hand anchors are stamped
        // with, and this filter mixes the two sources every step.
        var dt = Float(now - lastMotionStamp)
        if lastMotionStamp == 0 || dt <= 0 || dt > 0.25 {
            dt = 1.0 / 68.0     // nominal report interval
        }
        lastMotionStamp = now

        // Snapshots are pulled, not pushed, and this runs at the 90Hz display
        // rate against a ~68Hz device — so the same report is routinely seen
        // more than once. That is fine and even wanted: gyro is a *rate*, so
        // re-integrating the last known rate over the new interval is a correct
        // zero-order hold.
        //
        // What is not fine is a controller that has stopped reporting entirely.
        // `snapshot()` keeps handing back the last value forever, so a frozen
        // gyro would spin the pose away and a frozen capsense bit would keep
        // claiming a grip that was released — both with no indication anything
        // is wrong. Past this age the whole report is treated as absent.
        let isFresh = snapshot.hostTimestamp > 0 && (now - snapshot.hostTimestamp) < 0.2

        // Capsense decides which hands to believe. This is the only source for
        // that decision: both hands are on one rigid body, and a hand resting
        // near the controller looks identical to a hand holding it.
        let heldLeft = isFresh && snapshot.buttons.contains(.leftGripTouch)
        let heldRight = isFresh && snapshot.buttons.contains(.rightGripTouch)
        let leftHand = (heldLeft && (left?.isTracked ?? false)) ? left : nil
        let rightHand = (heldRight && (right?.isTracked ?? false)) ? right : nil

        // MARK: Orientation

        if !isFresh {
            angularVelocity = .zero
        }
        if isFresh, let motion = snapshot.motion {
            // Gyro is body-frame and already rotated into these axes by the
            // driver's `deviceToOpenXR`. Right-multiply: a body-frame delta.
            let omega = motion.gyro
            let angle = simd_length(omega) * dt
            if angle > 1e-7 {
                let axis = omega / simd_length(omega)
                orientation = simd_normalize(orientation * simd_quatf(angle: angle, axis: axis))
                hasOrientation = true
            }
            angularVelocity = orientation.act(omega)
        }

        // Gravity, as a body-frame direction, for the fit below to use. Only
        // trusted when the accel magnitude says the device is not being
        // manoeuvred — specific force is gravity *plus* whatever the hands are
        // doing, and only the former is a reference.
        var gravityModel: simd_float3?
        if isFresh, let motion = snapshot.motion {
            let mag = simd_length(motion.accel)
            if abs(mag - Self.gravityMS2) < Self.gravityToleranceMS2 {
                // Specific force points *up* at rest, so this body-frame
                // direction is where world up currently lies on the device.
                gravityModel = motion.accel / mag
                if !hasOrientation {
                    // No integration history: adopt gravity outright rather than
                    // starting at identity and swinging into place.
                    orientation = simd_quatf(from: gravityModel!, to: simd_float3(0, 1, 0))
                    hasOrientation = true
                }
            }
        }

        // MARK: Hand-derived correction

        // Correspondences: points whose position in *model* space is known and
        // whose position in the world has just been observed.
        //
        // Two palms is the case this filter was built around, but it is not the
        // only source. A thumb on a trackpad is a much better one in a sense —
        // the pad reports its own contact coordinate, and the pads are flat
        // rectangles measured off the CAD, so `padPoint` turns that coordinate
        // into an exact model-space point that moves as the thumb does.
        //
        // What that buys is the weak case: one hand on a grip plus that thumb
        // on a pad is *two* correspondences, which is the same class of
        // constraint two hands give. Yaw stops being dead-reckoned.
        var pairs: [Correspondence] = []
        if let l = leftHand {
            pairs.append(Correspondence(model: leftHandAnchor ?? IbexModel.mountedGripAnchor(isLeft: true),
                                        world: l.palmPosition))
        }
        if let r = rightHand {
            pairs.append(Correspondence(model: rightHandAnchor ?? IbexModel.mountedGripAnchor(isLeft: false),
                                        world: r.palmPosition))
        }
        for isLeft in [true, false] {
            guard let pad = isLeft ? snapshot.leftTouchpad : snapshot.rightTouchpad, pad.isTouched,
                  let hand = isLeft ? leftHand : rightHand, let tip = hand.thumbTip,
                  // Needs the thumb-tip-to-contact offset, which is learned
                  // while two-handed. ARKit reports the tip *joint*; the pad is
                  // pressed by the fleshy underside, a finger's thickness away.
                  let offset = isLeft ? leftPadThumbOffset : rightPadThumbOffset
            else { continue }
            pairs.append(Correspondence(model: IbexModel.padPoint(isLeft: isLeft, position: pad.position) + offset,
                                        world: tip))
        }

        // One fit over everything observed, rather than a sequence of
        // single-axis corrections at different gains. See `fitOrientation`.
        if let fitted = fitOrientation(points: pairs, gravityModel: gravityModel, seed: orientation) {
            let gain = pairs.count >= 2 ? Self.twoHandGain : Self.gravityGain
            if hasOrientation {
                orientation = simd_normalize(simd_slerp(orientation, alignedTo(fitted, orientation),
                                                        min(1, gain * dt)))
            } else {
                orientation = fitted
                hasOrientation = true
            }
        } else if hasOrientation, let single = leftHand ?? rightHand {
            // One palm. Yaw is not observable from a single point, so the only
            // honest reference is the rotation learned during a two-handed grip.
            //
            // With no calibration yet, yaw is simply *held*: it keeps
            // integrating gyro and accepts the drift. That is deliberately
            // better than pulling it toward the raw wrist orientation, which
            // would look like a fix while quietly steering the device to a
            // wrong constant heading.
            let isLeft = leftHand != nil
            if let offset = isLeft ? leftHandRotationOffset : rightHandRotationOffset {
                // Twist decomposition about world up, rather than the heading of
                // some chosen palm axis.
                //
                // The old version took `atan2` of a wrist axis projected onto
                // the floor. Hold a gamepad naturally and that axis is close to
                // vertical, where its horizontal projection is nearly zero and
                // its heading is numerically meaningless — it can swing by half
                // a turn between frames from noise alone. That is where the
                // 180° flips came from.
                let target = simd_normalize(single.palmOrientation * offset)
                let step = yawTwist(from: orientation, to: target) * min(1, Self.oneHandYawGain * dt)
                orientation = simd_normalize(simd_quatf(angle: step, axis: simd_float3(0, 1, 0)) * orientation)
            }
        }

        // MARK: Position
        //
        // Always from the hands. Nothing here integrates acceleration: over the
        // ~0.75s this filter has to survive an occlusion, double integration of
        // a 2g-range accel accumulates metres of error.

        // `mounted*` rather than the raw anchors: the measured centroids sit
        // inside the handle, and the CAD centre is not where ARKit puts a palm
        // joint on a grasped handle. See `IbexModel.mountCorrection`.
        var observed: simd_float3?
        if let l = leftHand, let r = rightHand {
            let midpoint = (l.palmPosition + r.palmPosition) * 0.5
            observed = midpoint - orientation.act(IbexModel.mountedGripCentre)
        } else if let single = leftHand ?? rightHand {
            // Use the anchor *learned for this hand* if there is one.
            //
            // The nominal anchors are where a palm should be. Real palms are
            // not there — hands differ, and people hold a controller how they
            // like. Two-handed, the solve averages the two errors and the body
            // lands between them. Drop to one hand and that hand's whole error
            // applies at once, so the controller visibly jumps at the moment of
            // release even though nothing about it moved. That is exactly the
            // symptom, and learning the per-hand anchor removes it by
            // construction: the one-handed solve reproduces what the two-handed
            // solve was already reporting.
            let isLeft = leftHand != nil
            let anchor = (isLeft ? leftHandAnchor : rightHandAnchor)
                ?? IbexModel.mountedGripAnchor(isLeft: isLeft)
            observed = single.palmPosition - orientation.act(anchor)
        }

        if let target = observed {
            lastObservationStamp = now
            if hasPosition {
                let blend = min(1, Self.positionGain * dt)
                let next = simd_mix(position, target, simd_float3(repeating: blend))
                linearVelocity = (next - position) / dt
                position = next
            } else {
                position = target
                linearVelocity = .zero
                hasPosition = true
            }
        } else if hasPosition {
            // Coast. Velocity is kept but decayed, so a hand that reappears does
            // not have to fight a stale extrapolation.
            position += linearVelocity * dt
            linearVelocity *= max(0, 1 - 4 * dt)
        }

        // MARK: Learn, while the answer is visible
        //
        // Only two-handed, and only once there is a position to measure
        // against. Slow blends on purpose: this is calibration, and a fast one
        // would just track the noise in whichever hand ARKit is currently least
        // sure about.
        //
        // Two-handed specifically because position is solved from the *nominal*
        // anchors in that case. Learn against a position that was itself
        // derived from the learned values and the whole thing becomes a fixed
        // point — stable, self-consistent, and free to sit anywhere.
        if hasPosition, let l = leftHand, let r = rightHand {
            learn(hand: l, isLeft: true)
            learn(hand: r, isLeft: false)
            removeAnchorGaugeFreedom()
            // Same trick for the thumb: ARKit reports the tip joint, the pad
            // reports where it was actually pressed, and the difference is a
            // finger's thickness in a direction nothing else can supply.
            for isLeft in [true, false] {
                guard let pad = isLeft ? snapshot.leftTouchpad : snapshot.rightTouchpad, pad.isTouched,
                      let tip = (isLeft ? l : r).thumbTip else { continue }
                let observed = orientation.inverse.act(tip - position)
                    - IbexModel.padPoint(isLeft: isLeft, position: pad.position)
                let blend = simd_float3(repeating: Self.learnRate)
                if isLeft {
                    leftPadThumbOffset = leftPadThumbOffset.map { simd_mix($0, observed, blend) } ?? observed
                } else {
                    rightPadThumbOffset = rightPadThumbOffset.map { simd_mix($0, observed, blend) } ?? observed
                }
            }
        }

        guard hasPosition, hasOrientation else { return nil }

        let staleness = lastObservationStamp == 0 ? Self.deadReckonHorizon : (now - lastObservationStamp)
        var confidence: Float = Float(max(0, 1 - staleness / Self.deadReckonHorizon))
        if !(heldLeft && heldRight) { confidence *= 0.6 }

        return FusedControllerPose(orientation: orientation,
                                   position: position,
                                   linearVelocity: linearVelocity,
                                   angularVelocity: angularVelocity,
                                   confidence: confidence,
                                   heldLeft: heldLeft,
                                   heldRight: heldRight)
    }

    // MARK: - Derived poses

    /// Where each virtual controller goes, given a fused pose.
    ///
    /// Both hands get a pose even when only one is holding the device, and the
    /// two move rigidly together — because they physically do. Sending the same
    /// pose to both (the obvious shortcut) puts two virtual controllers inside
    /// each other and looks wrong the instant a game draws them.
    static func handPose(_ pose: FusedControllerPose, isLeft: Bool) -> (simd_quatf, simd_float3) {
        // The same anchor that solved position, so this lands back exactly on
        // the palm. Using the raw anchor here would reintroduce the mount
        // correction as an error between the body and the virtual controllers.
        let anchor = IbexModel.mountedGripAnchor(isLeft: isLeft)
        return (pose.orientation, pose.position + pose.orientation.act(anchor))
    }

    /// Model-to-world matrix, for drawing the body wireframe.
    static func modelMatrix(_ pose: FusedControllerPose) -> simd_float4x4 {
        var m = simd_float4x4(pose.orientation)
        m.columns.3 = simd_float4(pose.position, 1)
        return m
    }

    // MARK: - Helpers

    /// A fraction of the rotation carrying `a` onto `b`.
    private func partialRotation(from a: simd_float3, to b: simd_float3, fraction: Float) -> simd_quatf {
        let full = simd_quatf(from: a, to: b)
        return simd_normalize(simd_slerp(simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), full, fraction))
    }

    /// Rotation about world up needed to carry `a` onto `b`, in radians.
    ///
    /// Twist decomposition: take the relative rotation, keep only its component
    /// about the up axis, discard the swing. This replaced projecting a chosen
    /// axis onto the floor and taking `atan2` of it, which is degenerate exactly
    /// when that axis is vertical — the normal way to hold a gamepad — and gave
    /// meaningless headings that could flip by half a turn frame to frame.
    private func yawTwist(from a: simd_quatf, to b: simd_quatf) -> Float {
        var err = b * a.inverse
        // Both signs name the same rotation; pick the one that twists the short
        // way, or the correction happily takes the 358° route.
        if err.real < 0 { err = simd_quatf(vector: -err.vector) }
        return 2 * atan2(err.imag.y, err.real)
    }

    /// Folds one observation into a hand's learned rotation offset and anchor.
    ///
    /// Both are slewed slowly rather than snapped. They are properties of how
    /// this person holds this controller, which changes over seconds, while
    /// ARKit's per-frame estimate of an occluded hand is far noisier than that.
    private func learn(hand: HandObservation, isLeft: Bool) {
        let rotation = simd_normalize(hand.palmOrientation.inverse * orientation)
        let anchor = orientation.inverse.act(hand.palmPosition - position)

        if isLeft {
            leftHandRotationOffset = leftHandRotationOffset
                .map { simd_normalize(simd_slerp($0, alignedTo(rotation, $0), Self.learnRate)) } ?? rotation
            leftHandAnchor = leftHandAnchor
                .map { simd_mix($0, anchor, simd_float3(repeating: Self.learnRate)) } ?? anchor
        } else {
            rightHandRotationOffset = rightHandRotationOffset
                .map { simd_normalize(simd_slerp($0, alignedTo(rotation, $0), Self.learnRate)) } ?? rotation
            rightHandAnchor = rightHandAnchor
                .map { simd_mix($0, anchor, simd_float3(repeating: Self.learnRate)) } ?? anchor
        }
    }

    /// Strips the one thing the learned anchors must never carry: a net
    /// rotation of the pair.
    ///
    /// `learn` measures each anchor as `orientation.inverse * (palm - position)`
    /// — in the frame of the orientation we currently believe. If that belief is
    /// wrong by some rotation ε, the anchors absorb ε, and the next fit is then
    /// perfectly satisfied by the *same* wrong orientation. The error stops
    /// being an error and becomes the calibration. Worked through:
    ///
    ///     anchor_learned = R_cur⁻¹ · R_true · anchor_true  =  ε · anchor_true
    ///     next fit solves  R · ε · anchor_true = R_true · anchor_true
    ///                  ⇒  R = R_true · ε⁻¹ = R_cur
    ///
    /// so the fit returns exactly what it was given and ε never decays.
    /// Gravity pins pitch and roll, but it is a direction along the vertical and
    /// supplies no yaw gradient at all — so **yaw** is the component that
    /// latches, at whatever error happened to exist when learning began. That
    /// is the intermittent stuck yaw offset seen on hardware: not a solver
    /// artefact, and not a constant bias, but a frozen initial condition, which
    /// is why it differed from session to session.
    ///
    /// The fix is to keep the anchors' *shape* and drop their orientation. A
    /// common translation (where the pair sits, i.e. how this person's palms
    /// ride the handle) and the separation (hand size) are both observable and
    /// are preserved. The pair's direction is pinned back to nominal, because a
    /// rotation of the anchors is indistinguishable from a rotation of the body
    /// and therefore is not something observation can ever justify.
    ///
    /// Only the two-point pair needs this. A rotation about the line joining
    /// them does not move either point, so it cannot be encoded here at all.
    private func removeAnchorGaugeFreedom() {
        guard let l = leftHandAnchor, let r = rightHandAnchor else { return }
        let separation = l - r
        guard simd_length(separation) > 1e-4 else { return }
        let learnedAxis = simd_normalize(separation)
        let nominalAxis = simd_normalize(IbexModel.mountedGripAnchor(isLeft: true)
                                         - IbexModel.mountedGripAnchor(isLeft: false))
        let centre = (l + r) * 0.5
        let correction = simd_quatf(from: nominalAxis, to: learnedAxis).inverse
        leftHandAnchor = centre + correction.act(l - centre)
        rightHandAnchor = centre + correction.act(r - centre)
    }

    /// Picks the sign of `q` closest to `reference`, so slerp takes the short
    /// way round. Quaternions double-cover rotations and a basis-built one has
    /// no memory of which cover it came from; without this the correction
    /// occasionally spins the controller the long way.
    private func alignedTo(_ q: simd_quatf, _ reference: simd_quatf) -> simd_quatf {
        simd_dot(q.vector, reference.vector) < 0 ? simd_quatf(vector: -q.vector) : q
    }
}
