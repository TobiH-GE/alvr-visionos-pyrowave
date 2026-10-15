//
//  FrameExtrapolation.swift
//
//  When the stream runs slower than the display (a game at 45 fps on a 90 Hz headset), every
//  frame is shown twice. The renderer already turns the repeated frame to the current head pose,
//  but everything that moves in the scene stands still for a display frame and then jumps. With
//  the "Frame Interpolation" setting the second display shows an in-between of the frame and the
//  one before it instead, built once the frame has arrived:
//
//  1. The motion between the two frames comes from the streamer (its NVIDIA Optical Flow vectors,
//     receiveServerMotion; for HEVC, H.264 and AV1 the only source, they arrive ahead of the frame,
//     EventHandler.receiveVideoMotion), or, with "Use Server Motion Vectors" off, is estimated here on a queue
//     of its own (FrameExtrapolation.metal): PyroWaveDecoder copies the decoder's lowpass (LL)
//     bands of every frame (capture), a free image pyramid from 1/2 down to 1/32 of the picture,
//     and a coarse to fine block search finds 4x4 vectors per 32x32 PyroWave block. The motion
//     head rotation alone explains (from the two frames' head poses) is subtracted: the renderer's
//     reprojection already handles it, with the newer pose.
//  2. Per 8x8 pixels the motion the in-between follows can be chosen between that field, head
//     rotation only and none (HUD): by the streamer on the original frames (streamerChoice) or
//     here on the frames' lowpass level 0 (selection).
//  3. The renderer moves the newer frame back along the field (alpha -0.5 on its first display at
//     45 fps on 90 Hz), optionally blending in the previous frame moved forward.
//
//  tools/frameflow on the branch frame-interpolation-archive measures the same kernels offline on
//  recordings.
//

import CompositorServices
import Foundation
import CoreVideo
import Metal
import QuartzCore
import simd

// Head orientation of every tracking sample the app sent, by target timestamp: the pose the
// streamer rendered the frame with that timestamp from.
final class FramePoseHistory {
    static let shared = FramePoseHistory()

    private let lock = NSLock()
    private var timestamps = [UInt64](repeating: 0, count: 512)
    private var orientations = [simd_quatf](repeating: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), count: 512)
    private var next = 0

    func record(timestampNs: UInt64, orientation: simd_quatf) {
        lock.withLock {
            timestamps[next] = timestampNs
            orientations[next] = orientation
            next = (next + 1) % timestamps.count
        }
    }

    // The pose the frame was really rendered with: the one ALVR's client core sent, which it
    // predicts ahead from the one recorded here with the head's angular velocity
    // (ClientCoreContext::send_tracking, up to the pipeline latency). Measured 2026-10-08, the
    // recorded one is off by velocity noise x ~40 ms at 45 Hz, which made the in-betweens shake.
    // The recorded one only when the client core no longer has it.
    func orientation(timestampNs: UInt64) -> simd_quatf? {
        var pose = AlvrPose()
        if alvr_get_sent_head_pose(timestampNs, &pose) {
            return simd_quatf(ix: pose.orientation.x, iy: pose.orientation.y, iz: pose.orientation.z, r: pose.orientation.w)
        }
        return lock.withLock {
            timestamps.firstIndex(of: timestampNs).map { orientations[$0] }
        }
    }
}

final class FrameExtrapolation {
    // The eye tangents the renderer draws frames with (its sentViewTangents), set every frame.
    struct ViewGeometry: Equatable {
        var tangents: [simd_float4]
    }

    // The stream's foveation as the renderer's video pipelines use it, set when they are built.
    private struct Foveation: Equatable {
        var enabled = false
        var eyeSizeRatio = SIMD2<Float>(1, 1)
        var centerSize = SIMD2<Float>(0, 0)
        var centerShift = SIMD2<Float>(0, 0)
        var edgeRatio = SIMD2<Float>(1, 1)
    }
    private static let foveationLock = NSLock()
    private static var foveation = Foveation()

    static func setFoveation(_ vars: FoveationVars) {
        let value = Foveation(enabled: vars.enabled, eyeSizeRatio: SIMD2(vars.eyeWidthRatio, vars.eyeHeightRatio),
                              centerSize: SIMD2(vars.centerSizeX, vars.centerSizeY),
                              centerShift: SIMD2(vars.centerShiftX, vars.centerShiftY),
                              edgeRatio: SIMD2(vars.edgeRatioX, vars.edgeRatioY))
        foveationLock.withLock { foveation = value }
    }

    // Must match the structs in FrameExtrapolation.metal.
    private struct SearchParams {
        var levelSize: SIMD2<Int32>
        var eyeWidth: Int32
        var cellsPerEye: Int32
        var gridSize: SIMD2<Int32>
        var parentCellsPerEye: Int32
        var parentGridSize: SIMD2<Int32>
        var searchRadius: Int32
        var pixelsPerTexel: Float
        var valueScale: Float
        var lambda: Float
        var subpixel: Int32
        var predictedShift: SIMD2<Float>
        var rotationCellsPerEye: Int32
        var rotationCellScale: Int32
        var debugStop: Int32
        var confidenceRatio: Float
        var confidenceMinGain: Float
    }

    private struct RotationParams {
        var rotation: simd_float3x3
        var tangentsLeft: SIMD4<Float>
        var tangentsRight: SIMD4<Float>
        var frameSize: SIMD2<Float>
        var eyeSizeRatio: SIMD2<Float>
        var cellPixels: Float
        var cellsPerEye: Int32
        var lutSize: Int32
        var pad: Int32 = 0
    }

    // Lowpass levels: 0 (1/2 of the picture) to 4 (1/32, the coarsest band).
    private static let levels = 5
    private static let coarseRadius: Int32 = 8
    private static let refineRadius: Int32 = 2
    private static let lambda: Float = 4
    // LL texels are about picture values in 0...1; the search works in 8 bit units.
    private static let valueScale: Float = 255
    private static let slots = 4
    // Source intervals longer than this many display periods count as a slow stream.
    private static let slowStreamFactor = 1.4
    // Interpolation: how far a frame shown once more because the next one is late (it missed its
    // pickup) may be carried forward on the content clock. 0.5 = the full display period at 45 fps
    // on 90 Hz; it was 0.25, which made the content step 0.25 / 0.7 / 0.3 frames around each late
    // frame instead of 0.5 each (recording 2026-10-08, two frames 5 ms late).
    private static let maxLateAlpha = 0.5
    private static let warpIterations: Int32 = 3
    // Finest level searched: 1 = cells of 16x16 picture pixels. Level 0 (8x8) cost 11-19 ms of GPU per
    // frame on the headset at 4864x1728 (2026-10-07) and the offline bench shows it adds almost
    // nothing to the extrapolation (22.74 vs 22.79 dB, 27.43 vs 27.48 dB on two Subnautica pieces).
    private static let finestLevel = 1
    // A refined vector must beat "head rotation only" by 15 % and 96 (8 bit units over the 8x8
    // window) of SAD, else the cell keeps the rotation: flat areas (menu backgrounds) otherwise
    // got noise vectors that scratched up the edges of text.
    private static let confidenceRatio: Float = 0.15
    private static let confidenceMinGain: Float = 96
    // Scene motion shorter than this (picture pixels per source interval) is set to none.
    private static let deadzonePixels: Float = 1.5

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let width: Int
    private let height: Int
    private let coarsePipeline: MTLComputePipelineState
    private let refinePipeline: MTLComputePipelineState
    private let medianPipeline: MTLComputePipelineState
    private let rotationPipeline: MTLComputePipelineState
    private let captureEvent: MTLEvent
    private var captureEventValue: UInt64 = 0

    private final class Capture {
        let lowpass: [MTLTexture]
        var timestamp: UInt64 = 0
        var eventValue: UInt64 = 0
        // lowpass[0] holds this frame's level 0 (under lock with timestamp).
        var hasLevel0 = false
        init(_ lowpass: [MTLTexture]) { self.lowpass = lowpass }
    }

    private final class Field {
        let texture: MTLTexture
        var timestamp: UInt64 = 0
        var previousTimestamp: UInt64 = 0
        var ready = false
        // From the streamer's motion vectors rather than estimated here.
        var fromServer = false
        init(_ texture: MTLTexture) { self.texture = texture }
    }

    // Decode worker queue only.
    private var captures: [Capture]
    private var nextCapture = 0
    private var lastCapture: Capture?
    private var pendingEstimate: (Capture, Capture)?

    // Scratch of the estimation queue (serial, so one set is enough).
    private let levelFields: [MTLTexture]
    private let rotationField: MTLTexture
    private var foveationTables: MTLBuffer?
    private var foveationTablesFor: Foveation?
    private let tablesLock = NSLock()

    // Motion vectors from the streamer (setting "Use Server Motion Vectors"), network thread only;
    // the uploads and the rotation scratch are used by its command buffers.
    private var serverTimestamp: UInt64 = 0
    private var serverRowsReceived: [Bool] = []
    private var serverRowCount = 0
    // The streamer's motion choice (PyroWaveStripes::MotionChoiceMagic, setting "Interpolation:
    // Motion Choice from Streamer"), written by the network thread straight into shared memory the
    // renderer reads in place of the selection texture: per FX_SELECT_BLOCK block, 0 the field,
    // 1 head rotation only, 2 none, as a half float in .r. Three slots, like the vectors.
    private var choiceTextures: [MTLTexture] = []
    private var choiceBuffers: [MTLBuffer] = []
    private var choiceRowHalves = 0
    private var choiceNext = 0
    private var choiceSlot = 0
    private var choiceTimestamp: UInt64 = 0
    private var choiceRowsReceived: [Bool] = []
    private var choiceRowCount = 0
    private var choiceMismatchLogged = false
    // Under lock: the complete choices, newest last.
    private var choicesComplete: [(timestamp: UInt64, reference: UInt64, slot: Int)] = []
    private var serverUploadBuffers: [MTLBuffer] = []
    private var serverUploadRowFloats = 0
    private var serverUploadCurrent = 0
    private var serverField: Field?
    private var serverMismatchLogged = false
    private var serverUploads: [MTLTexture] = []
    private var serverUploadNext = 0
    private let serverRotationField: MTLTexture
    private let fillPipeline: MTLComputePipelineState
    private let serverFilled: MTLTexture

    // Shared with the renderer, under lock.
    private let lock = NSLock()
    private var fields: [Field]
    private var nextField = 0
    private var geometry: ViewGeometry?
    private var displayPeriod = 1.0 / 90.0
    // The streamer's motion vectors: the timestamp of the newest frame they came for and the
    // frames' interval (seconds). With NVENC they follow their frame (FrameMotion.h), so a frame
    // whose vectors have not begun to arrive is held back for them too (shouldHoldBack).
    private var serverMotionTimestamp: UInt64 = 0
    private var serverMotionInterval = 1.0 / 45.0
    private var lastDisplayVsync = 0.0
    private var firstShown: [UInt64: Double] = [:]
    // Smoothed time from a frame's timestamp to its first display (seconds), for interpolation.
    private var contentLag = 0.0
    private var contentLagJumps = 0
    private var contentLagJumpSide = 0
    let idleTexture: MTLTexture

    // "Choose motion per pixel" (render thread only): the choice per 4x4 pixels for the frame shown,
    // made once per frame; three textures, since renders still in flight read the older ones.
    private lazy var selectPipeline: MTLComputePipelineState? = {
        guard let function = device.makeDefaultLibrary()?.makeFunction(name: "fxSelectMotion") else { return nil }
        return try? device.makeComputePipelineState(function: function)
    }()
    private lazy var selectionTextures: [MTLTexture] = (0..<3).compactMap { slot in
        let block = Int(FX_SELECT_BLOCK)
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Float, width: (width + block - 1) / block,
                                                         height: (height + block - 1) / block, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        let t = device.makeTexture(descriptor: d)
        t?.label = "FX selection \(slot)"
        return t
    }
    private var selectionNext = 0
    private var selectionFor: (timestamp: UInt64, previous: UInt64, texture: MTLTexture)?

    let cellsPerEye: Int
    let gridRows: Int
    static let cellPixels = 8 << finestLevel

    init?(device: MTLDevice, width: Int, height: Int) {
        // The eye boundary must fall on a texel boundary at the coarsest level.
        guard width % 64 == 0, height % 32 == 0,
              let library = device.makeDefaultLibrary(),
              let queue = device.makeCommandQueue(),
              let coarse = library.makeFunction(name: "fxMotionSearchCoarse"),
              let refine = library.makeFunction(name: "fxMotionSearchRefine"),
              let median = library.makeFunction(name: "fxMedian"),
              let rotation = library.makeFunction(name: "fxRotationField"),
              let fill = library.makeFunction(name: "fxFillUnreliable"),
              let fillPipeline = try? device.makeComputePipelineState(function: fill),
              let coarsePipeline = try? device.makeComputePipelineState(function: coarse),
              let refinePipeline = try? device.makeComputePipelineState(function: refine),
              let medianPipeline = try? device.makeComputePipelineState(function: median),
              let rotationPipeline = try? device.makeComputePipelineState(function: rotation),
              let captureEvent = device.makeEvent() else {
            pyroLog("Frame extrapolation: unavailable for \(width)x\(height)")
            return nil
        }
        self.device = device
        self.queue = queue
        queue.label = "Frame extrapolation"
        self.width = width
        self.height = height
        self.coarsePipeline = coarsePipeline
        self.refinePipeline = refinePipeline
        self.medianPipeline = medianPipeline
        self.rotationPipeline = rotationPipeline
        self.fillPipeline = fillPipeline
        self.captureEvent = captureEvent

        func texture(_ format: MTLPixelFormat, _ w: Int, _ h: Int, _ label: String) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            let t = device.makeTexture(descriptor: d)
            t?.label = label
            return t
        }

        var captures: [Capture] = []
        for slot in 0..<Self.slots {
            var lowpass: [MTLTexture] = []
            for level in 0..<Self.levels {
                guard let t = texture(.r16Float, (width / 2) >> level, (height / 2) >> level, "FX capture \(slot) LL\(level)") else {
                    return nil
                }
                lowpass.append(t)
            }
            captures.append(Capture(lowpass))
        }
        self.captures = captures

        cellsPerEye = Self.cellsPerEye(width: width, level: Self.finestLevel)
        gridRows = Self.gridRows(height: height, level: Self.finestLevel)
        var levelFields: [MTLTexture] = []
        for level in 0..<Self.levels {
            guard let t = texture(.rg16Float, 2 * Self.cellsPerEye(width: width, level: level),
                                  Self.gridRows(height: height, level: level), "FX search L\(level)") else {
                return nil
            }
            levelFields.append(t)
        }
        self.levelFields = levelFields
        var fields: [Field] = []
        for slot in 0..<Self.slots {
            // rg: scene motion, ba: head rotation motion (see fxMedian).
            guard let t = texture(.rgba16Float, 2 * cellsPerEye, gridRows, "FX motion \(slot)") else {
                return nil
            }
            fields.append(Field(t))
        }
        self.fields = fields
        guard let rotationField = texture(.rg16Float, 2 * cellsPerEye, gridRows, "FX rotation"),
              let serverRotationField = texture(.rg16Float, 2 * cellsPerEye, gridRows, "FX streamer rotation"),
              let serverFilled = texture(.rg16Float, 2 * cellsPerEye, gridRows, "FX streamer filled"),
              let idle = texture(.rg16Float, 1, 1, "FX idle") else {
            return nil
        }
        self.rotationField = rotationField
        self.serverRotationField = serverRotationField
        self.serverFilled = serverFilled
        self.idleTexture = idle
        // The streamer's vectors are written by the network thread straight into shared memory the
        // GPU reads as a texture (unified memory: no staging array, no copy into the texture).
        // Three, so the one being filled is never one a command buffer may still read.
        let rowBytes = (2 * cellsPerEye * 16 + device.minimumLinearTextureAlignment(for: .rgba32Float) - 1)
            / device.minimumLinearTextureAlignment(for: .rgba32Float) * device.minimumLinearTextureAlignment(for: .rgba32Float)
        for i in 0..<3 {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 2 * cellsPerEye, height: gridRows, mipmapped: false)
            d.usage = [.shaderRead]
            d.storageMode = .shared
            guard let buffer = device.makeBuffer(length: rowBytes * gridRows, options: .storageModeShared),
                  let t = buffer.makeTexture(descriptor: d, offset: 0, bytesPerRow: rowBytes) else { return nil }
            t.label = "FX streamer motion \(i)"
            serverUploads.append(t)
            serverUploadBuffers.append(buffer)
        }
        serverUploadRowFloats = rowBytes / 4
        let block = Int(FX_SELECT_BLOCK)
        let choiceWidth = (width + block - 1) / block, choiceHeight = (height + block - 1) / block
        let choiceAlignment = device.minimumLinearTextureAlignment(for: .rg16Float)
        let choiceRowBytes = (choiceWidth * 4 + choiceAlignment - 1) / choiceAlignment * choiceAlignment
        for i in 0..<3 {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Float, width: choiceWidth, height: choiceHeight, mipmapped: false)
            d.usage = [.shaderRead]
            d.storageMode = .shared
            guard let buffer = device.makeBuffer(length: choiceRowBytes * choiceHeight, options: .storageModeShared),
                  let t = buffer.makeTexture(descriptor: d, offset: 0, bytesPerRow: choiceRowBytes) else { return nil }
            t.label = "FX streamer motion choice \(i)"
            choiceTextures.append(t)
            choiceBuffers.append(buffer)
        }
        choiceRowHalves = choiceRowBytes / 2
        pyroLog("Frame extrapolation: ready for \(width)x\(height), \(2 * cellsPerEye)x\(gridRows) motion vectors")
    }

    private static func cellsPerEye(width: Int, level: Int) -> Int { (((width / 2) >> (level + 1)) + 3) / 4 }
    private static func gridRows(height: Int, level: Int) -> Int { ((height >> (level + 1)) + 3) / 4 }

    // MARK: - Renderer side

    // The eye tangents frames are drawn with.
    func setViewGeometry(_ newGeometry: ViewGeometry) {
        lock.withLock {
            if geometry != newGeometry {
                geometry = newGeometry
            }
        }
    }

    // Called once per rendered display frame, with the drawable's presentation time.
    func noteDisplay(vsync: Double) {
        lock.withLock {
            let period = vsync - lastDisplayVsync
            // Only steps of about one display period: a frame the GPU did not finish in time made
            // the gap two periods long, and averaging those in (2026-10-09, an overloaded GPU at
            // 7040x2496) raised the estimate to 13-16 ms at 90 Hz, so a third of the 45 fps frames
            // no longer counted as a slow stream and got no in-between.
            if period > 0.004 && period < 0.05 && period < displayPeriod * 1.3 {
                displayPeriod = 0.9 * displayPeriod + 0.1 * period
            } else if period > 0.004 && period < displayPeriod * 0.7 {
                // A faster display (a new stream, another refresh rate): follow it at once.
                displayPeriod = period
            }
            lastDisplayVsync = vsync
        }
    }

    // What the video fragment shader needs for the frame with this timestamp, shown at vsync.
    //
    // Extrapolation: every repeat of a frame moves its content further along the motion, by the
    // time since it was first shown over the source interval (0.5 on the second display of a 45 fps
    // frame). Interpolation: the frame's first display is moved BACK towards the previous frame by
    // (interval - display period) / interval (-0.5 at 45 fps on 90 Hz, -0.67 then -0.33 at 30 fps),
    // and it reaches 0 one display period later. That is the in-between of the two frames, from known
    // motion, for one display period of extra scene latency; head rotation is still reprojected to
    // the current pose, so head latency does not change.
    // Returns the field, the uniform and the timestamp of the frame the field's motion starts from
    // (0 if none), which the renderer needs to blend in the previous frame.
    func renderState(timestamp: UInt64, vsync: Double) -> (MTLTexture, FrameExtrapolationUniform, UInt64) {
        var uniform = FrameExtrapolationUniform(frameSize: SIMD2(Float(width), Float(height)), alpha: 0,
                                                cellPixels: Float(Self.cellPixels), cellsPerEye: Int32(cellsPerEye),
                                                gridRows: Int32(gridRows), iterations: Self.warpIterations, blendPrevious: 0)
        return lock.withLock {
            let shownAt: Double
            if let first = firstShown[timestamp] {
                shownAt = first
            } else {
                firstShown[timestamp] = vsync
                // How long after its timestamp a frame is first shown, smoothed: the base of the content
                // clock below. Outliers (a stall, a new stream) are limited to 2 ms per frame.
                let lag = vsync - Double(timestamp) / 1e9
                if lag > 0, lag < 0.5 {
                    // Three frames in a row more than 8 ms off the same way: the pipeline's delay
                    // changed (a stall, a reconnect), so the clock jumps there. Catching up at 2 ms
                    // per frame left every display moved by a whole interval (alpha -1) for a
                    // quarter of a second, which doubled every error of the field (2026-10-12).
                    let off = lag - contentLag
                    let side = abs(off) > 0.008 ? (off > 0 ? 1 : -1) : 0
                    contentLagJumps = side != 0 && side == contentLagJumpSide ? contentLagJumps + 1 : (side != 0 ? 1 : 0)
                    contentLagJumpSide = side
                    if contentLag == 0 || contentLagJumps >= 3 {
                        contentLag = lag
                        contentLagJumps = 0
                    } else {
                        contentLag += 0.1 * min(max(off, -0.02), 0.02)
                    }
                }
                if firstShown.count > 16, let oldest = firstShown.keys.min() {
                    firstShown.removeValue(forKey: oldest)
                }
                shownAt = vsync
            }
            // "Predict with Streamer Vectors": the content clock runs with the frames' first display,
            // so a frame is first shown as decoded, and once it is behind the clock (its repeat) it
            // moves forward along the next frame's field, which the streamer's vectors bring before
            // that frame (mode 5).
            let predict = ALVRClientApp.gStore.settings.frameInterpolationPredict
            if predict, contentLag > 0 {
                let ahead = vsync - contentLag - Double(timestamp) / 1e9
                if ahead > 0, let next = fields.first(where: { $0.previousTimestamp == timestamp && $0.timestamp > timestamp && $0.ready }) {
                    let alpha = min(1, ahead / (Double(next.timestamp - timestamp) / 1e9))
                    guard alpha > 0.1 else {
                        return (idleTexture, uniform, 0)
                    }
                    uniform.alpha = Float(alpha)
                    uniform.blendPrevious = 5
                    return (next.texture, uniform, 0)
                }
            }
            guard let field = fields.first(where: { $0.timestamp == timestamp }) else {
                // A stream not slower than the display: nothing to do.
                return (idleTexture, uniform, 0)
            }
            guard field.ready, field.timestamp > field.previousTimestamp else {
                return (idleTexture, uniform, 0)
            }
            let interval = Double(field.timestamp - field.previousTimestamp) / 1e9
            // The content shown runs on a clock that advances by exactly one display period per
            // display, one source interval minus one display period behind the frames' smoothed
            // first display: on its first display a frame is (interval - period) / interval back
            // towards the previous one (-0.5 at 45 fps on 90 Hz), and the steps stay even when
            // frames are shown 2 or 3 times (45 fps on the 100 Hz visionOS falls back to under
            // 50 Hz flicker), which the per-frame formula before did not (it wobbled).
            // Predicting: no extra period; a frame whose next field is not in yet is extrapolated
            // along its own.
            let contentTime = predict ? vsync - contentLag : vsync - contentLag - interval + displayPeriod
            // Predicting, a frame ahead of the clock is shown as decoded: blending it back cost 6-7 ms
            // of GPU per display with HEVC (2026-10-15) and only added latency.
            let alpha = max(predict ? 0 : -1, min(Self.maxLateAlpha, (contentTime - Double(timestamp) / 1e9) / interval))
            // A frame's second display lands within a few hundredths of 0 (the content clock is
            // smoothed); moving it by that is under a pixel or two but ran the interpolation path
            // on every display instead of every second one (2026-10-09: 445 of 450 displays
            // "moved"; with 0.05, again 405-422 once the decode slowed down and the clock lagged).
            guard abs(alpha) > 0.1 else {
                return (idleTexture, uniform, 0)
            }
            uniform.alpha = Float(alpha)
            return (field.texture, uniform, field.previousTimestamp)
        }
    }

    // Render thread: the per-4x4-pixel motion choice for the in-betweens of the frame at timestamp
    // (luma planes of it and the previous frame, its motion field from renderState), encoded into
    // the display's command buffer on the frame's first display and reused after that.
    // The choice compares the two frames on their PyroWave lowpass level 0 (half resolution, copied
    // when the frames were decoded) instead of the full pictures: at 7040x2496 the full resolution
    // choice cost ~20 ms of GPU per frame (2026-10-09); offline the half resolution one with 8x8
    // blocks chose as well on the HUD, menu and panel recordings.
    func selection(commandBuffer: MTLCommandBuffer, field: MTLTexture, timestamp: UInt64, previousTimestamp: UInt64) -> MTLTexture? {
        if let s = selectionFor, s.timestamp == timestamp, s.previous == previousTimestamp {
            return s.texture
        }
        let (currentLL, previousLL): (MTLTexture?, MTLTexture?) = lock.withLock {
            (captures.first(where: { $0.timestamp == timestamp && $0.hasLevel0 })?.lowpass[0],
             captures.first(where: { $0.timestamp == previousTimestamp && $0.hasLevel0 })?.lowpass[0])
        }
        // A command buffer of its own on the renderer's queue, committed now, so before the
        // display's (same queue, so it runs first) and with its own GPU time for the log.
        guard let pipeline = selectPipeline, selectionTextures.count == 3, let currentLL, let previousLL,
              let selectBuffer = commandBuffer.commandQueue.makeCommandBuffer(),
              let encoder = selectBuffer.makeComputeCommandEncoder() else {
            return nil
        }
        selectBuffer.label = "Frame interpolation, choose motion"
        let currentLuma = currentLL, previousLuma = previousLL
        let texture = selectionTextures[selectionNext]
        selectionNext = (selectionNext + 1) % selectionTextures.count
        struct Params {
            var frameSize: SIMD2<Float>
            var cellPixels: Float
            var cellsPerEye: Int32
            var gridRows: Int32
            var iterations: Int32
            var candidates: Int32
            var samplesPerSide: Int32
        }
        let settings = ALVRClientApp.gStore.settings
        var params = Params(frameSize: SIMD2(Float(width), Float(height)), cellPixels: Float(Self.cellPixels),
                            cellsPerEye: Int32(cellsPerEye), gridRows: Int32(gridRows), iterations: Self.warpIterations,
                            candidates: Int32(min(max(settings.frameInterpolationSelectCandidates.rounded(), 3), 7)),
                            samplesPerSide: settings.frameInterpolationSelectGrid >= 2.5 ? 3 : 2)
        encoder.label = "Frame interpolation, choose motion"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(currentLuma, index: 0)
        encoder.setTexture(previousLuma, index: 1)
        encoder.setTexture(field, index: 2)
        encoder.setTexture(texture, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 0)
        encoder.dispatchThreads(MTLSize(width: texture.width, height: texture.height, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        encoder.endEncoding()
        selectBuffer.addCompletedHandler { GpuSplitDiag.shared.recordFx(start: $0.gpuStartTime, end: $0.gpuEndTime) }
        selectBuffer.commit()
        selectionFor = (timestamp, previousTimestamp, texture)
        return texture
    }

    // Interpolation needs the motion of a new frame before its first display. True while that
    // motion is still being estimated, or the streamer's vectors for it have not come yet while
    // the streamer sends vectors (they follow their frame, FrameMotion.h on the streamer), and the
    // frame has waited less than a display period, or for the streamer's vectors 90 % of the
    // frames' interval, as long as the streamer used to hold the frame for them: the renderer then
    // shows the current frame once more instead (counted as held back).
    func shouldHoldBack(timestamp: UInt64, queuedTime: Double) -> Bool {
        // Predicting, a new frame is shown as decoded on its first display: nothing to wait for.
        if ALVRClientApp.gStore.settings.frameInterpolationPredict {
            return false
        }
        let now = CACurrentMediaTime()
        return lock.withLock {
            guard queuedTime > 0 else {
                return false
            }
            let streamerSends = serverMotionTimestamp != 0 && timestamp >= serverMotionTimestamp
                && timestamp - serverMotionTimestamp < 500_000_000
            let limit = streamerSends ? max(displayPeriod, 0.9 * min(serverMotionInterval, 0.1)) : displayPeriod
            if let field = fields.first(where: { $0.timestamp == timestamp }) {
                return !field.ready && now - queuedTime < limit
            }
            // Not begun to arrive: a frame newer than the last one with vectors.
            return streamerSends && timestamp > serverMotionTimestamp && now - queuedTime < limit
        }
    }

    // MARK: - Decoder side (PyroWave decode worker queue)

    // Encodes the copy of the decoder's lowpass bands into the frame's final decode command
    // buffer, after the decode work. Returns false if nothing was captured.
    @discardableResult
    func capture(decoder: pyrowave_decoder, commandBuffer: MTLCommandBuffer, timestamp: UInt64) -> Bool {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            return false
        }
        let slot = captures[nextCapture]
        nextCapture = (nextCapture + 1) % captures.count
        blit.label = "Frame extrapolation capture"
        // Level 0 (half resolution) for the motion choice per block; levels finestLevel and coarser
        // only for the estimation here, which the streamer's vectors replace. The decoder reuses its
        // bands for the next frame, so what is read later has to be copied (8.8 MB at 7040x2496).
        let settings = ALVRClientApp.gStore.settings
        let ownEstimate = !settings.useServerMotion
        let choose = settings.frameInterpolation && settings.frameInterpolationSelect
        let copyLevels = (choose ? [0] : []) + (ownEstimate ? Array(Self.finestLevel..<Self.levels) : [])
        for level in copyLevels {
            guard let raw = pyrowave_decoder_get_lowpass_texture(decoder, 0, Int32(level)) else {
                blit.endEncoding()
                return false
            }
            let view = Unmanaged<MTLTexture>.fromOpaque(raw).takeUnretainedValue()
            let target = slot.lowpass[level]
            blit.copy(from: view, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(),
                      sourceSize: MTLSize(width: target.width, height: target.height, depth: 1),
                      to: target, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin())
        }
        blit.endEncoding()
        captureEventValue += 1
        commandBuffer.encodeSignalEvent(captureEvent, value: captureEventValue)
        lock.withLock {
            slot.timestamp = timestamp
            slot.hasLevel0 = choose
        }
        slot.eventValue = captureEventValue

        let previous = lastCapture
        lastCapture = slot
        // No estimate of its own for a frame whose motion came from the streamer.
        let fromServer = lock.withLock { fields.contains(where: { $0.timestamp == timestamp && $0.fromServer }) }
        // With "Use Server Motion Vectors" the streamer's vectors are on their way even when they
        // arrive after the frame is decoded (at 7040x2496 they are ready 15 ms after the frame's
        // copy); estimating here as well cost 12-34 ms of GPU per frame for nothing (2026-10-09).
        if let previous, previous.timestamp < timestamp, !fromServer, !ALVRClientApp.gStore.settings.useServerMotion {
            pendingEstimate = (slot, previous)
        }
        return true
    }

    // Commits the motion estimation for the frame just captured. Call it after committing the
    // command buffer capture() encoded into: the estimation waits for that buffer's event, and
    // Metal needs the signalling command buffer committed before the waiting one. Committed the
    // other way round, decode and estimation hung for good (2026-10-07, no picture at all).
    func commitPendingEstimate() {
        guard let (newer, older) = pendingEstimate else {
            return
        }
        pendingEstimate = nil
        estimate(newer: newer, older: older)
    }

    private func estimate(newer: Capture, older: Capture) {
        let interval = Double(newer.timestamp - older.timestamp) / 1e9
        let (period, geometry) = lock.withLock { (displayPeriod, self.geometry) }
        let foveation = Self.foveationLock.withLock { Self.foveation }
        guard interval > Self.slowStreamFactor * period, interval < 0.1, let geometry, geometry.tangents.count >= 2 else {
            return
        }
        let field = reserveField(timestamp: newer.timestamp, previousTimestamp: older.timestamp, fromServer: false)

        guard let cmd = queue.makeCommandBuffer() else {
            return
        }
        cmd.label = "Frame extrapolation"
        cmd.encodeWaitForEvent(captureEvent, value: newer.eventValue)
        guard let enc = cmd.makeComputeCommandEncoder() else {
            return
        }

        encodeRotationField(enc, into: rotationField, older: older.timestamp, newer: newer.timestamp, geometry: geometry, foveation: foveation)
        enc.memoryBarrier(scope: .textures)

        for level in stride(from: Self.levels - 1, through: Self.finestLevel, by: -1) {
            let coarsest = level == Self.levels - 1
            let eyeWidth = (width / 2) >> (level + 1)
            let cpe = Self.cellsPerEye(width: width, level: level)
            let rows = Self.gridRows(height: height, level: level)
            var prm = SearchParams(
                levelSize: SIMD2(Int32(2 * eyeWidth), Int32(height >> (level + 1))),
                eyeWidth: Int32(eyeWidth),
                cellsPerEye: Int32(cpe),
                gridSize: SIMD2(Int32(2 * cpe), Int32(rows)),
                parentCellsPerEye: coarsest ? 0 : Int32(Self.cellsPerEye(width: width, level: level + 1)),
                parentGridSize: coarsest ? SIMD2(0, 0) : SIMD2(Int32(2 * Self.cellsPerEye(width: width, level: level + 1)),
                                                             Int32(Self.gridRows(height: height, level: level + 1))),
                searchRadius: coarsest ? Self.coarseRadius : Self.refineRadius,
                pixelsPerTexel: Float(2 << level),
                valueScale: Self.valueScale,
                lambda: Self.lambda,
                subpixel: level == Self.finestLevel ? 1 : 0,
                predictedShift: SIMD2(0, 0),
                rotationCellsPerEye: Int32(cellsPerEye),
                rotationCellScale: Int32(1 << (level - Self.finestLevel)),
                debugStop: 0,
                confidenceRatio: Self.confidenceRatio,
                confidenceMinGain: Self.confidenceMinGain)
            enc.setComputePipelineState(coarsest ? coarsePipeline : refinePipeline)
            enc.setTexture(newer.lowpass[level], index: 0)
            enc.setTexture(older.lowpass[level], index: 1)
            enc.setTexture(coarsest ? levelFields[level] : levelFields[level + 1], index: 2)
            enc.setTexture(levelFields[level], index: 3)
            enc.setTexture(rotationField, index: 4)
            enc.setBytes(&prm, length: MemoryLayout<SearchParams>.stride, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: (2 * cpe + 3) / 4, height: rows, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.memoryBarrier(scope: .textures)
        }

        var cpe = Int32(cellsPerEye)
        enc.setComputePipelineState(medianPipeline)
        enc.setTexture(levelFields[Self.finestLevel], index: 0)
        enc.setTexture(field.texture, index: 1)
        enc.setTexture(rotationField, index: 2)
        enc.setBytes(&cpe, length: 4, index: 0)
        var deadzone = Self.deadzonePixels
        enc.setBytes(&deadzone, length: 4, index: 1)
        enc.dispatchThreads(MTLSize(width: field.texture.width, height: field.texture.height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        enc.endEncoding()

        let timestamp = newer.timestamp
        cmd.addCompletedHandler { [weak self] buffer in
            guard let self else { return }
            let ok = buffer.status == .completed
            self.lock.withLock {
                // The slot may already belong to a newer frame.
                if field.timestamp == timestamp {
                    field.ready = ok
                }
            }
        }
        cmd.addCompletedHandler { GpuSplitDiag.shared.recordFx(start: $0.gpuStartTime, end: $0.gpuEndTime) }
        cmd.commit()
    }

    // MARK: - Shared by both motion sources

    // Takes the next field slot for the frame with this timestamp (it may still hold an old frame
    // the renderer no longer shows).
    private func reserveField(timestamp: UInt64, previousTimestamp: UInt64, fromServer: Bool) -> Field {
        lock.withLock {
            let f = fields[nextField]
            nextField = (nextField + 1) % fields.count
            f.ready = false
            f.fromServer = fromServer
            f.timestamp = timestamp
            f.previousTimestamp = previousTimestamp
            return f
        }
    }

    // The foveation tables for the rotation kernel, rebuilt when the stream's foveation changes.
    private func foveationTablesFor(_ foveation: Foveation) -> MTLBuffer? {
        tablesLock.withLock {
            if foveationTables == nil || foveationTablesFor != foveation {
                foveationTables = makeFoveationTables(foveation)
                foveationTablesFor = foveation
            }
            return foveationTables
        }
    }

    // Encodes fxRotationField: the motion head rotation alone explains between the frames with
    // these timestamps, from the poses they were rendered with (identity if one is unknown).
    private func encodeRotationField(_ enc: MTLComputeCommandEncoder, into target: MTLTexture, older: UInt64, newer: UInt64,
                                     geometry: ViewGeometry, foveation: Foveation) {
        var rotation = matrix_identity_float3x3
        if let olderPose = FramePoseHistory.shared.orientation(timestampNs: older),
           let newerPose = FramePoseHistory.shared.orientation(timestampNs: newer) {
            rotation = FrameExtrapolationGeometry.relativeRotation(older: olderPose, newer: newerPose)
        }
        var rp = RotationParams(rotation: rotation, tangentsLeft: geometry.tangents[0], tangentsRight: geometry.tangents[1],
                                frameSize: SIMD2(Float(width), Float(height)),
                                eyeSizeRatio: foveation.enabled ? foveation.eyeSizeRatio : SIMD2(1, 1),
                                cellPixels: Float(Self.cellPixels), cellsPerEye: Int32(cellsPerEye),
                                lutSize: foveation.enabled ? Int32(FrameExtrapolationGeometry.lutSize) : 0)
        enc.setComputePipelineState(rotationPipeline)
        enc.setTexture(target, index: 0)
        enc.setBytes(&rp, length: MemoryLayout<RotationParams>.stride, index: 0)
        enc.setBuffer(foveationTablesFor(foveation), offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: target.width, height: target.height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
    }

    // MARK: - Motion vectors from the streamer (network thread)

    // One motion packet of the frame with this timestamp (PyroWaveStripes::MotionHeader, see
    // pyrowave_stripes.hpp): the streamer's NVIDIA Optical Flow vectors, from this frame back to the
    // reference frame, int16 in 1/32 pixels. Once all rows are in, they become the frame's field,
    // with head rotation taken out as for the field estimated here, and the frame skips its own
    // estimation. Called with the whole packet, PacketPrefix included.
    // previousFrame: the frame this client decoded before this one. When that is older than the
    // streamer's reference (it skipped frames, as "Test: Half Frame Rate" does), the vectors are
    // scaled up to it, assuming steady motion.
    func receiveServerMotion(packet: UnsafeRawBufferPointer, timestamp: UInt64, previousFrame: UInt64) {
        let prefixBytes = 16, headerBytes = 32
        guard packet.count >= prefixBytes + headerBytes else { return }
        func u16(_ offset: Int) -> Int { Int(UInt16(littleEndian: packet.loadUnaligned(fromByteOffset: offset, as: UInt16.self))) }
        func u32(_ offset: Int) -> UInt32 { UInt32(littleEndian: packet.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
        let h = prefixBytes
        if u32(h) == 0x434d_5750 {
            receiveServerChoice(packet: packet, timestamp: timestamp)
            return
        }
        let version = u32(h + 4)
        guard u32(h) == 0x564d_5750, version == 1 || version == 2 else { return }
        let reference = UInt64(littleEndian: packet.loadUnaligned(fromByteOffset: h + 8, as: UInt64.self))
        let gridWidth = u16(h + 16), gridHeight = u16(h + 18), cell = u16(h + 20)
        let firstRow = u16(h + 22), rows = u16(h + 24)
        guard gridWidth == 2 * cellsPerEye, gridHeight == gridRows, cell == Self.cellPixels else {
            if !serverMismatchLogged {
                serverMismatchLogged = true
                pyroLog("Frame extrapolation: the streamer's motion vectors are \(gridWidth)x\(gridHeight) every \(cell) pixels, this field \(2 * cellsPerEye)x\(gridRows) every \(Self.cellPixels); ignoring them")
            }
            return
        }
        guard firstRow + rows <= gridHeight, version == 2 || packet.count >= prefixBytes + headerBytes + rows * gridWidth * 4,
              reference < timestamp else { return }
        let base = (previousFrame > 0 && previousFrame < reference) ? previousFrame : reference
        let scale = Float(Double(timestamp - base) / Double(timestamp - reference))

        if serverTimestamp != timestamp {
            // A new frame: the previous one's rows are complete or given up on.
            if serverField != nil {
                MotionTimingDiag.shared.dropped("rows missing")
            }
            MotionTimingDiag.shared.note(.motionBegan, timestamp: timestamp)
            serverTimestamp = timestamp
            serverRowsReceived = [Bool](repeating: false, count: gridHeight)
            serverRowCount = 0
            serverUploadCurrent = serverUploadNext
            serverUploadNext = (serverUploadNext + 1) % serverUploads.count
            let target = serverUploadBuffers[serverUploadCurrent]
            memset(target.contents(), 0, target.length)
            let taken = lock.withLock {
                serverMotionTimestamp = max(serverMotionTimestamp, timestamp)
                serverMotionInterval = Double(timestamp - reference) / 1e9
                return fields.contains(where: { $0.timestamp == timestamp })
            }
            serverField = taken ? nil : reserveField(timestamp: timestamp, previousTimestamp: base, fromServer: true)
            if taken {
                MotionTimingDiag.shared.dropped("own estimate first")
            }
        }
        guard serverField != nil else { return } // the frame's own estimate came first
        let dataStart = prefixBytes + headerBytes
        let flow = serverUploadBuffers[serverUploadCurrent].contents().assumingMemoryBound(to: Float.self)
        // Version 2: the rows coded one after the other (pyrowave_stripes.hpp), in 1/8 pixel steps.
        var reader = dataStart
        let stepPixels = scale / 8
        for r in 0..<rows {
            let row = flow + (firstRow + r) * serverUploadRowFloats
            let fresh = !serverRowsReceived[firstRow + r]
            if version == 2 {
                // Decoded also when the row is in already, to find where the next one starts.
                var x = 0, px: Int32 = 0, py: Int32 = 0
                func cell(_ vx: Int32, _ vy: Int32) {
                    if fresh {
                        row[x * 4] = -Float(vx) * stepPixels
                        row[x * 4 + 1] = -Float(vy) * stepPixels
                        row[x * 4 + 2] = 1
                    }
                    x += 1
                }
                func varint() -> Int32? {
                    var value: UInt32 = 0, shift: UInt32 = 0
                    while reader < packet.count, shift < 35 {
                        let byte = packet[reader]
                        reader += 1
                        value |= UInt32(byte & 0x7f) << shift
                        if byte < 0x80 {
                            return Int32(bitPattern: (value >> 1) ^ (0 &- (value & 1)))
                        }
                        shift += 7
                    }
                    return nil
                }
                while x < gridWidth {
                    guard reader < packet.count else { return }
                    let token = Int(packet[reader])
                    reader += 1
                    if token < 0x40 {
                        // Unreliable cells keep the zero the upload was cleared to.
                        x = min(gridWidth, x + token + 1)
                    } else if token < 0x80 {
                        for _ in 0..<min(token - 0x3f, gridWidth - x) {
                            cell(px, py)
                        }
                    } else if token < 0xc0 {
                        px += Int32((token >> 3) & 7) - 4
                        py += Int32(token & 7) - 4
                        cell(px, py)
                    } else {
                        guard token == 0xc0, let dx = varint(), let dy = varint() else { return }
                        px &+= dx
                        py &+= dy
                        cell(px, py)
                    }
                }
            } else {
                for x in 0..<gridWidth where fresh {
                    let offset = dataStart + (r * gridWidth + x) * 4
                    let fx = Int16(littleEndian: packet.loadUnaligned(fromByteOffset: offset, as: Int16.self))
                    let fy = Int16(littleEndian: packet.loadUnaligned(fromByteOffset: offset + 2, as: Int16.self))
                    // The vectors point back to where the content was: motion is their negative. x =
                    // Int16.min marks a flat cell without reliable motion (MotionUnreliable).
                    if fx != Int16.min {
                        row[x * 4] = -Float(fx) / 32 * scale
                        row[x * 4 + 1] = -Float(fy) / 32 * scale
                        row[x * 4 + 2] = 1
                    }
                }
            }
            if fresh {
                serverRowsReceived[firstRow + r] = true
                serverRowCount += 1
            }
        }
        if serverRowCount == gridHeight, let field = serverField {
            serverField = nil
            finishServerField(field, timestamp: timestamp, reference: base)
        }
    }

    // One packet of the streamer's motion choice of the frame with this timestamp (PacketPrefix and
    // MotionHeader as for the vectors, magic "PWMC"): grid rows of 2 bits per block, four blocks per
    // byte, the first in the low bits. Once all rows are in, streamerChoice hands it to the renderer.
    private func receiveServerChoice(packet: UnsafeRawBufferPointer, timestamp: UInt64) {
        let prefixBytes = 16, headerBytes = 32, h = prefixBytes
        func u16(_ offset: Int) -> Int { Int(UInt16(littleEndian: packet.loadUnaligned(fromByteOffset: offset, as: UInt16.self))) }
        let version = UInt32(littleEndian: packet.loadUnaligned(fromByteOffset: h + 4, as: UInt32.self))
        guard version == 1 || version == 2, let texture = choiceTextures.first else { return }
        let reference = UInt64(littleEndian: packet.loadUnaligned(fromByteOffset: h + 8, as: UInt64.self))
        let gridWidth = u16(h + 16), gridHeight = u16(h + 18), block = u16(h + 20)
        let firstRow = u16(h + 22), rows = u16(h + 24)
        guard gridWidth == texture.width, gridHeight == texture.height, block == Int(FX_SELECT_BLOCK) else {
            if !choiceMismatchLogged {
                choiceMismatchLogged = true
                pyroLog("Frame interpolation: the streamer's motion choice is \(gridWidth)x\(gridHeight) blocks of \(block) pixels, the selection here \(texture.width)x\(texture.height) of \(FX_SELECT_BLOCK); ignoring it")
            }
            return
        }
        let rowBytes = (gridWidth + 3) / 4
        guard firstRow + rows <= gridHeight, version == 2 || packet.count >= prefixBytes + headerBytes + rows * rowBytes else { return }
        if choiceTimestamp != timestamp {
            // A new frame: take the next slot, and forget a complete choice that still names it.
            choiceTimestamp = timestamp
            choiceSlot = choiceNext
            choiceNext = (choiceNext + 1) % choiceTextures.count
            choiceRowsReceived = [Bool](repeating: false, count: gridHeight)
            choiceRowCount = 0
            let slot = choiceSlot
            lock.withLock { choicesComplete.removeAll { $0.slot == slot } }
        }
        // Half floats 0, 1 and 2 in .r (.g stays 0), as fxSelectMotion writes them.
        let values: [UInt16] = [0x0000, 0x3c00, 0x4000, 0x4000]
        let out = choiceBuffers[choiceSlot].contents().assumingMemoryBound(to: UInt16.self)
        let dataStart = prefixBytes + headerBytes
        // Version 2: each row as runs of one byte each, choice << 6 | (length - 1).
        var reader = dataStart
        for r in 0..<rows {
            let fresh = !choiceRowsReceived[firstRow + r]
            let row = out + (firstRow + r) * choiceRowHalves
            if version == 2 {
                var x = 0
                while x < gridWidth {
                    guard reader < packet.count else { return }
                    let byte = packet[reader]
                    reader += 1
                    let end = min(gridWidth, x + Int(byte & 0x3f) + 1)
                    if fresh {
                        let value = values[Int(byte >> 6)]
                        for i in x..<end {
                            row[2 * i] = value
                            row[2 * i + 1] = 0
                        }
                    }
                    x = end
                }
            } else if fresh {
                let source = dataStart + r * rowBytes
                for x in 0..<gridWidth {
                    let byte = packet[source + x / 4]
                    row[2 * x] = values[Int((byte >> UInt8(2 * (x % 4))) & 3)]
                    row[2 * x + 1] = 0
                }
            }
            if fresh {
                choiceRowsReceived[firstRow + r] = true
                choiceRowCount += 1
            }
        }
        if choiceRowCount == gridHeight {
            let slot = choiceSlot
            lock.withLock {
                choicesComplete.append((timestamp, reference, slot))
                if choicesComplete.count > choiceTextures.count {
                    choicesComplete.removeFirst()
                }
            }
            tryPrecompute()
        }
    }

    // Whether the streamer sends motion choices at all (one within half a second of timestamp), so a
    // frame without one has lost it rather than the streamer not making any.
    func streamerSendsChoice(near timestamp: UInt64) -> Bool {
        lock.withLock {
            choicesComplete.contains { $0.timestamp &+ 500_000_000 > timestamp && timestamp &+ 500_000_000 > $0.timestamp }
        }
    }

    // Render thread: the streamer's motion choice for the in-betweens of the frame at timestamp
    // after previousTimestamp, if it has fully arrived; it replaces selection(...) then.
    func streamerChoice(timestamp: UInt64, previousTimestamp: UInt64) -> MTLTexture? {
        lock.withLock {
            choicesComplete.last(where: { $0.timestamp == timestamp && $0.reference == previousTimestamp })
                .map { choiceTextures[$0.slot] }
        }
    }

    // MARK: - In-betweens made ahead (Apple's private formats, Blend Both Frames)

    // The render queue (Renderer sets it): the in-between made on it is done before any render
    // committed after it, so a display that samples it never waits for or races it.
    static var renderQueue: MTLCommandQueue?
    private var precomputePipelines: [Int32: MTLComputePipelineState] = [:]
    private var precomputeTextures: [MTLTexture] = []
    private var precomputeNext = 0
    private var precomputeTextureCache: CVMetalTextureCache?
    // Pipelines and textures, made on first use by whichever thread gets there.
    private let precomputeLock = NSLock()
    // Under lock: the newest decoded frame and the one before it, the pair last started, and the
    // in-betweens committed (newest last).
    private var precomputePair: (newer: QueuedFrame, previous: QueuedFrame)?
    private var precomputeStarted: UInt64 = 0
    private var precomputed: [(timestamp: UInt64, previous: UInt64, alpha: Float, mode: Int32, slot: Int)] = []

    // Any thread: a frame was decoded (VideoToolbox), previous the one decoded before it.
    func noteDecoded(_ frame: QueuedFrame, previous: QueuedFrame?) {
        // Disabled: keep no decoded frames alive for it (they would hold decoder buffers).
        guard Self.precomputeEnabled, let previous else { return }
        lock.withLock { precomputePair = (frame, previous) }
        tryPrecompute()
    }

    // The in-between made ahead for this display, if one was committed for the same pair, mode and
    // (within 0.08) alpha.
    func precomputedInBetween(timestamp: UInt64, previousTimestamp: UInt64, alpha: Float, mode: Int32) -> MTLTexture? {
        lock.withLock {
            precomputed.last(where: { $0.timestamp == timestamp && $0.previous == previousTimestamp && $0.mode == mode
                                      && abs($0.alpha - alpha) < 0.08 })
                .map { precomputeTextures[$0.slot] }
        }
    }

    // Any thread, after a frame was decoded, its field got ready or its choice complete: makes the
    // newest frame's in-between once all three are there.
    // Off (2026-10-14): made ahead at the stream's full size the in-between cost 4-6 ms of the render
    // queue, as much as blending in the render pass (which only shades the pixels drawn, coarser at
    // the edges) and right in front of the other displays' renders; many in-betweens still blended
    // in the render pass on top. Repeats went from 2-3 to up to 9 ms, deadline misses up.
    static let precomputeEnabled = false

    func tryPrecompute() {
        let settings = ALVRClientApp.gStore.settings
        guard Self.precomputeEnabled, settings.frameInterpolation, settings.frameInterpolationBlend,
              let renderQueue = Self.renderQueue else {
            return
        }
        let streamerChoice = settings.frameInterpolationStreamerChoice && settings.useServerMotion
        var job: (newer: QueuedFrame, previous: QueuedFrame, field: MTLTexture, choice: MTLTexture?, alpha: Float, slot: Int)?
        lock.withLock {
            guard let pair = precomputePair, pair.newer.timestamp != precomputeStarted,
                  let field = fields.first(where: { $0.timestamp == pair.newer.timestamp }), field.ready,
                  field.previousTimestamp == pair.previous.timestamp else {
                return
            }
            let interval = Double(pair.newer.timestamp - pair.previous.timestamp) / 1e9
            guard interval > Self.slowStreamFactor * displayPeriod, interval < 0.1 else { return }
            // The alpha of the frame's first display on the content clock (renderState): -0.5 at
            // 45 fps on 90 Hz.
            let alpha = Float(-(interval - displayPeriod) / interval)
            guard alpha < -0.1 else { return }
            var choice: MTLTexture?
            if streamerChoice {
                if let c = choicesComplete.last(where: { $0.timestamp == pair.newer.timestamp && $0.reference == pair.previous.timestamp }) {
                    choice = choiceTextures[c.slot]
                } else if choicesComplete.contains(where: { $0.timestamp &+ 500_000_000 > pair.newer.timestamp }) {
                    return // the streamer sends choices: wait for this one's
                }
            }
            precomputeStarted = pair.newer.timestamp
            let slot = precomputeNext
            precomputeNext = (precomputeNext + 1) % 2
            job = (pair.newer, pair.previous, field.texture, choice, alpha, slot)
        }
        guard let job else { return }

        let format = CVPixelBufferGetPixelFormatType(job.newer.imageBuffer)
        guard !PyroWaveDecoder.isPyroWaveFrame(job.newer.imageBuffer), VideoHandler.isFormatSecret(format) else {
            return
        }
        let split = job.newer.rightImageBuffer != nil
        guard split == (job.previous.rightImageBuffer != nil),
              let pipeline = precomputeLock.withLock({ precomputePipeline(filter: Renderer.videoFilterIndex(settings)) }),
              precomputeLock.withLock({ ensurePrecomputeTextures() }) else {
            return
        }
        let textureType = VideoHandler.getTextureTypesForFormat(format)[0]
        func texture(_ buffer: CVImageBuffer?) -> (CVMetalTexture, MTLTexture)? {
            guard let buffer, let cache = precomputeTextureCache else { return nil }
            var out: CVMetalTexture?
            guard CVMetalTextureCacheCreateTextureFromImage(nil, cache, buffer, nil, textureType,
                                                            CVPixelBufferGetWidthOfPlane(buffer, 0),
                                                            CVPixelBufferGetHeightOfPlane(buffer, 0), 0, &out) == kCVReturnSuccess,
                  let out, let texture = CVMetalTextureGetTexture(out) else { return nil }
            return (out, texture)
        }
        guard let curL = texture(job.newer.imageBuffer), let prevL = texture(job.previous.imageBuffer) else { return }
        let curR = split ? texture(job.newer.rightImageBuffer) : curL
        let prevR = split ? texture(job.previous.rightImageBuffer) : prevL
        guard let curR, let prevR,
              let cmd = renderQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else {
            return
        }
        cmd.label = "In-between made ahead"
        var uniform = FrameExtrapolationUniform(frameSize: SIMD2(Float(width), Float(height)), alpha: job.alpha,
                                                cellPixels: Float(Self.cellPixels), cellsPerEye: Int32(cellsPerEye),
                                                gridRows: Int32(gridRows), iterations: Self.warpIterations,
                                                blendPrevious: job.choice != nil ? 2 : 1)
        var splitEyes: Int32 = split ? 1 : 0
        let out = precomputeTextures[job.slot]
        enc.setComputePipelineState(pipeline)
        enc.setTexture(curL.1, index: 0)
        enc.setTexture(curR.1, index: 1)
        enc.setTexture(prevL.1, index: 2)
        enc.setTexture(prevR.1, index: 3)
        enc.setTexture(job.field, index: 4)
        enc.setTexture(job.choice ?? idleTexture, index: 5)
        enc.setTexture(out, index: 6)
        enc.setBytes(&uniform, length: MemoryLayout<FrameExtrapolationUniform>.stride, index: 0)
        enc.setBytes(&splitEyes, length: MemoryLayout<Int32>.size, index: 1)
        enc.dispatchThreads(MTLSize(width: out.width, height: out.height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        enc.endEncoding()
        // The CVMetalTextures (and through them the pixel buffers) stay alive until the GPU is done.
        let keep = [curL.0, curR.0, prevL.0, prevR.0]
        cmd.addCompletedHandler { buffer in
            _ = keep
            GpuSplitDiag.shared.recordFx(start: buffer.gpuStartTime, end: buffer.gpuEndTime)
        }
        cmd.commit()
        let mode = uniform.blendPrevious
        lock.withLock {
            precomputed.append((job.newer.timestamp, job.previous.timestamp, job.alpha, mode, job.slot))
            if precomputed.count > 2 {
                precomputed.removeFirst()
            }
        }
    }

    private func precomputePipeline(filter: Int32) -> MTLComputePipelineState? {
        if let pipeline = precomputePipelines[filter] {
            return pipeline
        }
        let constants = MTLFunctionConstantValues()
        var value = filter
        constants.setConstantValue(&value, type: .int, index: ALVRFunctionConstant.videoFilter.rawValue)
        guard let function = try? device.makeDefaultLibrary()?.makeFunction(name: "fxPrecomputeInBetween", constantValues: constants),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            pyroLog("Frame interpolation: cannot build the in-between kernel")
            return nil
        }
        precomputePipelines[filter] = pipeline
        return pipeline
    }

    // Two 10 bit textures of the stream's size: one shown while the next is made.
    private func ensurePrecomputeTextures() -> Bool {
        if precomputeTextures.count == 2, precomputeTextureCache != nil {
            return true
        }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgb10a2Unorm, width: width, height: height, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        precomputeTextures = (0..<2).compactMap { slot in
            let t = device.makeTexture(descriptor: d)
            t?.label = "In-between \(slot)"
            return t
        }
        if precomputeTextureCache == nil {
            CVMetalTextureCacheCreate(nil, nil, device, nil, &precomputeTextureCache)
        }
        return precomputeTextures.count == 2 && precomputeTextureCache != nil
    }

    private func finishServerField(_ field: Field, timestamp: UInt64, reference: UInt64) {
        let (geometry, displayPeriod) = lock.withLock { (self.geometry, self.displayPeriod) }
        let foveation = Self.foveationLock.withLock { Self.foveation }
        let interval = Double(timestamp - reference) / 1e9
        guard let geometry, geometry.tangents.count >= 2, interval > Self.slowStreamFactor * displayPeriod, interval < 0.1 else {
            // Not a slow stream (or no geometry yet): nothing to show in between.
            MotionTimingDiag.shared.dropped(geometry == nil ? "no geometry"
                : String(format: "interval %.0f ms, display %.1f ms", interval * 1000, displayPeriod * 1000))
            lock.withLock { if field.timestamp == timestamp { field.timestamp = 0 } }
            return
        }
        let upload = serverUploads[serverUploadCurrent]
        guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else {
            return
        }
        cmd.label = "Frame extrapolation, streamer motion"
        encodeRotationField(enc, into: serverRotationField, older: reference, newer: timestamp, geometry: geometry, foveation: foveation)
        enc.memoryBarrier(scope: .textures)
        // Flat cells take their reliable neighbours' motion, then the same 3x3 median as for the field
        // estimated here, rotation subtracted: single outliers at text edges would otherwise drag
        // picture pieces along.
        var cpe = Int32(cellsPerEye)
        enc.setComputePipelineState(fillPipeline)
        enc.setTexture(upload, index: 0)
        enc.setTexture(serverRotationField, index: 1)
        enc.setTexture(serverFilled, index: 2)
        enc.setBytes(&cpe, length: 4, index: 0)
        enc.dispatchThreads(MTLSize(width: serverFilled.width, height: serverFilled.height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        enc.memoryBarrier(scope: .textures)
        enc.setComputePipelineState(medianPipeline)
        enc.setTexture(serverFilled, index: 0)
        enc.setTexture(field.texture, index: 1)
        enc.setTexture(serverRotationField, index: 2)
        enc.setBytes(&cpe, length: 4, index: 0)
        var deadzone = Self.deadzonePixels
        enc.setBytes(&deadzone, length: 4, index: 1)
        enc.dispatchThreads(MTLSize(width: field.texture.width, height: field.texture.height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        enc.endEncoding()
        cmd.addCompletedHandler { [weak self] buffer in
            guard let self else { return }
            self.lock.withLock {
                if field.timestamp == timestamp {
                    field.ready = buffer.status == .completed
                }
            }
            MotionTimingDiag.shared.note(.fieldReady, timestamp: timestamp)
            self.tryPrecompute()
        }
        cmd.addCompletedHandler { GpuSplitDiag.shared.recordFx(start: $0.gpuStartTime, end: $0.gpuEndTime) }
        cmd.commit()
    }

    private func makeFoveationTables(_ foveation: Foveation) -> MTLBuffer? {
        guard foveation.enabled else {
            // The kernel does not read the tables then, but Metal wants a buffer bound.
            return device.makeBuffer(length: 16, options: .storageModeShared)
        }
        let tables = FrameExtrapolationGeometry.foveationTables(centerSize: foveation.centerSize,
                                                                centerShift: foveation.centerShift,
                                                                edgeRatio: foveation.edgeRatio)
        return device.makeBuffer(bytes: tables, length: tables.count * MemoryLayout<Float>.size, options: .storageModeShared)
    }

}

// Debug capture (2026-10-12, frame interpolation): records the finished picture of the next
// displays as the app hands it to visionOS, once the file Documents/display_trigger appears
// (pushed with devicectl; its content may give the number of displays, default 24, at most 60).
// Per display a piece of the left eye's render target (physical pixels of the foveated render
// target, as is: rgba16Float, linear) around the foveation's full-rate center, copied with a blit
// before the present, and a line of what the interpolation did. Written to
// Documents/display_capture.bin (pieces back to back) and display_capture.csv. Costs a file check
// every 0.5 s while idle.
final class DisplayRecorder {
    static let shared = DisplayRecorder()

    // The full-rate center of the left eye in screen fractions ("Rate map" in the log, static).
    private static let center = (x: 0.634, y: 0.458)
    private static let side = 1024

    private let lock = NSLock()
    private let queue = DispatchQueue(label: "DisplayRecorder", qos: .utility)
    private var timer: DispatchSourceTimer?
    private var remaining = 0
    private var buffers: [MTLBuffer?] = []
    // The motion field (rgba16Float: scene motion, head rotation motion) and the choice (rg16Float)
    // each display used; nil where it had none.
    private var fieldBuffers: [MTLBuffer?] = []
    private var choiceBuffers: [MTLBuffer?] = []
    private var fieldSize = (w: 0, h: 0)
    private var choiceSize = (w: 0, h: 0)
    private var lines: [String] = []
    private var outstanding = 0
    private var piece = (x: 0, y: 0, w: 0, h: 0, bytesPerPixel: 0, format: "")

    private static var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private func startIfNeeded() {
        guard timer == nil else { return }
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 0.5, repeating: 0.5)
        t.setEventHandler { [weak self] in self?.poll() }
        t.resume()
        timer = t
    }

    // Recorder queue.
    private func poll() {
        let trigger = Self.documents.appendingPathComponent("display_trigger")
        guard FileManager.default.fileExists(atPath: trigger.path) else { return }
        let text = (try? String(contentsOf: trigger, encoding: .utf8)) ?? ""
        try? FileManager.default.removeItem(at: trigger)
        let count = min(max(Int(text.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 24, 1), 60)
        lock.withLock {
            guard remaining == 0, outstanding == 0 else {
                pyroLog("Display capture: trigger ignored, still capturing")
                return
            }
            buffers = []
            fieldBuffers = []
            choiceBuffers = []
            lines = ["index,vsync_s,frame_timestamp_ns,new_frame,alpha,mode,choice,sent_x,sent_y,sent_z,sent_w,now_x,now_y,now_z,now_w"]
            remaining = count
            pyroLog("Display capture: capturing \(count) displays")
        }
    }

    // Render thread, once per display, after its render passes and before its present.
    func capture(commandBuffer: MTLCommandBuffer, drawable: LayerRenderer.Drawable, vsync: Double,
                 frameTimestamp: UInt64, newFrame: Bool, alpha: Float, mode: Int32, choice: Bool,
                 field: MTLTexture?, selection: MTLTexture?, headNow: simd_quatf?) {
        let index: Int? = lock.withLock {
            startIfNeeded()
            guard remaining > 0 else { return nil }
            remaining -= 1
            outstanding += 1
            return buffers.count
        }
        guard let index else { return }
        let texture = drawable.colorTextures[0]
        let bytesPerPixel = texture.pixelFormat == .rgba16Float ? 8 : 4
        var cx = texture.width / 2, cy = texture.height / 2
        if let vrr = drawable.rasterizationRateMaps.first {
            let screen = vrr.screenSize
            let p = vrr.physicalCoordinates(
                screenCoordinates: MTLCoordinate2D(x: Float(Double(screen.width) * Self.center.x), y: Float(Double(screen.height) * Self.center.y)),
                layer: 0)
            cx = Int(p.x)
            cy = Int(p.y)
        }
        let w = min(Self.side, texture.width), h = min(Self.side, texture.height)
        let x = min(max(cx - w / 2, 0), texture.width - w), y = min(max(cy - h / 2, 0), texture.height - h)
        let buffer = commandBuffer.device.makeBuffer(length: w * h * bytesPerPixel, options: .storageModeShared)
        if let buffer, let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "Display capture"
            blit.copy(from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: x, y: y, z: 0),
                      sourceSize: MTLSize(width: w, height: h, depth: 1), to: buffer, destinationOffset: 0,
                      destinationBytesPerRow: w * bytesPerPixel, destinationBytesPerImage: w * h * bytesPerPixel)
            blit.endEncoding()
        }
        let fieldBuffer = copyTexture(field, bytesPerPixel: 8, commandBuffer: commandBuffer)
        let choiceBuffer = copyTexture(selection, bytesPerPixel: 4, commandBuffer: commandBuffer)
        lock.withLock {
            if let field, field.width > 1 { fieldSize = (field.width, field.height) }
            if let selection, selection.width > 1 { choiceSize = (selection.width, selection.height) }
            fieldBuffers.append(fieldBuffer)
            choiceBuffers.append(choiceBuffer)
            piece = (x, y, w, h, bytesPerPixel, bytesPerPixel == 8 ? "rgbaf16le" : "bgra")
            buffers.append(buffer)
            // The pose the streamer rendered the frame with, and the head at this display.
            let sent = FramePoseHistory.shared.orientation(timestampNs: frameTimestamp) ?? simd_quatf(ix: 0, iy: 0, iz: 0, r: 0)
            let now = headNow ?? simd_quatf(ix: 0, iy: 0, iz: 0, r: 0)
            lines.append(String(format: "%d,%.6f,%llu,%d,%.3f,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f", index, vsync, frameTimestamp,
                                newFrame ? 1 : 0, alpha, mode, choice ? 1 : 0,
                                sent.imag.x, sent.imag.y, sent.imag.z, sent.real, now.imag.x, now.imag.y, now.imag.z, now.real))
        }
        commandBuffer.addCompletedHandler { [self] _ in
            let done: Bool = lock.withLock {
                outstanding -= 1
                return outstanding == 0 && remaining == 0
            }
            if done {
                queue.async { [self] in write() }
            }
        }
    }

    private func copyTexture(_ texture: MTLTexture?, bytesPerPixel: Int, commandBuffer: MTLCommandBuffer) -> MTLBuffer? {
        guard let texture, texture.width > 1,
              let buffer = commandBuffer.device.makeBuffer(length: texture.width * texture.height * bytesPerPixel, options: .storageModeShared),
              let blit = commandBuffer.makeBlitCommandEncoder() else { return nil }
        blit.label = "Display capture, motion"
        blit.copy(from: texture, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: texture.width, height: texture.height, depth: 1), to: buffer, destinationOffset: 0,
                  destinationBytesPerRow: texture.width * bytesPerPixel, destinationBytesPerImage: texture.width * texture.height * bytesPerPixel)
        blit.endEncoding()
        return buffer
    }

    // Back to back, zeros for a display without one.
    private func writeAll(_ buffers: [MTLBuffer?], bytes: Int, name: String) {
        let url = Self.documents.appendingPathComponent(name)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        guard bytes > 0, let handle = try? FileHandle(forWritingTo: url) else { return }
        let zeros = Data(count: bytes)
        for buffer in buffers {
            if let buffer, buffer.length == bytes {
                handle.write(Data(bytesNoCopy: buffer.contents(), count: buffer.length, deallocator: .none))
            } else {
                handle.write(zeros)
            }
        }
        try? handle.close()
    }

    // Recorder queue.
    private func write() {
        let (buffers, lines, piece) = lock.withLock { (self.buffers, self.lines, self.piece) }
        let (fields, choices, fieldSize, choiceSize) = lock.withLock { (fieldBuffers, choiceBuffers, self.fieldSize, self.choiceSize) }
        writeAll(fields, bytes: fieldSize.w * fieldSize.h * 8, name: "display_field.bin")
        writeAll(choices, bytes: choiceSize.w * choiceSize.h * 4, name: "display_choice.bin")
        let binURL = Self.documents.appendingPathComponent("display_capture.bin")
        let csvURL = Self.documents.appendingPathComponent("display_capture.csv")
        FileManager.default.createFile(atPath: binURL.path, contents: nil)
        guard let handle = try? FileHandle(forWritingTo: binURL) else {
            pyroLog("Display capture: cannot write \(binURL.path)")
            return
        }
        for buffer in buffers {
            if let buffer {
                handle.write(Data(bytesNoCopy: buffer.contents(), count: buffer.length, deallocator: .none))
            }
        }
        try? handle.close()
        let header = "# \(piece.w)x\(piece.h) at physical \(piece.x),\(piece.y) of the left eye, pix_fmt \(piece.format) (linear);"
            + " field \(fieldSize.w)x\(fieldSize.h) rgba16f, choice \(choiceSize.w)x\(choiceSize.h) rg16f"
        try? ([header] + lines).joined(separator: "\n").write(to: csvURL, atomically: true, encoding: .utf8)
        lock.withLock { self.buffers = []; fieldBuffers = []; choiceBuffers = [] }
        pyroLog("Display capture: written, \(buffers.count) displays of \(piece.w)x\(piece.h)")
    }
}

// When the streamer's vectors arrive against their frame (2026-10-15, for interpolating before the
// frame is in): per frame the first motion packet, the field ready on the GPU and the frame
// decoded; every 5 s how far the field was ready ahead of the decoded frame (negative: after it).
final class MotionTimingDiag {
    static let shared = MotionTimingDiag()

    enum Event { case motionBegan, fieldReady, frameDecoded }

    private struct Times { var began = 0.0, ready = 0.0, decoded = 0.0, recorded = false }
    private let lock = NSLock()
    private var frames: [UInt64: Times] = [:]
    private var readyLead: [Double] = []
    private var beganLead: [Double] = []
    private var withoutVectors = 0
    private var windowStart = 0.0
    private var drops: [String: Int] = [:]

    // A field of the streamer's vectors that was not made, and why.
    func dropped(_ reason: String) {
        lock.lock()
        drops[reason, default: 0] += 1
        lock.unlock()
    }

    func note(_ event: Event, timestamp: UInt64) {
        let now = CACurrentMediaTime()
        var line: String?
        lock.lock()
        var t = frames[timestamp] ?? Times()
        switch event {
        case .motionBegan: if t.began == 0 { t.began = now }
        case .fieldReady: if t.ready == 0 { t.ready = now }
        case .frameDecoded: if t.decoded == 0 { t.decoded = now }
        }
        if !t.recorded, t.ready > 0, t.decoded > 0 {
            t.recorded = true
            readyLead.append((t.decoded - t.ready) * 1000)
            if t.began > 0 { beganLead.append((t.decoded - t.began) * 1000) }
        }
        frames[timestamp] = t
        // Frames decoded half a second ago without a field never got vectors in time.
        frames = frames.filter { _, f in
            let newest = max(f.began, f.ready, f.decoded)
            if now - newest < 0.5 { return true }
            if f.decoded > 0 && f.ready == 0 { withoutVectors += 1 }
            return false
        }
        if windowStart == 0 { windowStart = now }
        if now - windowStart >= 5, !readyLead.isEmpty {
            func p(_ v: [Double], _ q: Double) -> Double { v.sorted()[min(v.count - 1, Int(Double(v.count) * q))] }
            let after = readyLead.filter { $0 < 0 }.count
            line = String(format: "Streamer vectors vs frame: %d frames, field ready ahead of the decoded frame p10 %.2f p50 %.2f p90 %.2f ms, after it %d times; first motion packet ahead p50 %.2f ms; %d decoded frames without vectors",
                          readyLead.count, p(readyLead, 0.1), p(readyLead, 0.5), p(readyLead, 0.9), after,
                          beganLead.isEmpty ? 0 : p(beganLead, 0.5), withoutVectors)
            if !drops.isEmpty {
                line! += "; fields not made: " + drops.sorted { $0.value > $1.value }.prefix(6)
                    .map { "\($0.key) \($0.value)x" }.joined(separator: ", ")
                drops.removeAll()
            }
            readyLead.removeAll(keepingCapacity: true)
            beganLead.removeAll(keepingCapacity: true)
            withoutVectors = 0
            windowStart = now
        }
        lock.unlock()
        if let line { pyroLog(line) }
    }
}
