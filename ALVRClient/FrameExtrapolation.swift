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
//     receiveServerMotion), or, with "Use Server Motion Vectors" off, is estimated here on a queue
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

import Foundation
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
    private var lastDisplayVsync = 0.0
    private var firstShown: [UInt64: Double] = [:]
    // Smoothed time from a frame's timestamp to its first display (seconds), for interpolation.
    private var contentLag = 0.0
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
                    if contentLag == 0 {
                        contentLag = lag
                    } else {
                        contentLag += 0.1 * min(max(lag - contentLag, -0.02), 0.02)
                    }
                }
                if firstShown.count > 16, let oldest = firstShown.keys.min() {
                    firstShown.removeValue(forKey: oldest)
                }
                shownAt = vsync
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
            let contentTime = vsync - contentLag - interval + displayPeriod
            let alpha = max(-1, min(Self.maxLateAlpha, (contentTime - Double(timestamp) / 1e9) / interval))
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
        selectBuffer.commit()
        selectionFor = (timestamp, previousTimestamp, texture)
        return texture
    }

    // Interpolation needs the motion of a new frame before its first display. True while that
    // motion is still being estimated and the frame has waited less than a display period: the
    // renderer then shows the current frame once more instead (counted as held back).
    func shouldHoldBack(timestamp: UInt64, queuedTime: Double) -> Bool {
        let now = CACurrentMediaTime()
        return lock.withLock {
            guard let field = fields.first(where: { $0.timestamp == timestamp }), !field.ready else {
                return false
            }
            return queuedTime > 0 && now - queuedTime < displayPeriod
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
        guard u32(h) == 0x564d_5750, u32(h + 4) == 1 else { return }
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
        guard firstRow + rows <= gridHeight, packet.count >= prefixBytes + headerBytes + rows * gridWidth * 4,
              reference < timestamp else { return }
        let base = (previousFrame > 0 && previousFrame < reference) ? previousFrame : reference
        let scale = Float(Double(timestamp - base) / Double(timestamp - reference))

        if serverTimestamp != timestamp {
            // A new frame: the previous one's rows are complete or given up on.
            serverTimestamp = timestamp
            serverRowsReceived = [Bool](repeating: false, count: gridHeight)
            serverRowCount = 0
            serverUploadCurrent = serverUploadNext
            serverUploadNext = (serverUploadNext + 1) % serverUploads.count
            let target = serverUploadBuffers[serverUploadCurrent]
            memset(target.contents(), 0, target.length)
            let taken = lock.withLock { fields.contains(where: { $0.timestamp == timestamp }) }
            serverField = taken ? nil : reserveField(timestamp: timestamp, previousTimestamp: base, fromServer: true)
        }
        guard serverField != nil else { return } // the frame's own estimate came first
        let dataStart = prefixBytes + headerBytes
        let flow = serverUploadBuffers[serverUploadCurrent].contents().assumingMemoryBound(to: Float.self)
        for r in 0..<rows where !serverRowsReceived[firstRow + r] {
            serverRowsReceived[firstRow + r] = true
            serverRowCount += 1
            let row = flow + (firstRow + r) * serverUploadRowFloats
            for x in 0..<gridWidth {
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
        guard UInt32(littleEndian: packet.loadUnaligned(fromByteOffset: h + 4, as: UInt32.self)) == 1,
              let texture = choiceTextures.first else { return }
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
        guard firstRow + rows <= gridHeight, packet.count >= prefixBytes + headerBytes + rows * rowBytes else { return }
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
        for r in 0..<rows where !choiceRowsReceived[firstRow + r] {
            choiceRowsReceived[firstRow + r] = true
            choiceRowCount += 1
            let row = out + (firstRow + r) * choiceRowHalves
            let source = dataStart + r * rowBytes
            for x in 0..<gridWidth {
                let byte = packet[source + x / 4]
                row[2 * x] = values[Int((byte >> UInt8(2 * (x % 4))) & 3)]
                row[2 * x + 1] = 0
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

    private func finishServerField(_ field: Field, timestamp: UInt64, reference: UInt64) {
        let (geometry, displayPeriod) = lock.withLock { (self.geometry, self.displayPeriod) }
        let foveation = Self.foveationLock.withLock { Self.foveation }
        let interval = Double(timestamp - reference) / 1e9
        guard let geometry, geometry.tangents.count >= 2, interval > Self.slowStreamFactor * displayPeriod, interval < 0.1 else {
            // Not a slow stream (or no geometry yet): nothing to show in between.
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
        }
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
