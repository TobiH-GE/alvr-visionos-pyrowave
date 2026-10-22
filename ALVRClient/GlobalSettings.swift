//
//  GlobalSettings.swift
//
// Client-side settings and defaults
//

import Foundation
import SwiftUI

struct GlobalSettings: Codable {
    var keepSteamVRCenter: Bool = false
    var showHandsOverlaid: Bool = false
    var disablePersistentSystemOverlays: Bool = true
    var enableDoubleTapForHands: Bool = false
    var streamFPS: String = "Default"
    var realityKitRenderer: Bool = false
    var chromaKeyEnabled: Bool = false
    var chromaKeyDistRangeMin: Float = 0.35
    var chromaKeyDistRangeMax: Float = 0.7
    var chromaKeyColorR: Float = 16.0 / 255.0
    var chromaKeyColorG: Float = 124.0 / 255.0
    var chromaKeyColorB: Float = 16.0 / 255.0
    var dismissWindowOnEnter: Bool = true
    var emulatedPinchInteractions: Bool = false
    var fovRenderScale: Float = 1.0
    var forceMipmapEyeTracking = false
    var targetHandsAtRoundtripLatency = false
    var enablePersonaFaceTracking = false
    var showFaceTrackingDebug = false
    var enableProgressive = false
    var lastUsedAppVersion = "never launched"
    var chaperoneDistanceCm: Int = 0
    var showPerformanceHud: Bool = false
    // Frame queue policy. true (default) = always render the newest decoded frame and drop older
    // ones. false = the original two-frame queue, oldest frame first: it smooths arrival jitter but
    // shows a frame up to one display frame late whenever two are queued. Measured live with JPEG XS
    // (2026-09-13): decoder_queue 12.3 -> 1.1 ms, total latency 61.6 -> 50.5 ms p50, picture fine.
    var singleFrameBuffer: Bool = true
    // Metal renderer: pick the video frame just before the rendering deadline instead of at
    // visionOS's optimalInputTime, so a fresher frame is shown. Self-adjusting safety margin, see
    // LateFramePickup in Renderer.swift. A/B 2026-09-13: total latency 61.6 -> 58.5 ms p50,
    // Client System 22.5 -> 20.7 ms, at most 1 missed deadline per 450 frames, no visible judder.
    var lateFramePickup: Bool = true
    // Metal renderer: send the head pose at a controlled point of the display cycle instead of right
    // after the frame pickup, steered by the measured frame buffering. See TrackingSendPhase in
    // Renderer.swift. On by default since runs 7-13 (2026-09-27): total 40-42 ms at light load,
    // 47-51 ms under game load, against 46.6/57.9 ms p50/p95 without it (run 6).
    var trackingSendPhase: Bool = true
    // Metal renderer, visionOS 26+: render at the maximum render quality (1.0) instead of the system
    // default. The drawable then has 6262x5020 per eye instead of 4338x3478 (logical size), so a
    // stream with a sharper center is not scaled down again. Applied when the immersive space
    // opens. See RateMapDiag in Renderer.swift for the resulting center pixels per degree.
    var maxRenderQuality: Bool = false
    // Metal renderer: how the video frame is resampled into the drawable (foveation undone,
    // reprojection). "Bilinear" (default), "Bicubic" (Catmull-Rom, nine reads per plane, crisper
    // edges) or "FSR" (AMD FSR 1 EASU on the luma: edge-adaptive, interpolates along edges instead
    // of across them, 12 reads; chroma bicubic). The stream center is upscaled ~1.5x into the
    // drawable (stream/drawable 0.67 at 4320x3456), which is what FSR's quality mode is built for.
    var videoFilter: String = "Bilinear"
    // Metal renderer: contrast adaptive sharpening (after AMD's CAS) of the video's luma, to
    // offset the softening of low bits per pixel. sharpenStrength 0...1; too much amplifies codec
    // noise and ringing and can shimmer with head motion.
    var sharpenEnabled: Bool = false
    var sharpenStrength: Float = 0.5
    // PyroWave, Metal renderer: when the stream runs slower than the display (45 fps at 90 Hz, 50 fps
    // at 100 Hz), show an in-between frame on every second display, built once the next frame is
    // there from the motion between the two (head rotation is still reprojected to the current
    // pose). See FrameExtrapolation.swift.
    var frameInterpolation: Bool = false
    // Frame interpolation: use the motion vectors the streamer sends (its setting "Send motion
    // vectors", NVIDIA Optical Flow on the original frames). Off: the motion is estimated here, on
    // the GPU, from PyroWave's lowpass bands.
    var useServerMotion: Bool = true
    // Frame interpolation: build the in-between from both frames, keep still areas as decoded and
    // fade to a plain mix where the motion does not fit. Off: only the newer frame is moved, which
    // costs far less GPU time at high stream resolutions.
    var frameInterpolationBlend: Bool = false
    // Frame interpolation: choose per 8x8 pixels between the motion field, head rotation only and no
    // motion (HUD), by how well the two frames agree on their PyroWave lowpass level 0.
    var frameInterpolationSelect: Bool = false
    // How many motions the choice compares (3: the field, head rotation only, none; 4-7 add the
    // field of the neighbour cells) and on how many points per block (2: 2x2, 3: 3x3).
    // 3 and 2x2 cost about a fifth of 7 and 3x3 and looked the same on the recordings (2026-10-09).
    var frameInterpolationSelectCandidates: Float = 3
    var frameInterpolationSelectGrid: Float = 2
    // Frame interpolation: take the per-block motion choice the streamer makes on the original frames
    // (its setting "Send motion choice") instead of choosing on the headset's GPU. For frames it has
    // not come for, the choice here applies if on.
    var frameInterpolationStreamerChoice: Bool = true
    // Frame interpolation without Blend Both Frames: where the newer frame does not show what the
    // in-between does (the background an object moved over, torn at its leading edge), the previous
    // frame there.
    var frameInterpolationFillCovered: Bool = true
    // Frame interpolation: show each new frame at once (no display period of extra latency) and
    // its repeat as the frame moved forward along the next frame's motion, which the streamer's
    // vectors bring ahead of that frame (2026-10-15: 5-7 ms ahead with HEVC). Without them in time,
    // the repeat is extrapolated along the frame's own motion as before.
    var frameInterpolationPredict: Bool = false
    // Filters the head's velocities before the client core predicts the pose with them (up to 90 ms
    // ahead): the raw ones, differences of two poses ~11 ms apart, made the predicted pose jitter
    // 0.03-0.08 deg per frame with the head still, 3-5 times the head's own (2026-10-12).
    var smoothHeadPrediction: Bool = true

    init() {}
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        self.keepSteamVRCenter = try container.decodeIfPresent(Bool.self, forKey: .keepSteamVRCenter) ?? self.keepSteamVRCenter
        self.showHandsOverlaid = try container.decodeIfPresent(Bool.self, forKey: .showHandsOverlaid) ?? self.showHandsOverlaid
        self.disablePersistentSystemOverlays = try container.decodeIfPresent(Bool.self, forKey: .disablePersistentSystemOverlays) ?? self.disablePersistentSystemOverlays
        self.enableDoubleTapForHands = try container.decodeIfPresent(Bool.self, forKey: .enableDoubleTapForHands) ?? self.enableDoubleTapForHands
        self.streamFPS = try container.decodeIfPresent(String.self, forKey: .streamFPS) ?? self.streamFPS
        self.realityKitRenderer = try container.decodeIfPresent(Bool.self, forKey: .realityKitRenderer) ?? self.realityKitRenderer
        self.chromaKeyEnabled = try container.decodeIfPresent(Bool.self, forKey: .chromaKeyEnabled) ?? self.chromaKeyEnabled
        self.chromaKeyDistRangeMin = try container.decodeIfPresent(Float.self, forKey: .chromaKeyDistRangeMin) ?? self.chromaKeyDistRangeMin
        self.chromaKeyDistRangeMax = try container.decodeIfPresent(Float.self, forKey: .chromaKeyDistRangeMax) ?? self.chromaKeyDistRangeMax
        self.chromaKeyColorR = try container.decodeIfPresent(Float.self, forKey: .chromaKeyColorR) ?? self.chromaKeyColorR
        self.chromaKeyColorG = try container.decodeIfPresent(Float.self, forKey: .chromaKeyColorG) ?? self.chromaKeyColorG
        self.chromaKeyColorB = try container.decodeIfPresent(Float.self, forKey: .chromaKeyColorB) ?? self.chromaKeyColorB
        self.dismissWindowOnEnter = try container.decodeIfPresent(Bool.self, forKey: .dismissWindowOnEnter) ?? self.dismissWindowOnEnter
        self.emulatedPinchInteractions = try container.decodeIfPresent(Bool.self, forKey: .emulatedPinchInteractions) ?? self.emulatedPinchInteractions
        self.fovRenderScale = try container.decodeIfPresent(Float.self, forKey: .fovRenderScale) ?? self.fovRenderScale
        self.forceMipmapEyeTracking = try container.decodeIfPresent(Bool.self, forKey: .forceMipmapEyeTracking) ?? self.forceMipmapEyeTracking
        self.targetHandsAtRoundtripLatency = try container.decodeIfPresent(Bool.self, forKey: .targetHandsAtRoundtripLatency) ?? self.targetHandsAtRoundtripLatency
        self.enablePersonaFaceTracking = try container.decodeIfPresent(Bool.self, forKey: .enablePersonaFaceTracking) ?? self.enablePersonaFaceTracking
        self.showFaceTrackingDebug = try container.decodeIfPresent(Bool.self, forKey: .showFaceTrackingDebug) ?? self.showFaceTrackingDebug
        self.enableProgressive = try container.decodeIfPresent(Bool.self, forKey: .enableProgressive) ?? self.enableProgressive
        self.lastUsedAppVersion = try container.decodeIfPresent(String.self, forKey: .lastUsedAppVersion) ?? self.lastUsedAppVersion
        self.chaperoneDistanceCm = try container.decodeIfPresent(Int.self, forKey: .chaperoneDistanceCm) ?? self.chaperoneDistanceCm
        self.showPerformanceHud = try container.decodeIfPresent(Bool.self, forKey: .showPerformanceHud) ?? self.showPerformanceHud
        self.singleFrameBuffer = try container.decodeIfPresent(Bool.self, forKey: .singleFrameBuffer) ?? self.singleFrameBuffer
        self.lateFramePickup = try container.decodeIfPresent(Bool.self, forKey: .lateFramePickup) ?? self.lateFramePickup
        self.trackingSendPhase = try container.decodeIfPresent(Bool.self, forKey: .trackingSendPhase) ?? self.trackingSendPhase
        self.maxRenderQuality = try container.decodeIfPresent(Bool.self, forKey: .maxRenderQuality) ?? self.maxRenderQuality
        self.videoFilter = try container.decodeIfPresent(String.self, forKey: .videoFilter) ?? self.videoFilter
        self.sharpenEnabled = try container.decodeIfPresent(Bool.self, forKey: .sharpenEnabled) ?? self.sharpenEnabled
        self.sharpenStrength = try container.decodeIfPresent(Float.self, forKey: .sharpenStrength) ?? self.sharpenStrength
        self.frameInterpolation = try container.decodeIfPresent(Bool.self, forKey: .frameInterpolation) ?? self.frameInterpolation
        self.useServerMotion = try container.decodeIfPresent(Bool.self, forKey: .useServerMotion) ?? self.useServerMotion
        self.frameInterpolationBlend = try container.decodeIfPresent(Bool.self, forKey: .frameInterpolationBlend) ?? self.frameInterpolationBlend
        self.frameInterpolationSelect = try container.decodeIfPresent(Bool.self, forKey: .frameInterpolationSelect) ?? self.frameInterpolationSelect
        self.frameInterpolationSelectCandidates = try container.decodeIfPresent(Float.self, forKey: .frameInterpolationSelectCandidates) ?? self.frameInterpolationSelectCandidates
        self.frameInterpolationSelectGrid = try container.decodeIfPresent(Float.self, forKey: .frameInterpolationSelectGrid) ?? self.frameInterpolationSelectGrid
        self.frameInterpolationStreamerChoice = try container.decodeIfPresent(Bool.self, forKey: .frameInterpolationStreamerChoice) ?? self.frameInterpolationStreamerChoice
        self.frameInterpolationFillCovered = try container.decodeIfPresent(Bool.self, forKey: .frameInterpolationFillCovered) ?? self.frameInterpolationFillCovered
        self.smoothHeadPrediction = try container.decodeIfPresent(Bool.self, forKey: .smoothHeadPrediction) ?? self.smoothHeadPrediction
        self.frameInterpolationPredict = try container.decodeIfPresent(Bool.self, forKey: .frameInterpolationPredict) ?? self.frameInterpolationPredict
    }
}

extension GlobalSettingsStore {
    static let sampleData: GlobalSettingsStore =
    GlobalSettingsStore()
}

class GlobalSettingsStore: ObservableObject {
    @Published var settings: GlobalSettings = GlobalSettings()
    
    init() {
        try? load()
    }

    private static func fileURL() throws -> URL {
        try FileManager.default.url(for: .documentDirectory,
                                    in: .userDomainMask,
                                    appropriateFor: nil,
                                    create: true)
        .appendingPathComponent("globalsettings.data")
    }
    
    func load() throws {
        let fileURL = try Self.fileURL()
        guard let data = try? Data(contentsOf: fileURL) else {
            return self.settings = GlobalSettings()
        }
        let globalSettings = try JSONDecoder().decode(GlobalSettings.self, from: data)
        self.settings = globalSettings
    }
    
    func save(settings: GlobalSettings) throws {
        let data = try JSONEncoder().encode(settings)
        let outfile = try Self.fileURL()
        try data.write(to: outfile)
    }
}
