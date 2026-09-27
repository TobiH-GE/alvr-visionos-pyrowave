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
