/*
Abstract:
Controls that allow entry into the ALVR environment.
*/

import SwiftUI

/// Controls that allow entry into the ALVR environment.
struct EntryControls: View {
    @Environment(ViewModel.self) private var model
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    @Environment(\.dismissWindow) private var dismissWindow
    @ObservedObject var eventHandler = EventHandler.shared
    @EnvironmentObject var gStore: GlobalSettingsStore
    
    @State private var showImmersiveSpace = false
    @State private var immersiveSpaceIsShown = false
    
    let saveAction: ()->Void

    var body: some View {
        @Bindable var model = model
        
        HStack(spacing: 17) {
            if eventHandler.connectionState == .connected {
                // Search the streamer again over the USB Ethernet adapter only, or over Wi-Fi only.
                Button {
                    eventHandler.rescan(.wired)
                } label: {
                    HStack(spacing: 8) {
                        Text("Rescan")
                        Image(systemName: "cable.connector.horizontal")
                    }
                    .padding(15)
                }
                .help("Search the streamer over the USB Ethernet adapter only")
                .disabled(model.isShowingClient || eventHandler.rescanning)

                Button {
                    eventHandler.rescan(.wifi)
                } label: {
                    HStack(spacing: 8) {
                        Text("Rescan")
                        Image(systemName: "wifi")
                    }
                    .padding(15)
                }
                .help("Search the streamer over Wi-Fi only")
                .disabled(model.isShowingClient || eventHandler.rescanning)

                Toggle(isOn: $model.isShowingClient) {
                    Label("Enter", systemImage: "visionpro")
                        .labelStyle(.titleAndIcon)
                        .padding(15)
                }
            } else {
                Label("Connecting...", systemImage: "visionpro")
                    .labelStyle(.titleOnly)
                    .padding(15)
            }

            if eventHandler.displayAt100Hz {
                Label("100 Hz", systemImage: "exclamationmark.triangle.fill")
                    .labelStyle(.titleAndIcon)
                    .foregroundStyle(.red)
                    .padding(15)
                    .help("visionOS runs the display at 100 Hz: it detected 50 Hz flicker from artificial light. A 45 or 90 fps stream will stutter. Switch off or replace the flickering light.")
            }
        }
        .onAppear {
            DisplayRateProbe.shared.start()
        }
        .toggleStyle(.button)
        .buttonStyle(.borderless)
        .glassBackgroundEffect(in: .rect(cornerRadius: 50))

        //Enable Client
        .onChange(of: model.isShowingClient) { _, isShowing in
            Task {
                if isShowing {

                    saveAction()
                    print("Opening Immersive Space")
                    if gStore.settings.realityKitRenderer {
                        if !DummyMetalRenderer.haveRenderInfo {
                            var dummySpaceIsOpened = false
                            while !dummySpaceIsOpened {
                                switch await openImmersiveSpace(id: "DummyImmersiveSpace") {
                                case .opened:
                                    dummySpaceIsOpened = true
                                case .error, .userCancelled:
                                    fallthrough
                                @unknown default:
                                    dummySpaceIsOpened = false
                                }
                            }
                            
                            while dummySpaceIsOpened && !DummyMetalRenderer.haveRenderInfo {
                                try! await Task.sleep(nanoseconds: 1_000_000)
                            }
                            
                            await dismissImmersiveSpace()
                            try! await Task.sleep(nanoseconds: 1_000_000_000)
                        }
                        
                        if !DummyMetalRenderer.haveRenderInfo {
                            print("MISSING VIEW INFO!!")
                        }
                        
                        WorldTracker.shared.worldTrackingAddedOriginAnchor = false
                        
                        print("Open real immersive space")
                        
                        switch await openImmersiveSpace(id: "RealityKitClient") {
                        case .opened:
                            immersiveSpaceIsShown = true
                        case .error, .userCancelled:
                            fallthrough
                        @unknown default:
                            immersiveSpaceIsShown = false
                            showImmersiveSpace = false
                        }
                    }
                    else {
                        await openImmersiveSpace(id: "MetalClient")
                    }
                    VideoHandler.applyRefreshRate(videoFormat: EventHandler.shared.videoFormat)
                    if gStore.settings.dismissWindowOnEnter {
                        dismissWindow(id: "Entry")
                    }
                }
            }
        }

    }
}
