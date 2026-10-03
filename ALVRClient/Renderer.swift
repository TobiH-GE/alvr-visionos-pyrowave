//
//  Renderer.swift
//
// Primarily, stuff for the MetalClientSystem rendering, but portions are shared
// with RealityKitClientSystem
//
// Notable portions include:
// - Pipeline setup for different color formats and compiled Metal constants (rebuildRenderPipelines)
//
import CompositorServices
import Metal
import MetalKit
import simd
import Spatial
import ARKit
import VideoToolbox
import ObjectiveC

// The 256 byte aligned size of our uniform structure
let alignedUniformsSize = (MemoryLayout<UniformsArray>.size + 0xFF) & -0x100
let alignedPlaneUniformSize = (MemoryLayout<PlaneUniform>.size + 0xFF) & -0x100

let maxBuffersInFlight = 6
let maxPlanesDrawn = 512

enum RendererError: Error {
    case badVertexDescriptor
}

// Focal depth of the timewarp panel, ideally would be adjusted based on the depth
// of what the user is looking at.
let panel_depth: Float = 1

// TODO(zhuowei): what's the z supposed to be?
// x, y, z
// u, v
let fullscreenQuadVertices:[Float] = [-panel_depth, -panel_depth, -panel_depth,
                                       panel_depth, -panel_depth, -panel_depth,
                                       -panel_depth, panel_depth, -panel_depth,
                                       panel_depth, panel_depth, -panel_depth,
                                       0, 1,
                                       0.5, 1,
                                       0, 0,
                                       0.5, 0]

let hudQuadVertices: [Float] = [
    -0.5, -0.5, 0.0,
     0.5, -0.5, 0.0,
    -0.5,  0.5, 0.0,
     0.5,  0.5, 0.0,
     0.0,  1.0,
     1.0,  1.0,
     0.0,  0.0,
     1.0,  0.0
]
                                       
let unitVectorXYZVertices:[Float] = [0.0, 0.0, 0.0,
                                       1.0, 0.0, 0.0,
                                       0.01,0.0,0.0,
                                       
                                       0.0, 0.0, 0.0,
                                       0.0, 1.0, 0.0,
                                       0.0,0.01,0.0,
                                       
                                       0.0, 0.0, 0.0,
                                       0.0, 0.0, 1.0,
                                       0.0,0.0,0.01,
                                       
                                       0, 1,
                                       0.5, 1,
                                       0, 0,
                                       0.5, 0]

func NonlinearToLinearRGB(_ color: simd_float3) -> simd_float3 {
    let DIV12: Float = 1.0 / 12.92;
    let DIV1: Float = 1.0 / 1.055;
    let THRESHOLD: Float = 0.04045;
    let GAMMA = simd_float3(repeating: 2.4);
        
    let condition = simd_float3(color.x < THRESHOLD ? 1.0 : 0.0, color.y < THRESHOLD ? 1.0 : 0.0, color.z < THRESHOLD ? 1.0 : 0.0);
    let lowValues = color * DIV12;
    let highValues = pow((color + 0.055) * DIV1, GAMMA);
    return condition * lowValues + (1.0 - condition) * highValues;
}

class Renderer {

    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    var pipelineState: MTLRenderPipelineState
    var depthStateAlways: MTLDepthStencilState
    var depthStateGreater: MTLDepthStencilState

    var dynamicUniformBuffer: MTLBuffer
    var uniformBufferOffset = 0
    var uniformBufferIndex = 0
    var uniforms: UnsafeMutablePointer<UniformsArray>
    
    var dynamicPlaneUniformBuffer: MTLBuffer
    var planeUniformBufferOffset = 0
    var planeUniformBufferIndex = 0
    var planeUniforms: UnsafeMutablePointer<PlaneUniform>

    let layerRenderer: LayerRenderer?
    var metalTextureCache: CVMetalTextureCache!
    let mtlVertexDescriptor: MTLVertexDescriptor
    let mtlVertexDescriptorNoUV: MTLVertexDescriptor
    var videoFramePipelineState_YpCbCrBiPlanar: MTLRenderPipelineState!
    var videoFramePipelineState_SecretYpCbCrFormats: MTLRenderPipelineState!
    var videoFrameDepthPipelineState: MTLRenderPipelineState!
    var fullscreenQuadBuffer:MTLBuffer!
    var unitVectorXYZBuffer:MTLBuffer!
    /// Steam Controller body wireframe. Built on first use rather than at init,
    /// because the overwhelmingly common case is that no Steam Controller is
    /// connected and nothing ever asks to draw it.
    private var ibexMeshBufferStorage:MTLBuffer?
    private var ibexMeshBuildAttempted = false
    var ibexMeshVertexCount: Int = 0

    /// Uploads `IbexModel`'s decimated shell on first call. Returns nil if the
    /// resource is missing, and does not retry — a missing bundle resource will
    /// not fix itself, and retrying every frame would spam the log.
    func ibexMeshBuffer() -> MTLBuffer? {
        if ibexMeshBuildAttempted { return ibexMeshBufferStorage }
        ibexMeshBuildAttempted = true
        let verts = IbexModel.wireframeVertices
        guard !verts.isEmpty else { return nil }
        verts.withUnsafeBytes {
            ibexMeshBufferStorage = device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)
        }
        ibexMeshVertexCount = IbexModel.wireframeVertexCount
        print("Renderer: Ibex wireframe uploaded, \(ibexMeshVertexCount) verts")
        return ibexMeshBufferStorage
    }
    var hudPipelineState: MTLRenderPipelineState!
    var hudQuadBuffer: MTLBuffer!
    var depthStateAlwaysNoWrite: MTLDepthStencilState!
    var performanceHudRenderer: PerformanceHudRenderer?
    var encodingGamma: Float = 1.0
    var lastReconfigureTime: Double = 0.0
    
    var drawPlanesWithInformedColors: Bool = false
    var fadeInOverlayAlpha: Float = 0.0
    var coolPulsingColorsTime: Float = 0.0
    private let chaperoneSystem = ChaperoneSystem()
#if CHAPERONE_PROFILE
    private var chaperoneProfileAccumMs: Double = 0.0
    private var chaperoneProfileSamples: Int = 0
    private var chaperoneProfileLastLog: Double = 0.0
    private var chaperoneProfileWindow: [Double] = []
    private let chaperoneProfileWindowSize: Int = 120
#endif
    var reprojectedFramesInARow: Int = 0
    var roundTripRenderTime: Double = 0.0
    var lastRoundTripRenderTimestamp: Double = 0.0
    var currentYuvTransform: simd_float4x4 = matrix_identity_float4x4
    
    // Was curious if it improved; it's still juddery.
    var useApplesReprojection = false
    
    // More readable helper var than layerRenderer == nil
    var isRealityKit = false
    var hdrEnabled = false
    var currentRenderColorFormat = renderColorFormatSDR
    var currentDrawableRenderColorFormat = renderColorFormatDrawableSDR
    
    //
    // Chroma keying shader vars
    //
    var chromaKeyEnabled = false
    // Video resampling filters as the current pipelines were built (GlobalSettings.videoFilter:
    // 0 bilinear, 1 bicubic, 2 FSR;
    // sharpenEnabled/sharpenStrength; 0 = no sharpening).
    var videoFilter: Int32 = 0
    var videoSharpen: Float = 0.0
    var chromaKeyColor = simd_float3(0.0, 1.0, 0.0); // green
    
    //chromaKeyLerpDistRange is used to decide the amount of color to be used from either foreground or background
    //if the current distance from pixel color to chromaKey is smaller then chromaKeyLerpDistRange.x we use background,
    //if the current distance from pixel color to chromaKey is bigger then chromaKeyLerpDistRange.y we use foreground,
    //else, we alpha blend them
    //playing with this variable will decide how much the foreground and background blend together
    var chromaKeyLerpDistRange = simd_float2(0.005, 0.1);
    
    init(_ layerRenderer: LayerRenderer?) {
        self.layerRenderer = layerRenderer
        if layerRenderer == nil {
            isRealityKit = true
        }
        else {
#if XCODE_BETA_26
            if #available(visionOS 26.0, *) {
                // maxRenderQuality is 1.0 (ALVRClientApp), so any value up to 1.0 is allowed.
                if ALVRClientApp.gStore.settings.maxRenderQuality {
                    self.layerRenderer?.renderQuality = .init(1.0)
                    pyroLog("Render quality: 1.0 (Max Render Quality)")
                }
            }
#endif
        }

        encodingGamma = EventHandler.shared.encodingGamma
        hdrEnabled = EventHandler.shared.enableHdr
        if hdrEnabled {
            currentRenderColorFormat = renderColorFormatHDR
            currentDrawableRenderColorFormat = renderColorFormatDrawableHDR
        }
        else {
            currentRenderColorFormat = renderColorFormatSDR
            currentDrawableRenderColorFormat = renderColorFormatSDR
        }
        
        self.device = layerRenderer?.device ?? MTLCreateSystemDefaultDevice()!
        self.commandQueue = self.device.makeCommandQueue()!

        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight
        self.dynamicUniformBuffer = self.device.makeBuffer(length:uniformBufferSize,
                                                           options:[MTLResourceOptions.storageModeShared])!
        self.dynamicUniformBuffer.label = "UniformBuffer"
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to:UniformsArray.self, capacity:1)

        let planeUniformBufferSize = alignedPlaneUniformSize * maxPlanesDrawn
        self.dynamicPlaneUniformBuffer = self.device.makeBuffer(length:planeUniformBufferSize,
                                                           options:[MTLResourceOptions.storageModeShared])!
        self.dynamicPlaneUniformBuffer.label = "PlaneUniformBuffer"
        planeUniforms = UnsafeMutableRawPointer(dynamicPlaneUniformBuffer.contents()).bindMemory(to:PlaneUniform.self, capacity:1)
        
        mtlVertexDescriptor = Renderer.buildMetalVertexDescriptor()
        mtlVertexDescriptorNoUV = Renderer.buildMetalVertexDescriptorNoUV()

        do {
            pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                       mtlVertexDescriptor: mtlVertexDescriptor,
                                                                       colorFormat: layerRenderer?.configuration.colorFormat ?? currentRenderColorFormat,
                                                                       depthFormat: layerRenderer?.configuration.depthFormat ?? renderDepthFormat,
                                                                       viewCount: layerRenderer?.properties.viewCount ?? renderViewCount,
                                                                       vertexShaderName: "vertexShader",
                                                                       fragmentShaderName: "fragmentShader")
        } catch {
            fatalError("Unable to compile render pipeline state.  Error info: \(error)")
        }
        do {
            hudPipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                          mtlVertexDescriptor: mtlVertexDescriptor,
                                                                          colorFormat: layerRenderer?.configuration.colorFormat ?? currentRenderColorFormat,
                                                                          depthFormat: layerRenderer?.configuration.depthFormat ?? renderDepthFormat,
                                                                          viewCount: layerRenderer?.properties.viewCount ?? renderViewCount,
                                                                          vertexShaderName: "hudVertexShader",
                                                                          fragmentShaderName: "hudFragmentShader")
        } catch {
            fatalError("Unable to compile HUD render pipeline state.  Error info: \(error)")
        }

        let depthStateDescriptorAlways = MTLDepthStencilDescriptor()
        depthStateDescriptorAlways.depthCompareFunction = MTLCompareFunction.always
        depthStateDescriptorAlways.isDepthWriteEnabled = true
        self.depthStateAlways = device.makeDepthStencilState(descriptor:depthStateDescriptorAlways)!
        let depthStateDescriptorAlwaysNoWrite = MTLDepthStencilDescriptor()
        depthStateDescriptorAlwaysNoWrite.depthCompareFunction = .always
        depthStateDescriptorAlwaysNoWrite.isDepthWriteEnabled = false
        self.depthStateAlwaysNoWrite = device.makeDepthStencilState(descriptor: depthStateDescriptorAlwaysNoWrite)!
        
        let depthStateDescriptorGreater = MTLDepthStencilDescriptor()
        depthStateDescriptorGreater.depthCompareFunction = MTLCompareFunction.greater
        depthStateDescriptorGreater.isDepthWriteEnabled = true
        self.depthStateGreater = device.makeDepthStencilState(descriptor:depthStateDescriptorGreater)!
        
        if CVMetalTextureCacheCreate(nil, nil, self.device, nil, &metalTextureCache) != 0 {
            fatalError("CVMetalTextureCacheCreate")
        }
        fullscreenQuadVertices.withUnsafeBytes {
            fullscreenQuadBuffer = device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)
        }
        hudQuadVertices.withUnsafeBytes {
            hudQuadBuffer = device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)
        }
        
        unitVectorXYZVertices.withUnsafeBytes {
            unitVectorXYZBuffer = device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)
        }
        
        self.videoFrameDepthPipelineState = try! Renderer.buildRenderPipelineForVideoFrameDepthWithDevice(
                device: self.device,
                mtlVertexDescriptor: self.mtlVertexDescriptor,
                colorFormat: layerRenderer?.configuration.colorFormat ?? currentRenderColorFormat,
                depthFormat: layerRenderer?.configuration.depthFormat ?? renderDepthFormat,
                viewCount: layerRenderer?.properties.viewCount ?? renderViewCount
        )
        
        rebuildRenderPipelines()

        EventHandler.shared.handleRenderStarted()
        EventHandler.shared.renderStarted = true
    }
    
    func rebuildRenderPipelines() {
        print("rebuildRenderPipelines")

        encodingGamma = EventHandler.shared.encodingGamma
        hdrEnabled = EventHandler.shared.enableHdr
        if hdrEnabled {
            currentRenderColorFormat = renderColorFormatHDR
            currentDrawableRenderColorFormat = renderColorFormatDrawableHDR
        }
        else {
            currentRenderColorFormat = renderColorFormatSDR
            currentDrawableRenderColorFormat = renderColorFormatSDR
        }

        // Everything below builds the *video frame* pipelines, and both of its
        // inputs only exist once a streamer has told us about the stream:
        // `settings` comes from the session config, and `streamEvent` from
        // STREAMING_STARTED. The renderer can legitimately be running before
        // either arrives — the offline input debug space brings it up with no
        // server at all — so this is a normal state, not a failure.
        //
        // Nothing is lost by deferring: there are no video frames to draw yet,
        // the pipeline states are implicitly-unwrapped optionals that stay nil
        // until used, and the first IPD report after streaming goes active
        // re-enters this function (see the `lastIpd == -1` branch in
        // renderFrame) and builds them for real.
        guard let settings = Settings.getAlvrSettings(),
              let streamEvent = EventHandler.shared.streamEvent else {
            print("rebuildRenderPipelines: no stream yet, deferring video pipelines")
            return
        }

        let foveationVars = FFR.calculateFoveationVars(alvrEvent: streamEvent.STREAMING_STARTED, foveationSettings: settings.video.foveated_encoding)
        videoFramePipelineState_YpCbCrBiPlanar = try! buildRenderPipelineForVideoFrameWithDevice(
                            device: device,
                            mtlVertexDescriptor: mtlVertexDescriptor,
                            colorFormat: layerRenderer?.configuration.colorFormat ?? currentRenderColorFormat,
                            viewCount: layerRenderer?.properties.viewCount ?? renderViewCount,
                            foveationVars: foveationVars,
                            variantName: "YpCbCrBiPlanar"
        )
        videoFramePipelineState_SecretYpCbCrFormats = try! buildRenderPipelineForVideoFrameWithDevice(
                            device: device,
                            mtlVertexDescriptor: mtlVertexDescriptor,
                            colorFormat: layerRenderer?.configuration.colorFormat ?? currentRenderColorFormat,
                            viewCount: layerRenderer?.properties.viewCount ?? renderViewCount,
                            foveationVars: foveationVars,
                            variantName: "SecretYpCbCrFormats"
        )
        // What actually went into the shaders' function constants, so a log says which filter ran.
        let filterNames = ["Bilinear", "Bicubic", "FSR"]
        let filterName = filterNames.indices.contains(Int(videoFilter)) ? filterNames[Int(videoFilter)] : "unknown"
        pyroLog("Video pipelines built: filter \(filterName) (constant \(videoFilter), setting \"\(ALVRClientApp.gStore.settings.videoFilter)\"), sharpen "
            + (videoSharpen > 0 ? String(format: "%.2f", videoSharpen) : "off"))
        
        do {
            pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                       mtlVertexDescriptor: mtlVertexDescriptor,
                                                                       colorFormat: layerRenderer?.configuration.colorFormat ?? currentRenderColorFormat,
                                                                       depthFormat: layerRenderer?.configuration.depthFormat ?? renderDepthFormat,
                                                                       viewCount: layerRenderer?.properties.viewCount ?? renderViewCount,
                                                                       vertexShaderName: "vertexShader",
                                                                       fragmentShaderName: "fragmentShader")
        } catch {
            fatalError("Unable to compile render pipeline state.  Error info: \(error)")
        }
        
        self.videoFrameDepthPipelineState = try! Renderer.buildRenderPipelineForVideoFrameDepthWithDevice(
                device: self.device,
                mtlVertexDescriptor: self.mtlVertexDescriptor,
                colorFormat: layerRenderer?.configuration.colorFormat ?? currentRenderColorFormat,
                depthFormat: layerRenderer?.configuration.depthFormat ?? renderDepthFormat,
                viewCount: layerRenderer?.properties.viewCount ?? renderViewCount
        )
    }

    // Vertex descriptor with float3 position and float2 UVs
    class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
        // Create a Metal vertex descriptor specifying how vertices will by laid out for input into our render
        //   pipeline and how we'll layout our Model IO vertices

        let mtlVertexDescriptor = MTLVertexDescriptor()

        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue

        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].format = MTLVertexFormat.float2
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.texcoord.rawValue].bufferIndex = BufferIndex.meshGenerics.rawValue

        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stride = 8
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshGenerics.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        return mtlVertexDescriptor
    }
    
    // Vertex descriptor without any UV info
    class func buildMetalVertexDescriptorNoUV() -> MTLVertexDescriptor {
        // Create a Metal vertex descriptor specifying how vertices will by laid out for input into our render
        //   pipeline and how we'll layout our Model IO vertices

        let mtlVertexDescriptor = MTLVertexDescriptor()

        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format = MTLVertexFormat.float3
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue

        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = 12
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = MTLVertexStepFunction.perVertex

        return mtlVertexDescriptor
    }

    // Generic render pipeline, used for the wireframe rendering.
    class func buildRenderPipelineWithDevice(device: MTLDevice,
                                             mtlVertexDescriptor: MTLVertexDescriptor,
                                             colorFormat: MTLPixelFormat,
                                             depthFormat: MTLPixelFormat,
                                             viewCount: Int,
                                             vertexShaderName: String,
                                             fragmentShaderName: String) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object

        let library = device.makeDefaultLibrary()

        let vertexFunction = library?.makeFunction(name: vertexShaderName)
        let fragmentFunction = library?.makeFunction(name: fragmentShaderName)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor

        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = viewCount
        
        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    // Copy/"passthrough" pipeline for transferring from an offscreen MTLTexture
    // to the final RealityKit MTLTexture.
    func buildCopyPipelineWithDevice(device: MTLDevice,
                                             colorFormat: MTLPixelFormat,
                                             viewCount: Int,
                                             vrrScreenSize: MTLSize?,
                                             vrrPhysSize: MTLSize?,
                                             vertexShaderName: String,
                                             fragmentShaderName: String) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object

        let library = device.makeDefaultLibrary()
        
        let fragmentConstants = MTLFunctionConstantValues()
        let settings = ALVRClientApp.gStore.settings
        if #available(visionOS 2.0, *) {
            chromaKeyEnabled = settings.chromaKeyEnabled
        }
        else {
            chromaKeyEnabled = settings.chromaKeyEnabled && isRealityKit
        }
        chromaKeyColor = simd_float3(settings.chromaKeyColorR, settings.chromaKeyColorG, settings.chromaKeyColorB)
        chromaKeyLerpDistRange = simd_float2(settings.chromaKeyDistRangeMin, settings.chromaKeyDistRangeMax)

        var mutVrrScreenSize = simd_float2(Float(vrrScreenSize?.width ?? 1), Float(vrrScreenSize?.height ?? 1))
        var mutVrrPhysSize = simd_float2(Float(vrrPhysSize?.width ?? 1), Float(vrrPhysSize?.height ?? 1))
        var chromaKeyColorLinear = NonlinearToLinearRGB(chromaKeyColor)
        fragmentConstants.setConstantValue(&chromaKeyEnabled, type: .bool, index: ALVRFunctionConstant.chromaKeyEnabled.rawValue)
        fragmentConstants.setConstantValue(&chromaKeyColorLinear, type: .float3, index: ALVRFunctionConstant.chromaKeyColor.rawValue)
        fragmentConstants.setConstantValue(&chromaKeyLerpDistRange, type: .float2, index: ALVRFunctionConstant.chromaKeyLerpDistRange.rawValue)
        fragmentConstants.setConstantValue(&isRealityKit, type: .bool, index: ALVRFunctionConstant.realityKitEnabled.rawValue)
        fragmentConstants.setConstantValue(&mutVrrScreenSize, type: .float2, index: ALVRFunctionConstant.vrrScreenSize.rawValue)
        fragmentConstants.setConstantValue(&mutVrrPhysSize, type: .float2, index: ALVRFunctionConstant.vrrPhysSize.rawValue)

        let vertexFunction = try! library?.makeFunction(name: vertexShaderName, constantValues: fragmentConstants)
        let fragmentFunction = try! library?.makeFunction(name: fragmentShaderName, constantValues: fragmentConstants)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptorNoUV

        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat
        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = false

        pipelineDescriptor.maxVertexAmplificationCount = viewCount
        
        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    // Depth-only renderer, for correcting after overlay render just so Apple's compositor isn't annoying about it
    class func buildRenderPipelineForVideoFrameDepthWithDevice(device: MTLDevice,
                                                          mtlVertexDescriptor: MTLVertexDescriptor,
                                                          colorFormat: MTLPixelFormat,
                                                          depthFormat: MTLPixelFormat,
                                                          viewCount: Int) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object

        let library = device.makeDefaultLibrary()

        let vertexFunction = library?.makeFunction(name: "videoFrameVertexShader")
        let fragmentFunction = library?.makeFunction(name: "videoFrameDepthFragmentShader")

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "VideoFrameDepthRenderPipeline"
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        //pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor

        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = viewCount
        
        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    static func videoFilterIndex(_ settings: GlobalSettings) -> Int32 {
        switch settings.videoFilter {
        case "Bicubic": return 1
        case "FSR": return 2
        default: return 0
        }
    }

    static func effectiveSharpen(_ settings: GlobalSettings) -> Float {
        settings.sharpenEnabled ? max(settings.sharpenStrength, 0.0) : 0.0
    }

    // Video frame renderer, incl my own YCbCr stage and/or Apple's 48 private YCbCr texture formats.
    func buildRenderPipelineForVideoFrameWithDevice(device: MTLDevice,
                                                          mtlVertexDescriptor: MTLVertexDescriptor,
                                                          colorFormat: MTLPixelFormat,
                                                          viewCount: Int,
                                                          foveationVars: FoveationVars,
                                                          variantName: String) throws -> MTLRenderPipelineState {
        

        let library = device.makeDefaultLibrary()
        let vertexFunction = library?.makeFunction(name: "videoFrameVertexShader")
        let fragmentConstants = FFR.makeFunctionConstants(foveationVars)
        
        let settings = ALVRClientApp.gStore.settings
        if #available(visionOS 2.0, *) {
            chromaKeyEnabled = settings.chromaKeyEnabled
        }
        else {
            chromaKeyEnabled = settings.chromaKeyEnabled && isRealityKit
        }
        chromaKeyColor = simd_float3(settings.chromaKeyColorR, settings.chromaKeyColorG, settings.chromaKeyColorB)
        chromaKeyLerpDistRange = simd_float2(settings.chromaKeyDistRangeMin, settings.chromaKeyDistRangeMax)

        var chromaKeyColorLinear = NonlinearToLinearRGB(chromaKeyColor)
        fragmentConstants.setConstantValue(&chromaKeyEnabled, type: .bool, index: ALVRFunctionConstant.chromaKeyEnabled.rawValue)
        fragmentConstants.setConstantValue(&chromaKeyColorLinear, type: .float3, index: ALVRFunctionConstant.chromaKeyColor.rawValue)
        fragmentConstants.setConstantValue(&chromaKeyLerpDistRange, type: .float2, index: ALVRFunctionConstant.chromaKeyLerpDistRange.rawValue)
        fragmentConstants.setConstantValue(&isRealityKit, type: .bool, index: ALVRFunctionConstant.realityKitEnabled.rawValue)
        fragmentConstants.setConstantValue(&encodingGamma, type: .float, index: ALVRFunctionConstant.encodingGamma.rawValue)
        fragmentConstants.setConstantValue(&currentYuvTransform.columns.0, type: .float4, index: ALVRFunctionConstant.encodingYUVTransform0.rawValue)
        fragmentConstants.setConstantValue(&currentYuvTransform.columns.1, type: .float4, index: ALVRFunctionConstant.encodingYUVTransform1.rawValue)
        fragmentConstants.setConstantValue(&currentYuvTransform.columns.2, type: .float4, index: ALVRFunctionConstant.encodingYUVTransform2.rawValue)
        fragmentConstants.setConstantValue(&currentYuvTransform.columns.3, type: .float4, index: ALVRFunctionConstant.encodingYUVTransform3.rawValue)
        videoFilter = Renderer.videoFilterIndex(settings)
        videoSharpen = Renderer.effectiveSharpen(settings)
        fragmentConstants.setConstantValue(&videoFilter, type: .int, index: ALVRFunctionConstant.videoFilter.rawValue)
        fragmentConstants.setConstantValue(&videoSharpen, type: .float, index: ALVRFunctionConstant.videoSharpen.rawValue)
        
        let fragmentFunction = try library?.makeFunction(name: "videoFrameFragmentShader_" + variantName, constantValues: fragmentConstants)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "VideoFrameRenderPipeline_" + variantName
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        //pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor

        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat

        pipelineDescriptor.maxVertexAmplificationCount = viewCount
        
        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    // Advances the uniform buffer for the next frame, values can be written to `uniforms`
    // after this is called.
    private func updateDynamicBufferState() {
        /// Update the state of our uniform buffers before rendering

        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight
        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to:UniformsArray.self, capacity:1)
    }
    
    // Advances the Plane uniform buffer, values can be written to `planeUniforms`
    // after this is called.
    private func selectNextPlaneUniformBuffer() {
        /// Update the state of our uniform buffers before rendering

        planeUniformBufferIndex = (planeUniformBufferIndex + 1) % maxPlanesDrawn
        planeUniformBufferOffset = alignedPlaneUniformSize * planeUniformBufferIndex
        planeUniforms = UnsafeMutableRawPointer(dynamicPlaneUniformBuffer.contents() + planeUniformBufferOffset).bindMemory(to:PlaneUniform.self, capacity:1)
    }

    // Writes FOV/tangents/etc information to the uniform buffer.
    private func updateGameStateForVideoFrame(_ whichIdx: Int, drawable: LayerRenderer.Drawable?, viewTransforms: [simd_float4x4], sentViewTangents: [simd_float4], realViewTangents: [simd_float4], nearZ: Double, farZ: Double, framePose: simd_float4x4, simdDeviceAnchor: simd_float4x4) {
        let settings = ALVRClientApp.gStore.settings
        func uniforms(forViewIndex viewIndex: Int) -> Uniforms {
            let realTangents = realViewTangents[viewIndex]
            let sentTangents = sentViewTangents[viewIndex]
            
            var framePoseNoTranslation = framePose
            var simdDeviceAnchorNoTranslation = simdDeviceAnchor
            framePoseNoTranslation.columns.3 = simd_float4(0.0, 0.0, 0.0, 1.0)
            simdDeviceAnchorNoTranslation.columns.3 = simd_float4(0.0, 0.0, 0.0, 1.0)
            let viewMatrix = (simdDeviceAnchor * viewTransforms[viewIndex]).inverse
            let viewMatrixFrame = (framePoseNoTranslation.inverse * simdDeviceAnchorNoTranslation * viewTransforms[viewIndex]).inverse
            let viewMatrixFrameRk = (framePoseNoTranslation.inverse * simdDeviceAnchorNoTranslation).inverse // RealityKit implicitly applies the view transforms when we draw the quad entity
            var projection = matrix_identity_float4x4
            if #available(visionOS 2.0, *), drawable != nil {
#if XCODE_BETA_16
                projection = drawable!.computeProjection(viewIndex: viewIndex)
#else
                let p = ProjectiveTransform3D(leftTangent: Double(realTangents[0]),
                          rightTangent: Double(realTangents[1]),
                          topTangent: Double(realTangents[2]),
                          bottomTangent: Double(realTangents[3]),
                          nearZ: nearZ,
                          farZ: farZ,
                          reverseZ: true)
                projection = matrix_float4x4(p)
#endif
            }
            else {
                let p = ProjectiveTransform3D(leftTangent: Double(realTangents[0]),
                          rightTangent: Double(realTangents[1]),
                          topTangent: Double(realTangents[2]),
                          bottomTangent: Double(realTangents[3]),
                          nearZ: nearZ,
                          farZ: farZ,
                          reverseZ: true)
                projection = matrix_float4x4(p)
            }
            return Uniforms(projectionMatrix: projection, modelViewMatrixFrame: isRealityKit ? viewMatrixFrameRk : viewMatrixFrame, modelViewMatrix: viewMatrix, tangents: sentTangents)
        }
        
        self.uniforms[0].uniforms.0 = uniforms(forViewIndex: 0)
        if viewTransforms.count > 1 {
            self.uniforms[0].uniforms.1 = uniforms(forViewIndex: 1)
        }
    }
    
    // Checks if eye tracking was secretly added, maybe, hard to know really.
    func checkEyes(drawable: LayerRenderer.Drawable) {
        print("begin -----")
        print(drawable.colorTextures.first?.width as Any, drawable.colorTextures.first?.height as Any)
        //print(drawable.views[0].transform - EventHandler.shared.viewTransforms[0])
        //print(drawable.views[1].transform - EventHandler.shared.viewTransforms[1])
        if let vrr = drawable.rasterizationRateMaps.first {
            let eyeCenterX = Float(vrr.screenSize.width) / 2.0
            let eyeCenterY = Float(vrr.screenSize.height) / 2.0
            let physSizeL = vrr.physicalSize(layer: 0)
            let physCoordsL = vrr.physicalCoordinates(screenCoordinates: MTLCoordinate2D(x: eyeCenterX, y: eyeCenterY), layer: 0)
            
            let physSizeR = vrr.physicalSize(layer: 1)
            let physCoordsR = vrr.physicalCoordinates(screenCoordinates: MTLCoordinate2D(x: eyeCenterX, y: eyeCenterY), layer: 1)
            
            print(physSizeL, physSizeR, vrr.screenSize.width, vrr.screenSize.height, ":::", Float(physCoordsL.x) / Float(physSizeL.width), Float(physCoordsL.y) / Float(physSizeL.height), ":::", Float(physCoordsR.x) / Float(physSizeR.width), Float(physCoordsR.y) / Float(physSizeR.height))
        }
        
        print("end -------")
    }
    
    // Adjust view transforms for debugging various issues.
    func fixTransform(_ transform: simd_float4x4) -> simd_float4x4 {
        //var out = matrix_identity_float4x4
        //out.columns.3 = transform.columns.3
        //out.columns.3.w = 1.0
        return transform.translationOnly() // TODO: undo this when we fix canted views in 20.15
    }
    
    // Adjusts view tangents for debugging various issues.
    func fixTangents(_ tangents: simd_float4) -> simd_float4 {
        return tangents
    }
    
    func renderToDrawable(_ drawable: LayerRenderer.Drawable) {
        
    }

    // Render the frame, only used in MetalClientSystem renderer.
    func renderFrame() {
        /// Per frame updates hare
        EventHandler.shared.framesRendered += 1
        EventHandler.shared.totalFramesRendered += 1
        var streamingActiveForFrame = EventHandler.shared.streamingActive
        var isReprojected = false
        
        var queuedFrame:QueuedFrame? = nil
        
        guard let frame = layerRenderer!.queryNextFrame() else { return }
        guard let timing = frame.predictTiming() else { return }

        DisplayRateWatch.shared.observe(optimalInputTime: timing.optimalInputTime)
        frame.startUpdate()
        frame.endUpdate()
        // Client setting "tracking send phase" (GlobalSettings.trackingSendPhase): the head pose goes
        // out at optimalInputTime + a controlled phase, sent while this thread waits anyway, instead
        // of in the middle of the frame's work. See TrackingSendPhase at the end of this file.
        let trackingSendPhase = ALVRClientApp.gStore.settings.trackingSendPhase
        if trackingSendPhase {
            TrackingSendPhase.shared.wait(until: timing.optimalInputTime)
            TrackingSendPhase.shared.plan(timing: timing)
        }
        else {
            TrackingSendPhase.shared.cancel()
            LayerRenderer.Clock().wait(until: timing.optimalInputTime)
        }
        let diagWakeTime = CACurrentMediaTime()
        frame.startSubmission()
        
        roundTripRenderTime = CACurrentMediaTime() - lastRoundTripRenderTimestamp
        lastRoundTripRenderTimestamp = CACurrentMediaTime()

        // Client setting "single frame buffer" (GlobalSettings.singleFrameBuffer): take the newest
        // decoded frame instead of the oldest queued one, and never hold a frame back to let the
        // queue refill. The default two-frame policy left the displayed frame one behind whenever
        // two were queued -- measured as 7-13ms of decoder_queue with JPEG XS (2026-09-13).
        let singleFrameBuffer = ALVRClientApp.gStore.settings.singleFrameBuffer
        // Client setting "late frame pickup" (GlobalSettings.lateFramePickup): instead of picking the
        // video frame at optimalInputTime, wait until renderingDeadline minus our measured pick-to-GPU-done
        // budget, so the frame shown is fresher by the slack we used to leave before the deadline. See
        // LateFramePickup at the end of this file.
        let lateFramePickup = ALVRClientApp.gStore.settings.lateFramePickup
        if lateFramePickup {
            let budget = LateFramePickup.shared.budgetSeconds()
            let pickAt = timing.renderingDeadline.advanced(by: .nanoseconds(-Int64(budget * 1e9)))
            if pickAt > timing.optimalInputTime {
                if trackingSendPhase {
                    TrackingSendPhase.shared.wait(until: pickAt)
                }
                else {
                    LayerRenderer.Clock().wait(until: pickAt)
                }
            }
        }
        let startPollTime = CACurrentMediaTime()
        var pickedFromQueue = false
        while true {
            sched_yield()
            
            // If visionOS skipped our last frame, let the queue fill up a bit
            if EventHandler.shared.lastQueuedFrame != nil && !singleFrameBuffer {
                if EventHandler.shared.lastQueuedFrame!.timestamp != EventHandler.shared.lastSubmittedTimestamp && EventHandler.shared.frameQueue.count < 2 {
                    queuedFrame = EventHandler.shared.lastQueuedFrame
                    EventHandler.shared.framesRendered -= 1
                    isReprojected = false
                    break
                }
            }
            
            objc_sync_enter(EventHandler.shared.frameQueueLock)
            if singleFrameBuffer && EventHandler.shared.frameQueue.count > 1 {
                // Newest frame wins: an older one would only be shown late.
                EventHandler.shared.frameQueue.removeFirst(EventHandler.shared.frameQueue.count - 1)
            }
            queuedFrame = EventHandler.shared.frameQueue.count > 0 ? EventHandler.shared.frameQueue.removeFirst() : nil
            objc_sync_exit(EventHandler.shared.frameQueueLock)
            if queuedFrame != nil {
                pickedFromQueue = true
                break
            }
            
            // Picking late leaves no room to wait for a frame that is not there yet.
            if CACurrentMediaTime() - startPollTime > (lateFramePickup ? 0.0005 : 0.005) {
                //EventHandler.shared.framesRendered -= 1
                break
            }
        }
        
        // Recycle old frame with old timestamp/anchor (visionOS doesn't do timewarp for us?)
        if queuedFrame == nil && EventHandler.shared.lastQueuedFrame != nil {
            //print("Using last frame...")
            queuedFrame = EventHandler.shared.lastQueuedFrame
            EventHandler.shared.framesRendered -= 1
            isReprojected = true
        }
        
        if queuedFrame == nil && streamingActiveForFrame {
            streamingActiveForFrame = false
        }
        let renderingStreaming = streamingActiveForFrame && queuedFrame != nil
        
        let diagPickTime = CACurrentMediaTime()
        if trackingSendPhase && pickedFromQueue, let queuedTime = queuedFrame?.queuedTime, queuedTime > 0 {
            TrackingSendPhase.shared.recordFrameBuffering(diagPickTime - queuedTime)
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Failed to create command buffer")
        }

#if XCODE_BETA_26
        // MTLSize(width: 4338, height: 3478, depth: 1) MTLSize(width: 1888, height: 1792, depth: 1) by default
        // MTLSize(width: 6262, height: 5020, depth: 1) MTLSize(width: 2496, height: 2432, depth: 1) at 1.0 render quality
        var mainDrawable = LayerRenderer.Drawable()
        var drawables: [LayerRenderer.Drawable] = []
        if #available(visionOS 26.0, *) {
            drawables = frame.queryDrawables()
            if drawables.isEmpty {
                if queuedFrame != nil {
                    EventHandler.shared.lastQueuedFrame = queuedFrame
                }
                return
            }
            mainDrawable = drawables[0]
            //print(drawable.rasterizationRateMaps[0].screenSize, drawable.rasterizationRateMaps[0].physicalSize(layer: 0))
        }
        else {
            guard let _drawable = frame.queryDrawable() else {
                if queuedFrame != nil {
                    EventHandler.shared.lastQueuedFrame = queuedFrame
                }
                return
            }
            mainDrawable = _drawable
            drawables = [_drawable]
        }
#else
        guard let _drawable = frame.queryDrawable() else {
            if queuedFrame != nil {
                EventHandler.shared.lastQueuedFrame = queuedFrame
            }
            return
        }
        var mainDrawable = _drawable
        var drawables: [LayerRenderer.Drawable] = [_drawable]
#endif
        
        RateMapDiag.shared.observe(drawable: mainDrawable)

        let nalViewsPtr = UnsafeMutablePointer<AlvrViewParams>.allocate(capacity: 2)
        defer { nalViewsPtr.deallocate() }
        
        if queuedFrame != nil && !queuedFrame!.viewParamsValid /*&& EventHandler.shared.lastSubmittedTimestamp != queuedFrame!.timestamp*/ {
            alvr_report_compositor_start(queuedFrame!.timestamp, nalViewsPtr)
            queuedFrame = QueuedFrame(imageBuffer: queuedFrame!.imageBuffer, timestamp: queuedFrame!.timestamp, viewParamsValid: true, viewParams: [nalViewsPtr[0], nalViewsPtr[1]])
        }

        if EventHandler.shared.alvrInitialized && streamingActiveForFrame {
            let settings = ALVRClientApp.gStore.settings
            var ipd = mainDrawable.views.count > 1 ? simd_length(mainDrawable.views[0].transform.columns.3 - mainDrawable.views[1].transform.columns.3) : 0.063
#if targetEnvironment(simulator)
            ipd = 0.063
            print(EventHandler.shared.lastIpd)
#endif
            if abs(EventHandler.shared.lastIpd - ipd) > 0.001 {
                print("Send view config")
                
                if EventHandler.shared.lastIpd != -1 {
                    print("IPD changed!", EventHandler.shared.lastIpd, "->", ipd)
                }
                else {
                    print("IPD is", ipd)
                    EventHandler.shared.framesRendered = 0
                    lastReconfigureTime = CACurrentMediaTime()
                    
                    let rebuildThread = Thread {
                        self.rebuildRenderPipelines()
                    }
                    rebuildThread.name = "Rebuild Render Pipelines Thread"
                    rebuildThread.start()
                }
                
                // TODO: Fix tangents to fill entire FoV if canting ever becomes more intense like PFD was
                let tangentsLeft = mainDrawable.gimmeTangents(viewIndex: 0)
                let tangentsRight = mainDrawable.gimmeTangents(viewIndex: 1)
                let transformLeft = mainDrawable.views[0].transform
                let transformRight = mainDrawable.views[1].transform
                let leftAngles = atan(tangentsLeft * settings.fovRenderScale)
                let rightAngles = mainDrawable.views.count > 1 ? atan(tangentsRight * settings.fovRenderScale) : leftAngles
                let leftFov = AlvrFov(left: -leftAngles.x, right: leftAngles.y, up: leftAngles.z, down: -leftAngles.w)
                let rightFov = AlvrFov(left: -rightAngles.x, right: rightAngles.y, up: rightAngles.z, down: -rightAngles.w)
                
                let (leftFovCorrected, tangentsLeftCorrected, transformLeftCorrected) = mainDrawable._cantedViewToProportionalCircumscribedOrthogonal(fov: leftFov, viewTransform: transformLeft, fovPostScale: 1.0)
                let (rightFovCorrected, tangentsRightCorrected, transformRightCorrected) = mainDrawable._cantedViewToProportionalCircumscribedOrthogonal(fov: rightFov, viewTransform: transformRight, fovPostScale: 1.0)
                
                EventHandler.shared.viewFovs = [leftFovCorrected, rightFovCorrected]
                EventHandler.shared.viewTransforms = [fixTransform(transformLeftCorrected), mainDrawable.views.count > 1 ? fixTransform(transformRightCorrected) : fixTransform(transformLeftCorrected)]
                EventHandler.shared.lastIpd = ipd
                
                // Wait, is this correct?
                if mainDrawable.views.count > 1 {
                    EventHandler.shared.sentViewTangents = [tangentsLeftCorrected, tangentsRightCorrected]
                    EventHandler.shared.realViewTangents = [tangentsLeft, tangentsRight]
                }
                else {
                    EventHandler.shared.sentViewTangents = [tangentsLeftCorrected, tangentsRightCorrected]
                    EventHandler.shared.realViewTangents = [tangentsLeft, tangentsLeft]
                }
                
                
                WorldTracker.shared.sendViewParams(viewTransforms:  EventHandler.shared.viewTransforms, viewFovs: EventHandler.shared.viewFovs)
            }
            
            var needsPipelineRebuild = false
            if EventHandler.shared.encodingGamma != encodingGamma {
                needsPipelineRebuild = true
            }
            
            if CACurrentMediaTime() - lastReconfigureTime > 1.0 && (settings.chromaKeyEnabled != chromaKeyEnabled || settings.chromaKeyColorR != chromaKeyColor.x || settings.chromaKeyColorG != chromaKeyColor.y || settings.chromaKeyColorB != chromaKeyColor.z || settings.chromaKeyDistRangeMin != chromaKeyLerpDistRange.x || settings.chromaKeyDistRangeMax != chromaKeyLerpDistRange.y || Renderer.videoFilterIndex(settings) != videoFilter || Renderer.effectiveSharpen(settings) != videoSharpen) {
                lastReconfigureTime = CACurrentMediaTime()
                needsPipelineRebuild = true
            }
            
            if let videoFormat = EventHandler.shared.videoFormat {
                let nextYuvTransform = VideoHandler.getYUVTransformForVideoFormat(videoFormat)
                if nextYuvTransform != currentYuvTransform {
                    needsPipelineRebuild = true
                }
                currentYuvTransform = nextYuvTransform
            }
            
            if needsPipelineRebuild {
                lastReconfigureTime = CACurrentMediaTime()
                let rebuildThread = Thread {
                    self.rebuildRenderPipelines()
                }
                rebuildThread.name = "Rebuild Render Pipelines Thread"
                rebuildThread.start()
            }
        }
        
        //checkEyes(drawable: mainDrawable)
        
        objc_sync_enter(EventHandler.shared.frameQueueLock)
        EventHandler.shared.framesSinceLastDecode += 1
        objc_sync_exit(EventHandler.shared.frameQueueLock)
        
        if queuedFrame != nil && !queuedFrame!.viewParamsValid {
            print("aaaaaaaa bad view params")
        }
        
        let vsyncTime = LayerRenderer.Clock.Instant.epoch.duration(to: mainDrawable.frameTiming.presentationTime).timeInterval
        let vsyncTimeNs = UInt64(vsyncTime * Double(NSEC_PER_SEC))
        let framePreviouslyPredictedPose = queuedFrame != nil ? WorldTracker.shared.convertSteamVRViewPose(queuedFrame!.viewParams) : nil
        if ALVRClientApp.gStore.settings.showPerformanceHud {
            PerformanceTracker.shared.recordFramePresentation(presentationTime: mainDrawable.frameTiming.presentationTime, timestampNs: queuedFrame?.timestamp)
            PerformanceTracker.shared.logIfNeeded(presentationTime: mainDrawable.frameTiming.presentationTime)
        }
        
        // Do NOT move this, just in case, because DeviceAnchor is wonkey and every DeviceAnchor mutates each other.
        if EventHandler.shared.alvrInitialized && EventHandler.shared.lastIpd != -1 {
            if #available(visionOS 2.0, *) {
                EventHandler.shared.viewTransforms = [fixTransform(mainDrawable.views[0].transform), mainDrawable.views.count > 1 ? fixTransform(mainDrawable.views[1].transform) : fixTransform(mainDrawable.views[0].transform)]
            }
            // TODO: I suspect Apple changes view transforms every frame to account for pupil swim, figure out how to fit the latest view transforms in?
            // Since pupil swim is purely an axial thing, maybe we can just timewarp the view transforms as well idk
            let viewFovs = EventHandler.shared.viewFovs
            let viewTransforms = EventHandler.shared.viewTransforms

            let targetTimestamp = vsyncTime// + (Double(min(alvr_get_head_prediction_offset_ns(), WorldTracker.maxPrediction)) / Double(NSEC_PER_SEC))
            let reportedTargetTimestamp = vsyncTime
            var anchorTimestamp = vsyncTime// + (Double(min(alvr_get_head_prediction_offset_ns(), WorldTracker.maxPrediction)) / Double(NSEC_PER_SEC))//LayerRenderer.Clock.Instant.epoch.duration(to: mainDrawable.frameTiming.trackableAnchorTime).timeInterval

            if !ALVRClientApp.gStore.settings.targetHandsAtRoundtripLatency {
                if #available(visionOS 2.0, *) {
                    anchorTimestamp = LayerRenderer.Clock.Instant.epoch.duration(to: mainDrawable.frameTiming.trackableAnchorTime).timeInterval
                }
                else {
                    anchorTimestamp = LayerRenderer.Clock.Instant.epoch.duration(to: mainDrawable.frameTiming.renderingDeadline).timeInterval
                }
            }

            if trackingSendPhase {
                // Planned at the start of the frame; due now or while this thread next waits.
                TrackingSendPhase.shared.sendIfDue(drawableTiming: mainDrawable.frameTiming)
            }
            else {
                WorldTracker.shared.sendTracking(viewTransforms: viewTransforms, viewFovs: viewFovs, targetTimestamp: targetTimestamp, reportedTargetTimestamp: reportedTargetTimestamp, anchorTimestamp: anchorTimestamp, delay: 0.0)
            }
        }
        else {
#if DEBUG_ALVR_TRACKING
            let viewFovs = EventHandler.shared.viewFovs
            let viewTransforms = EventHandler.shared.viewTransforms

            let targetTimestamp = vsyncTime// + (Double(min(alvr_get_head_prediction_offset_ns(), WorldTracker.maxPrediction)) / Double(NSEC_PER_SEC))
            let reportedTargetTimestamp = vsyncTime
            var anchorTimestamp = vsyncTime// + (Double(min(alvr_get_head_prediction_offset_ns(), WorldTracker.maxPrediction)) / Double(NSEC_PER_SEC))//LayerRenderer.Clock.Instant.epoch.duration(to: mainDrawable.frameTiming.trackableAnchorTime).timeInterval
            
            WorldTracker.shared.sendTracking(viewTransforms: viewTransforms, viewFovs: viewFovs, targetTimestamp: targetTimestamp, reportedTargetTimestamp: reportedTargetTimestamp, anchorTimestamp: anchorTimestamp, delay: 0.0)
#endif
        }
        
        let deviceAnchor = WorldTracker.shared.worldTracking.queryDeviceAnchor(atTimestamp: vsyncTime)
        
        commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
            if EventHandler.shared.alvrInitialized && queuedFrame != nil && EventHandler.shared.lastSubmittedTimestamp != queuedFrame?.timestamp {
                let currentTimeNs = UInt64(CACurrentMediaTime() * Double(NSEC_PER_SEC))
                
                //let currentTimeNs = UInt64(commandBuffer.gpuEndTime * Double(NSEC_PER_SEC))
                //let actualRenderTimeNs = UInt64((commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * Double(NSEC_PER_SEC))
                //print((commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000.0)
                
                //print("Finished:", queuedFrame!.timestamp)
                alvr_report_submit(queuedFrame!.timestamp, vsyncTimeNs &- currentTimeNs)

                EventHandler.shared.lastSubmittedTimestamp = queuedFrame!.timestamp
            }
            if ALVRClientApp.gStore.settings.showPerformanceHud, let ts = queuedFrame?.timestamp {
                PerformanceTracker.shared.recordSubmit(timestampNs: ts)
                PerformanceTracker.shared.recordGpu(timestampNs: ts, gpuStartTime: commandBuffer.gpuStartTime, gpuEndTime: commandBuffer.gpuEndTime)
            }
        }

        // List of reasons to not display a frame
        var frameIsSuitableForDisplaying = true
        //print(EventHandler.shared.lastIpd, WorldTracker.shared.worldTrackingAddedOriginAnchor, EventHandler.shared.framesRendered)
        if EventHandler.shared.lastIpd == -1 || EventHandler.shared.framesRendered < Int(refreshRate) {
            // Don't show frame if we haven't sent the view config and received frames
            // with that config applied.
            frameIsSuitableForDisplaying = false
            print("IPD is bad, no frame", EventHandler.shared.framesRendered)
        }
        if !WorldTracker.shared.worldTrackingAddedOriginAnchor && EventHandler.shared.framesRendered < 300 {
            // Don't show frame if we haven't figured out our origin yet.
            frameIsSuitableForDisplaying = false
            print("Origin is bad, no frame")
        }
        if EventHandler.shared.videoFormat == nil {
            frameIsSuitableForDisplaying = false
            print("Missing video format, no frame")
        }
        
        if !(renderingStreaming && frameIsSuitableForDisplaying && queuedFrame != nil) {
            // Things to do once if no frame is rendering (show the wireframe)
            if EventHandler.shared.totalFramesRendered > 300 {
                fadeInOverlayAlpha += 0.02
            }
        }
        else {
            // Things to do if a frame is rendering (fade wireframe, or show wireframe is reprojection is too long)
            fadeInOverlayAlpha -= 0.01
            if fadeInOverlayAlpha < 0.0 {
                fadeInOverlayAlpha = 0.0
            }
            
            if isReprojected && useApplesReprojection {
                LayerRenderer.Clock().wait(until: mainDrawable.frameTiming.renderingDeadline)
            }
            
            if isReprojected {
                reprojectedFramesInARow += 1
                if reprojectedFramesInARow > Int(refreshRate) {
                    fadeInOverlayAlpha += 0.02
                }
            }
            else {
                reprojectedFramesInARow = 0
                fadeInOverlayAlpha -= 0.02
            }
        }

        
        if ALVRClientApp.gStore.settings.showPerformanceHud,
           renderingStreaming,
           frameIsSuitableForDisplaying,
           let queuedFrame {
            PerformanceTracker.shared.recordCompositorStart(timestampNs: queuedFrame.timestamp)
        }
        for drawable in drawables {
            drawable.deviceAnchor = deviceAnchor
            
            // TODO: check layerRenderer.configuration.layout == .layered ?
            let viewports = drawable.views.map { $0.textureMap.viewport }
            let rasterizationRateMap = drawable.rasterizationRateMaps.first
            let viewTransforms = drawable.views.map { $0.transform }
            let sentViewTangents = drawable.views.enumerated().map { (idx, v) in EventHandler.shared.sentViewTangents[idx] }
            let realViewTangents = drawable.views.enumerated().map { (idx, v) in EventHandler.shared.realViewTangents[idx] }
            let nearZ =  Double(drawable.depthRange.y)
            let farZ = Double(drawable.depthRange.x)
            let simdDeviceAnchor = WorldTracker.shared.floorCorrectionTransform.asFloat4x4() * (deviceAnchor?.originFromAnchorTransform ?? matrix_identity_float4x4)
            let framePose = framePreviouslyPredictedPose ?? matrix_identity_float4x4
            
#if DEBUG_ALVR_TRACKING
            WorldTracker.shared.lockDebuggables()
            
            let headDebug = (simdDeviceAnchor.columns.3.asFloat3() - simdDeviceAnchor.columns.2.asFloat3()).asFloat4x4() * simdDeviceAnchor.orientationOnly().asFloat4x4()
            WorldTracker.shared.addDebuggablePose(headDebug, 0.1)
            
            for transform in viewTransforms {
                let mat = headDebug * transform
                WorldTracker.shared.addDebuggablePose(mat, 0.05)
            }
            
            WorldTracker.shared.unlockDebuggables()
#endif
            
            if renderingStreaming && frameIsSuitableForDisplaying && queuedFrame != nil {
                //print("render")
                if let encoder = beginRenderStreamingFrame(0, commandBuffer: commandBuffer, renderTargetColor: drawable.colorTextures[0], renderTargetDepth: drawable.depthTextures[0], viewports: viewports, viewTransforms: viewTransforms, sentViewTangents: sentViewTangents, realViewTangents: realViewTangents, nearZ: nearZ, farZ: farZ, rasterizationRateMap: rasterizationRateMap, queuedFrame: queuedFrame, framePose: framePose, simdDeviceAnchor: simdDeviceAnchor, drawable: drawable) {
                    renderStreamingFrame(0, commandBuffer: commandBuffer, renderEncoder: encoder, renderTargetColor: drawable.colorTextures[0], renderTargetDepth: drawable.depthTextures[0], viewports: viewports, viewTransforms: viewTransforms, sentViewTangents: sentViewTangents, realViewTangents: realViewTangents, nearZ: nearZ, farZ: farZ, rasterizationRateMap: rasterizationRateMap, framePose: framePose, simdDeviceAnchor: simdDeviceAnchor)
                    endRenderStreamingFrame(renderEncoder: encoder)
                }
                renderStreamingFrameOverlays(0, commandBuffer: commandBuffer, renderTargetColor: drawable.colorTextures[0], renderTargetDepth: drawable.depthTextures[0], viewports: viewports, viewTransforms: viewTransforms, sentViewTangents: sentViewTangents, realViewTangents: realViewTangents, nearZ: nearZ, farZ: farZ, rasterizationRateMap: rasterizationRateMap, queuedFrame: queuedFrame, framePose: framePose, simdDeviceAnchor: simdDeviceAnchor, drawable: drawable)
            }
            else
            {
                reprojectedFramesInARow = 0;

                let noFramePose = simdDeviceAnchor
                // TODO: draw a cool loading logo
                renderNothing(0, commandBuffer: commandBuffer, renderTargetColor: drawable.colorTextures[0], renderTargetDepth: drawable.depthTextures[0], viewports: viewports, viewTransforms: viewTransforms, sentViewTangents: sentViewTangents, realViewTangents: realViewTangents, nearZ: nearZ, farZ: farZ, rasterizationRateMap: rasterizationRateMap, queuedFrame: queuedFrame, framePose: noFramePose, simdDeviceAnchor: simdDeviceAnchor, drawable: drawable)
                
                renderOverlay(commandBuffer: commandBuffer, renderTargetColor: drawable.colorTextures[0], renderTargetDepth: drawable.depthTextures[0], viewports: viewports, viewTransforms: viewTransforms, sentViewTangents: sentViewTangents, realViewTangents: realViewTangents, nearZ: nearZ, farZ: farZ, rasterizationRateMap: rasterizationRateMap, queuedFrame: queuedFrame, framePose: noFramePose, simdDeviceAnchor: simdDeviceAnchor)
                if !isRealityKit {
                    renderStreamingFrameDepth(commandBuffer: commandBuffer, renderTargetColor: drawable.colorTextures[0], renderTargetDepth: drawable.depthTextures[0], viewports: viewports, viewTransforms: viewTransforms, sentViewTangents: sentViewTangents, realViewTangents: realViewTangents, nearZ: nearZ, farZ: farZ, rasterizationRateMap: rasterizationRateMap, queuedFrame: queuedFrame)
                }
            }
            
            drawable.encodePresent(commandBuffer: commandBuffer)
        }
        
        coolPulsingColorsTime += 0.005
        if coolPulsingColorsTime > 4.0 {
            coolPulsingColorsTime = 0.0
        }
        
        if fadeInOverlayAlpha > 1.0 {
            fadeInOverlayAlpha = 1.0
        }
        if fadeInOverlayAlpha < 0.0 {
            fadeInOverlayAlpha = 0.0
        }
        
        if FrameTimingDiag.enabled || lateFramePickup {
            // See FrameTimingDiag. All times on the mach clock, the same base the
            // vsync_queue report above already mixes with CACurrentMediaTime().
            let diagCommitTime = CACurrentMediaTime()
            let diagOptimal = LayerRenderer.Clock.Instant.epoch.duration(to: timing.optimalInputTime).timeInterval
            let diagDeadline = LayerRenderer.Clock.Instant.epoch.duration(to: mainDrawable.frameTiming.renderingDeadline).timeInterval
            let diagPresent = LayerRenderer.Clock.Instant.epoch.duration(to: mainDrawable.frameTiming.presentationTime).timeInterval
            let diagNewFrame = queuedFrame != nil && !isReprojected
            commandBuffer.addCompletedHandler { cb in
                let diagDone = CACurrentMediaTime()
                LateFramePickup.shared.record(work: diagDone - diagPickTime, missedDeadline: diagDone > diagDeadline,
                                              active: lateFramePickup)
                if FrameTimingDiag.enabled {
                    FrameTimingDiag.shared.record(optimal: diagOptimal, deadline: diagDeadline, present: diagPresent,
                                                  wake: diagWakeTime, pick: diagPickTime, commit: diagCommitTime,
                                                  done: diagDone, gpuStart: cb.gpuStartTime, gpuEnd: cb.gpuEndTime,
                                                  newFrame: diagNewFrame)
                }
            }
        }
        commandBuffer.commit()
        frame.endSubmission()
        
        EventHandler.shared.lastQueuedFrame = queuedFrame
        EventHandler.shared.lastQueuedFramePose = framePreviouslyPredictedPose
    }
    
    // Pulse the wireframe between cyan and blue.
    func coolPulsingColor() -> simd_float4 {
        // Color picked from the ALVR logo
        let lightColor = simd_float4(0.05624, 0.73124, 0.75999, 1.0)
        let darkColor = simd_float4(0.01305, 0.26223, 0.63828, 1.0)
        var switchingFnT: Float = 0.0 // hold on light
        
        if coolPulsingColorsTime >= 1.0 && coolPulsingColorsTime < 2.0 {
            switchingFnT = coolPulsingColorsTime - 1.0 // light -> dark
        }
        else if coolPulsingColorsTime >= 2.0 && coolPulsingColorsTime < 3.0 {
            switchingFnT = 1.0 // hold on dark
        }
        else if coolPulsingColorsTime >= 3.0 && coolPulsingColorsTime < 4.0 {
            switchingFnT = coolPulsingColorsTime - 2.0 // dark -> light
        }

        var switchingFn = sin(switchingFnT * Float.pi * 0.5)
        if coolPulsingColorsTime >= 4.0 {
            switchingFn = 0.0
        }
        return simd_mix(lightColor, darkColor, simd_float4(repeating: switchingFn))
    }
    
    // Can draw planes with debug colors, or with a subtle transparency change based on the type.
    func planeToColor(plane: PlaneAnchor) -> simd_float4 {
        let planeAlpha = fadeInOverlayAlpha
        var subtleChange = 0.75 + ((Float(plane.id.hashValue & 0xFF) / Float(0xff)) * 0.25)
        
        if drawPlanesWithInformedColors {
            switch(plane.classification) {
                case .ceiling: // #62ea80
                    return simd_float4(0.3843137254901961, 0.9176470588235294, 0.5019607843137255, 1.0) * subtleChange * planeAlpha
                case .door: // #1a5ff4
                    return simd_float4(0.10196078431372549, 0.37254901960784315, 0.9568627450980393, 1.0) * subtleChange * planeAlpha
                case .floor: // #bf6505
                    return simd_float4(0.7490196078431373, 0.396078431372549, 0.0196078431372549, 1.0) * subtleChange * planeAlpha
                case .seat: // #ef67af
                    return simd_float4(0.9372549019607843, 0.403921568627451, 0.6862745098039216, 1.0) * subtleChange * planeAlpha
                case .table: // #c937d3
                    return simd_float4(0.788235294117647, 0.21568627450980393, 0.8274509803921568, 1.0) * subtleChange * planeAlpha
                case .wall: // #dced5e
                    return simd_float4(0.8627450980392157, 0.9294117647058824, 0.3686274509803922, 1.0) * subtleChange * planeAlpha
                case .window: // #4aefce
                    return simd_float4(0.2901960784313726, 0.9372549019607843, 0.807843137254902, 1.0) * subtleChange * planeAlpha
                case .unknown: // #0e576b
                    return simd_float4(0.054901960784313725, 0.3411764705882353, 0.4196078431372549, 1.0) * subtleChange * planeAlpha
                case .undetermined: // #749606
                    return simd_float4(0.4549019607843137, 0.5882352941176471, 0.023529411764705882, 1.0) * subtleChange * planeAlpha
                default:
                    return simd_float4(1.0, 0.0, 0.0, 1.0) * subtleChange * planeAlpha // red
            }
        }
        else {
            if plane.classification == .ceiling {
                subtleChange *= 0.4
            }
            else if plane.classification == .wall {
                subtleChange *= 0.1
            }
            else if plane.classification == .floor {
                subtleChange *= 0.2
            }
            else if plane.classification == .seat {
                subtleChange *= 0.5
            }
            else {
                subtleChange = 0.01
            }
            return coolPulsingColor() * subtleChange * planeAlpha
        }
    }
    
    // Line color for a given ARKit Plane
    func planeToLineColor(plane: PlaneAnchor) -> simd_float4 {
        let planeAlpha = fadeInOverlayAlpha
        let subtleChange = 0.75 + ((Float(plane.id.hashValue & 0xFF) / Float(0xff)) * 0.25)
        
        if drawPlanesWithInformedColors {
            return planeToColor(plane: plane)
        }
        else {
            return coolPulsingColor() * subtleChange * planeAlpha
        }
    }
    
    // Only renders the frame depth, used to correct depth after the overlay is rendered
    // because Apple's Metal renderer is kinda weird about it.
    func renderStreamingFrameDepth(commandBuffer: MTLCommandBuffer, renderTargetColor: MTLTexture, renderTargetDepth: MTLTexture, viewports: [MTLViewport], viewTransforms: [simd_float4x4], sentViewTangents: [simd_float4], realViewTangents: [simd_float4], nearZ: Double, farZ: Double, rasterizationRateMap: MTLRasterizationRateMap?, queuedFrame: QueuedFrame?) {
        if currentRenderColorFormat != renderTargetColor.pixelFormat && isRealityKit {
            return
        }

        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = renderTargetColor
        renderPassDescriptor.colorAttachments[0].loadAction = .load
        renderPassDescriptor.colorAttachments[0].storeAction = .dontCare
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: chromaKeyEnabled ? 0.0 : 1.0)
        renderPassDescriptor.depthAttachment.texture = renderTargetDepth
        renderPassDescriptor.depthAttachment.loadAction = .clear
        renderPassDescriptor.depthAttachment.storeAction = .store
        renderPassDescriptor.depthAttachment.clearDepth = 0.000000001
        renderPassDescriptor.rasterizationRateMap = rasterizationRateMap
        
        renderPassDescriptor.renderTargetArrayLength = viewports.count
        
        /// Final pass rendering code here
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render encoder")
        }
        
        renderEncoder.label = "Rerender depth"
        
        renderEncoder.pushDebugGroup("Draw ALVR Frame Depth")
        renderEncoder.setCullMode(.back)
        renderEncoder.setFrontFacing(.counterClockwise)
        renderEncoder.setRenderPipelineState(videoFrameDepthPipelineState)
        renderEncoder.setDepthStencilState(depthStateAlways)
#if !targetEnvironment(simulator)
        renderEncoder.setDepthClipMode(.clamp)
#endif
        
        renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        
        renderEncoder.setViewports(viewports)
        
        if viewports.count > 1 {
            var viewMappings = (0..<viewports.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }
        
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderEncoder.popDebugGroup()
        renderEncoder.endEncoding()
    }
    
    // Clears the render target, nothing more nothing less.
    func renderNothing(_ whichIdx: Int, commandBuffer: MTLCommandBuffer, renderTargetColor: MTLTexture, renderTargetDepth: MTLTexture, viewports: [MTLViewport], viewTransforms: [simd_float4x4], sentViewTangents: [simd_float4], realViewTangents: [simd_float4], nearZ: Double, farZ: Double, rasterizationRateMap: MTLRasterizationRateMap?, queuedFrame: QueuedFrame?, framePose: simd_float4x4, simdDeviceAnchor: simd_float4x4, drawable: LayerRenderer.Drawable?) {
        if currentRenderColorFormat != renderTargetColor.pixelFormat && isRealityKit {
            return
        }
        self.updateDynamicBufferState()
        
        self.updateGameStateForVideoFrame(whichIdx, drawable: drawable, viewTransforms: viewTransforms, sentViewTangents: sentViewTangents, realViewTangents: realViewTangents, nearZ: nearZ, farZ: farZ, framePose: framePose, simdDeviceAnchor: simdDeviceAnchor)
        
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = renderTargetColor
        renderPassDescriptor.colorAttachments[0].loadAction = isRealityKit ? (whichIdx == 0 ? .clear : .load) : .clear 
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)
        renderPassDescriptor.depthAttachment.texture = renderTargetDepth
        renderPassDescriptor.depthAttachment.loadAction = .clear
        renderPassDescriptor.depthAttachment.storeAction = .store
        renderPassDescriptor.depthAttachment.clearDepth = 0.0
        renderPassDescriptor.rasterizationRateMap = rasterizationRateMap
        
        renderPassDescriptor.renderTargetArrayLength = viewports.count

        
        /// Final pass rendering code here
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render encoder")
        }
        
        renderEncoder.label = "Rendering Nothing"
        
        renderEncoder.pushDebugGroup("Draw Nothing")
        renderEncoder.setCullMode(.back)
        renderEncoder.setFrontFacing(.counterClockwise)
        renderEncoder.setRenderPipelineState(videoFrameDepthPipelineState)
        renderEncoder.setDepthStencilState(depthStateAlways)
#if !targetEnvironment(simulator)
        renderEncoder.setDepthClipMode(.clamp)
#endif
        
        renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        renderEncoder.setVertexBuffer(dynamicPlaneUniformBuffer, offset:planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue) // unused
        
        renderEncoder.setViewports(viewports)
        
        if viewports.count > 1 {
            var viewMappings = (0..<viewports.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }
        
        renderEncoder.setVertexBuffer(fullscreenQuadBuffer, offset: 0, index: VertexAttribute.position.rawValue)
        renderEncoder.setVertexBuffer(fullscreenQuadBuffer, offset: (3*4)*4, index: VertexAttribute.texcoord.rawValue)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        
        renderEncoder.endEncoding()
    }
    
    // Renders a wireframe overlay on top of the existing video frame (or nothing)
    func renderOverlay(commandBuffer: MTLCommandBuffer, renderTargetColor: MTLTexture, renderTargetDepth: MTLTexture, viewports: [MTLViewport], viewTransforms: [simd_float4x4], sentViewTangents: [simd_float4], realViewTangents: [simd_float4],  nearZ: Double, farZ: Double, rasterizationRateMap: MTLRasterizationRateMap?, queuedFrame: QueuedFrame?, framePose: simd_float4x4, simdDeviceAnchor: simd_float4x4)
    {
        if currentRenderColorFormat != renderTargetColor.pixelFormat && isRealityKit {
            return
        }
        // Toss out the depth buffer, keep colors
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = renderTargetColor
        renderPassDescriptor.colorAttachments[0].loadAction = .load
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)
        renderPassDescriptor.depthAttachment.texture = renderTargetDepth
        renderPassDescriptor.depthAttachment.loadAction = .clear
        renderPassDescriptor.depthAttachment.storeAction = .dontCare
        renderPassDescriptor.depthAttachment.clearDepth = 0.0
        renderPassDescriptor.rasterizationRateMap = rasterizationRateMap
        
        renderPassDescriptor.renderTargetArrayLength = viewports.count
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render encoder")
        }
        renderEncoder.label = "Plane Render Encoder"
        renderEncoder.pushDebugGroup("Draw planes")
        renderEncoder.setCullMode(.back)
        renderEncoder.setFrontFacing(.counterClockwise)
        renderEncoder.setViewports(viewports)
        renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        renderEncoder.setVertexBuffer(dynamicPlaneUniformBuffer, offset:planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue) // unused
        
        if viewports.count > 1 {
            var viewMappings = (0..<viewports.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }
        
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setDepthStencilState(depthStateGreater)
#if !targetEnvironment(simulator)
        renderEncoder.setDepthClipMode(.clamp)
#endif

        WorldTracker.shared.lockPlaneAnchors()
        
        var firstBind = true
        if fadeInOverlayAlpha > 0.0 {
            // Render planes
            for planeEntry in WorldTracker.shared.planeAnchors {
                let plane = planeEntry.value
                let faces = plane.geometry.meshFaces
                
                // VRR can't do lines
                if faces.primitive != GeometryElement.Primitive.triangle {
                    continue
                }
                
                renderEncoder.setVertexBuffer(plane.geometry.meshVertices.buffer, offset: 0, index: VertexAttribute.position.rawValue)
                renderEncoder.setVertexBuffer(plane.geometry.meshVertices.buffer, offset: 0, index: VertexAttribute.texcoord.rawValue)
                
                //self.updateGameStateForVideoFrame(drawable: drawable, framePose: framePose, planeTransform: plane.originFromAnchorTransform)
                selectNextPlaneUniformBuffer()
                self.planeUniforms[0].planeTransform = plane.originFromAnchorTransform
                self.planeUniforms[0].planeColor = planeToColor(plane: plane)
                self.planeUniforms[0].planeDoProximity = 1.0
                if firstBind {
                    renderEncoder.setVertexBuffer(dynamicPlaneUniformBuffer, offset:planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue)
                    firstBind = false
                } else {
                    renderEncoder.setVertexBufferOffset(planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue)
                }
                
                renderEncoder.setTriangleFillMode(.fill)
                renderEncoder.drawIndexedPrimitives(type: faces.primitive == .triangle ? MTLPrimitiveType.triangle : MTLPrimitiveType.line,
                                                    indexCount: faces.count*3,
                                                    indexType: faces.bytesPerIndex == 2 ? MTLIndexType.uint16 : MTLIndexType.uint32,
                                                    indexBuffer: faces.buffer,
                                                    indexBufferOffset: 0)
            }
            
            // Render lines
            for planeEntry in WorldTracker.shared.planeAnchors {
                let plane = planeEntry.value
                let faces = plane.geometry.meshFaces
                
                // VRR can't do lines
                if faces.primitive != GeometryElement.Primitive.triangle {
                    continue
                }
                
                renderEncoder.setVertexBuffer(plane.geometry.meshVertices.buffer, offset: 0, index: VertexAttribute.position.rawValue)
                renderEncoder.setVertexBuffer(plane.geometry.meshVertices.buffer, offset: 0, index: VertexAttribute.texcoord.rawValue)
                
                //self.updateGameStateForVideoFrame(drawable: drawable, framePose: framePose, planeTransform: plane.originFromAnchorTransform)
                selectNextPlaneUniformBuffer()
                self.planeUniforms[0].planeTransform = plane.originFromAnchorTransform
                self.planeUniforms[0].planeColor = planeToLineColor(plane: plane)
                self.planeUniforms[0].planeDoProximity = 0.0
                if firstBind {
                    renderEncoder.setVertexBuffer(dynamicPlaneUniformBuffer, offset:planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue)
                    firstBind = false
                } else {
                    renderEncoder.setVertexBufferOffset(planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue)
                }
                
                renderEncoder.setTriangleFillMode(.lines)
                renderEncoder.drawIndexedPrimitives(type: faces.primitive == .triangle ? MTLPrimitiveType.triangle : MTLPrimitiveType.line,
                                                    indexCount: faces.count*3,
                                                    indexType: faces.bytesPerIndex == 2 ? MTLIndexType.uint16 : MTLIndexType.uint32,
                                                    indexBuffer: faces.buffer,
                                                    indexBufferOffset: 0)
            }
        }
        
        WorldTracker.shared.unlockPlaneAnchors()
        
        WorldTracker.shared.lockDebuggables()
        
        // Render lines
        for i in 0..<WorldTracker.shared.debuggableMats.count {
            let matTransform = WorldTracker.shared.debuggableMats[i]
            let basisColors = [simd_float4(1.0, 0.0, 0.0, 1.0), simd_float4(0.0, 1.0, 0.0, 1.0), simd_float4(0.0, 0.0, 1.0, 1.0)]
            let basisScaleFactor: Float = WorldTracker.shared.debuggableScales[i]
            let basisScale = simd_float3x3(basisScaleFactor).asFloat4x4()
            
            for i in 0..<3 {
                renderEncoder.setVertexBuffer(unitVectorXYZBuffer, offset: 0, index: VertexAttribute.position.rawValue)
                renderEncoder.setVertexBuffer(unitVectorXYZBuffer, offset: (3*4)*4, index: VertexAttribute.texcoord.rawValue)
                
                selectNextPlaneUniformBuffer()
                self.planeUniforms[0].planeTransform = (matTransform.columns.3.asFloat3()).asFloat4x4() * (basisScale * matTransform.orientationOnly().asFloat4x4())
                self.planeUniforms[0].planeColor = basisColors[i]
                self.planeUniforms[0].planeDoProximity = 0.0
                if firstBind {
                    renderEncoder.setVertexBuffer(dynamicPlaneUniformBuffer, offset:planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue)
                    firstBind = false
                } else {
                    renderEncoder.setVertexBufferOffset(planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue)
                }
                
                renderEncoder.setTriangleFillMode(.lines)
                renderEncoder.drawPrimitives(type: .triangle, vertexStart: 3*i, vertexCount: 3)
            }
        }

        // Render solid bodies (currently only the Steam Controller shell).
        if !WorldTracker.shared.debuggableMeshes.isEmpty, let meshBuffer = ibexMeshBuffer(), ibexMeshVertexCount > 0 {
            for (matTransform, color) in WorldTracker.shared.debuggableMeshes {
                renderEncoder.setVertexBuffer(meshBuffer, offset: 0, index: VertexAttribute.position.rawValue)
                // Texcoords are unused by the plane fragment shader, which is
                // flat-coloured — bind the position buffer again rather than
                // carrying a second one purely to satisfy the descriptor.
                renderEncoder.setVertexBuffer(meshBuffer, offset: 0, index: VertexAttribute.texcoord.rawValue)

                selectNextPlaneUniformBuffer()
                self.planeUniforms[0].planeTransform = matTransform
                self.planeUniforms[0].planeColor = color
                self.planeUniforms[0].planeDoProximity = 0.0
                if firstBind {
                    renderEncoder.setVertexBuffer(dynamicPlaneUniformBuffer, offset:planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue)
                    firstBind = false
                } else {
                    renderEncoder.setVertexBufferOffset(planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue)
                }

                renderEncoder.setTriangleFillMode(.lines)
                renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: ibexMeshVertexCount)
            }
        }

        WorldTracker.shared.debuggableMats = []
        WorldTracker.shared.debuggableScales = []
        WorldTracker.shared.debuggableMeshes = []
        WorldTracker.shared.unlockDebuggables()
        
        renderEncoder.popDebugGroup()
        renderEncoder.endEncoding()
    }

    // Renders the chaperone proximity overlay on top of the existing video frame.
    func renderChaperone(commandBuffer: MTLCommandBuffer, renderTargetColor: MTLTexture, renderTargetDepth: MTLTexture, viewports: [MTLViewport], rasterizationRateMap: MTLRasterizationRateMap?, simdDeviceAnchor: simd_float4x4) {
        if currentRenderColorFormat != renderTargetColor.pixelFormat && isRealityKit {
            return
        }

        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = renderTargetColor
        renderPassDescriptor.colorAttachments[0].loadAction = .load
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)
        renderPassDescriptor.depthAttachment.texture = renderTargetDepth
        renderPassDescriptor.depthAttachment.loadAction = .clear
        renderPassDescriptor.depthAttachment.storeAction = .dontCare
        renderPassDescriptor.depthAttachment.clearDepth = 0.0
        renderPassDescriptor.rasterizationRateMap = rasterizationRateMap
        renderPassDescriptor.renderTargetArrayLength = viewports.count

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render encoder")
        }
        renderEncoder.label = "Chaperone Render Encoder"
        renderEncoder.pushDebugGroup("Draw chaperone")
        renderEncoder.setCullMode(.back)
        renderEncoder.setFrontFacing(.counterClockwise)
        renderEncoder.setViewports(viewports)
        renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        renderEncoder.setVertexBuffer(dynamicPlaneUniformBuffer, offset:planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue) // unused

        if viewports.count > 1 {
            var viewMappings = (0..<viewports.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }

        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setDepthStencilState(depthStateGreater)
#if !targetEnvironment(simulator)
        renderEncoder.setDepthClipMode(.clamp)
#endif

        let planesSnapshot = WorldTracker.shared.snapshotPlaneData()
#if CHAPERONE_PROFILE
        let profileStart = CACurrentMediaTime()
#endif
        let output = chaperoneSystem.update(planes: planesSnapshot,
                                            headPose: simdDeviceAnchor,
                                            leftHandPose: WorldTracker.shared.lastLeftHandPose,
                                            rightHandPose: WorldTracker.shared.lastRightHandPose,
                                            worldFromSteamVR: WorldTracker.shared.worldTrackingSteamVRTransform,
                                            chaperoneDistanceCm: ALVRClientApp.gStore.settings.chaperoneDistanceCm,
                                            now: CACurrentMediaTime(),
                                            leftControllerPresent: WorldTracker.shared.leftControllerPose != nil,
                                            rightControllerPresent: WorldTracker.shared.rightControllerPose != nil) { isLeft, amplitude, duration in
            WorldTracker.shared.enqueueHapticsPulse(isLeft: isLeft, amplitude: amplitude, duration: duration)
        }

#if CHAPERONE_PROFILE
        let elapsedMs = (CACurrentMediaTime() - profileStart) * 1000.0
        chaperoneProfileAccumMs += elapsedMs
        chaperoneProfileSamples += 1
        chaperoneProfileWindow.append(elapsedMs)
        if chaperoneProfileWindow.count > chaperoneProfileWindowSize {
            chaperoneProfileWindow.removeFirst(chaperoneProfileWindow.count - chaperoneProfileWindowSize)
        }
        let now = CACurrentMediaTime()
        if now - chaperoneProfileLastLog > 2.0 {
            chaperoneProfileLastLog = now
            let avg = chaperoneProfileAccumMs / Double(max(1, chaperoneProfileSamples))
            let maxSample = chaperoneProfileWindow.max() ?? 0.0
            var median: Double = 0.0
            if !chaperoneProfileWindow.isEmpty {
                let sorted = chaperoneProfileWindow.sorted()
                median = sorted[sorted.count / 2]
            }
            let recomputeSamples = max(1, output.profile.recomputeSamples)
            let recomputePct = Int((Double(output.profile.recomputeTrue) / Double(recomputeSamples)) * 100.0)
            let avgRenderables = Double(output.profile.renderableCount) / Double(recomputeSamples)
            print("[ChaperoneProfile] avg=\(String(format: "%.3f", avg))ms median=\(String(format: "%.3f", median))ms max=\(String(format: "%.3f", maxSample))ms samples=\(chaperoneProfileSamples) planes=\(planesSnapshot.count) recompute=\(recomputePct)% exact=\(output.profile.exactProximityChecks) renderables=\(String(format: "%.1f", avgRenderables)) interval=\(output.profile.recomputeIntervalFrames)")
            chaperoneProfileAccumMs = 0.0
            chaperoneProfileSamples = 0
        }
#endif

        var firstBind = true
        for (plane, planeColor) in output.renderables {
            let faces = plane.geometry.meshFaces

            // VRR can't do lines
            if faces.primitive != GeometryElement.Primitive.triangle {
                continue
            }

            renderEncoder.setVertexBuffer(plane.geometry.meshVertices.buffer, offset: 0, index: VertexAttribute.position.rawValue)
            renderEncoder.setVertexBuffer(plane.geometry.meshVertices.buffer, offset: 0, index: VertexAttribute.texcoord.rawValue)

            selectNextPlaneUniformBuffer()
            self.planeUniforms[0].planeTransform = plane.originFromAnchorTransform
            self.planeUniforms[0].planeColor = planeColor
            self.planeUniforms[0].planeDoProximity = 1.0
            if firstBind {
                renderEncoder.setVertexBuffer(dynamicPlaneUniformBuffer, offset:planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue)
                firstBind = false
            } else {
                renderEncoder.setVertexBufferOffset(planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue)
            }
            
            renderEncoder.setTriangleFillMode(.fill)
            renderEncoder.drawIndexedPrimitives(type: faces.primitive == .triangle ? MTLPrimitiveType.triangle : MTLPrimitiveType.line,
                                                indexCount: faces.count*3,
                                                indexType: faces.bytesPerIndex == 2 ? MTLIndexType.uint16 : MTLIndexType.uint32,
                                                indexBuffer: faces.buffer,
                                                indexBufferOffset: 0)
        }

        renderEncoder.popDebugGroup()
        renderEncoder.endEncoding()
    }

    func renderPerformanceHud(commandBuffer: MTLCommandBuffer, renderTargetColor: MTLTexture, renderTargetDepth: MTLTexture, viewports: [MTLViewport], rasterizationRateMap: MTLRasterizationRateMap?, simdDeviceAnchor: simd_float4x4) {
        guard ALVRClientApp.gStore.settings.showPerformanceHud else { return }
        if performanceHudRenderer == nil {
            performanceHudRenderer = PerformanceHudRenderer(device: device)
        }
        guard let hudTexture = performanceHudRenderer?.updateIfNeeded() else { return }
        selectNextPlaneUniformBuffer()
        let worldFromSteamVR = WorldTracker.shared.worldTrackingSteamVRTransform
        let handPose = WorldTracker.shared.lastLeftHandPose
        let handWorld = worldFromSteamVR * handPose.asFloat4x4()
        let handPosition = handWorld.columns.3.asFloat3()
        var handRight = simd_normalize(handWorld.columns.0.asFloat3())
        let handUp = simd_normalize(handWorld.columns.1.asFloat3())
        let handForward = simd_normalize(handWorld.columns.2.asFloat3())
        if simd_length(handRight) < 0.001 || simd_length(handUp) < 0.001 || simd_length(handForward) < 0.001 {
            return
        }
        handRight = simd_normalize(handRight)
        // Hand axes derived from the current left-hand pose.
        // thumbAxis: toward thumb side of the hand (local +X).
        // palmAxis: palm normal (local +Y).
        // elbowAxis: from fingers toward wrist/elbow (local -Z).
        let thumbAxis = -handForward
        let palmAxis = handRight
        let elbowAxis = handUp

        // Panel axes in world space:
        // panelFront: the face normal of the panel (should align with palmAxis).
        // panelTopEdge: the top edge of the panel (should align with elbowAxis).
        // panelRightEdge: the right edge of the panel (perpendicular to both).
        let panelFront = simd_normalize(-palmAxis)
        var panelTopEdge = simd_normalize(-elbowAxis)
        var panelRightEdge = simd_normalize(-thumbAxis)
        let palmTwist = simd_quatf(angle: Float.pi / 180.0 * 20.0, axis: simd_normalize(palmAxis))
        panelTopEdge = simd_normalize(palmTwist.act(panelTopEdge))
        panelRightEdge = simd_normalize(palmTwist.act(panelRightEdge))
        let forearmDir = panelTopEdge

        // Orientation uses panelRightEdge (X), panelTopEdge (Y), panelFront (Z).
        let orientation = simd_float4x4(simd_float4(panelRightEdge, 0.0),
                                        simd_float4(panelTopEdge, 0.0),
                                        simd_float4(panelFront, 0.0),
                                        simd_float4(0.0, 0.0, 0.0, 1.0))
        // Position offsets:
        // offsetIn: shift toward the inner forearm (away from thumb).
        // offsetArm: shift toward elbow along the forearm.
        // offsetDown: move slightly into the arm to sit on the surface.
        let offsetIn = simd_float3()
        let offsetArm = -forearmDir * 0.10
        let offsetDown = panelFront * 0.025
        let offset = offsetIn + offsetArm + offsetDown
        let position = handPosition + offset
        let translation = position.asFloat4x4()
        let sizeY: Float = 0.07
        let aspect = Float(hudTexture.width) / Float(hudTexture.height)
        let sizeX: Float = sizeY * aspect
        let scale = simd_float4x4(simd_float4(sizeX, 0.0, 0.0, 0.0),
                                  simd_float4(0.0, sizeY, 0.0, 0.0),
                                  simd_float4(0.0, 0.0, 1.0, 0.0),
                                  simd_float4(0.0, 0.0, 0.0, 1.0))
        let transform = translation * orientation * scale

        self.planeUniforms[0].planeTransform = transform
        self.planeUniforms[0].planeColor = simd_float4(1.0, 1.0, 1.0, 1.0)
        self.planeUniforms[0].planeDoProximity = 0.0

        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = renderTargetColor
        renderPassDescriptor.colorAttachments[0].loadAction = .load
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.depthAttachment.texture = renderTargetDepth
        renderPassDescriptor.depthAttachment.loadAction = .load
        renderPassDescriptor.depthAttachment.storeAction = .dontCare
        renderPassDescriptor.depthAttachment.clearDepth = 0.0
        renderPassDescriptor.rasterizationRateMap = rasterizationRateMap
        renderPassDescriptor.renderTargetArrayLength = viewports.count

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create HUD render encoder")
        }

        renderEncoder.label = "HUD Render Encoder"
        renderEncoder.setCullMode(.back)
        renderEncoder.setFrontFacing(.counterClockwise)
        renderEncoder.setViewports(viewports)
        renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        renderEncoder.setVertexBuffer(dynamicPlaneUniformBuffer, offset: planeUniformBufferOffset, index: BufferIndex.planeUniforms.rawValue)

        if viewports.count > 1 {
            var viewMappings = (0..<viewports.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }

        renderEncoder.setRenderPipelineState(hudPipelineState)
        renderEncoder.setDepthStencilState(depthStateAlwaysNoWrite)
#if !targetEnvironment(simulator)
        renderEncoder.setDepthClipMode(.clamp)
#endif
        renderEncoder.setVertexBuffer(hudQuadBuffer, offset: 0, index: VertexAttribute.position.rawValue)
        renderEncoder.setVertexBuffer(hudQuadBuffer, offset: (3 * 4) * 4, index: VertexAttribute.texcoord.rawValue)
        renderEncoder.setFragmentTexture(hudTexture, index: TextureIndex.color.rawValue)
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderEncoder.endEncoding()
    }
    
    // Sets up rendering a video frame, including uniforms
    func beginRenderStreamingFrame(_ whichIdx: Int, commandBuffer: MTLCommandBuffer, renderTargetColor: MTLTexture, renderTargetDepth: MTLTexture, viewports: [MTLViewport], viewTransforms: [simd_float4x4], sentViewTangents: [simd_float4], realViewTangents: [simd_float4], nearZ: Double, farZ: Double, rasterizationRateMap: MTLRasterizationRateMap?, queuedFrame: QueuedFrame?, framePose: simd_float4x4, simdDeviceAnchor: simd_float4x4, drawable: LayerRenderer.Drawable?) -> (any MTLRenderCommandEncoder)? {
        if currentRenderColorFormat != renderTargetColor.pixelFormat && isRealityKit {
            return nil
        }
        
        // TODO refactor this
        if isRealityKit {
            fadeInOverlayAlpha -= 0.01
            if fadeInOverlayAlpha < 0.0 {
                fadeInOverlayAlpha = 0.0
            }
        }
    
        self.updateDynamicBufferState()
        
        self.updateGameStateForVideoFrame(whichIdx, drawable: drawable, viewTransforms: viewTransforms, sentViewTangents: sentViewTangents, realViewTangents: realViewTangents, nearZ: nearZ, farZ: farZ, framePose: framePose, simdDeviceAnchor: simdDeviceAnchor)
        
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = renderTargetColor
        renderPassDescriptor.colorAttachments[0].loadAction = whichIdx == 0 ? (isRealityKit ? .dontCare : .clear) : .load
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: chromaKeyEnabled ? 0.0 : 1.0)
        renderPassDescriptor.rasterizationRateMap = rasterizationRateMap
        
        renderPassDescriptor.renderTargetArrayLength = viewports.count
        
        /// Final pass rendering code here
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render encoder")
        }
        
        renderEncoder.label = "Primary Render Encoder"
        renderEncoder.pushDebugGroup("Draw ALVR Frames")
        
        guard let queuedFrame = queuedFrame else {
            renderEncoder.endEncoding()
            return nil
        }
        
        // https://cs.android.com/android/platform/superproject/main/+/main:external/webrtc/sdk/objc/components/renderer/metal/RTCMTLNV12Renderer.mm;l=108;drc=a81e9c82fc3fbc984f0f110407d1e44c9c01958a
        let pixelBuffer = queuedFrame.imageBuffer
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let formatStr = VideoHandler.coreVideoPixelFormatToStr[format, default: "unknown"]
        // PyroWave frames are converted by the shader with the transform from their format
        // description, not by the private YCbCr texture formats, whose conversion depends on
        // color metadata VideoToolbox attaches to its own surfaces.
        let allowSecret = !PyroWaveDecoder.isPyroWaveFrame(pixelBuffer)
        
        if VideoHandler.isFormatSecret(format, allowSecret: allowSecret) {
            renderEncoder.setRenderPipelineState(videoFramePipelineState_SecretYpCbCrFormats)
        }
        else {
            renderEncoder.setRenderPipelineState(videoFramePipelineState_YpCbCrBiPlanar)
        }
        
        //print("Pixel format \(formatStr) (\(format))")
        let textureTypes = VideoHandler.getTextureTypesForFormat(format, allowSecret: allowSecret)
        
        for i in 0...1 {
            var textureOut:CVMetalTexture! = nil
            var err:OSStatus = 0
            let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, i)
            let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, i)
            
            if textureTypes[i] == MTLPixelFormat.invalid {
                break
            }
            
            err = CVMetalTextureCacheCreateTextureFromImage(
                    nil, metalTextureCache, pixelBuffer, nil, textureTypes[i],
                    width, height, i, &textureOut);
            
            if err != 0 {
                fatalError("CVMetalTextureCacheCreateTextureFromImage \(err)")
            }
            guard let metalTexture = CVMetalTextureGetTexture(textureOut) else {
                fatalError("CVMetalTextureGetTexture")
            }
            if !((metalTexture.debugDescription?.contains("decompressedPixelFormat") ?? true) || (metalTexture.debugDescription?.contains("isCompressed = 1") ?? true)) && EventHandler.shared.totalFramesRendered % 90*5 == 0 {
                print("NO COMPRESSION ON VT FRAME!!!! AAAAAAAAA go file feedback again :(")
            }
            renderEncoder.setFragmentTexture(metalTexture, index: i)
        }
        
        // Snoop for pixel formats
        /*for idx in 620..<0xFFFF {
            guard let format = MTLPixelFormat.init(rawValue: UInt(idx)) else {
                continue
            }
            do {
                var desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: MTLPixelFormat.invalid, width: 1, height:1, mipmapped: false)
                desc.pixelFormat = format
                for line in desc.debugDescription.split(separator: "\n") {
                    if line.contains("pixelFormat") {
                        print(idx, line)
                        break
                    }
                }
            }
            catch {
                continue
            }
        }*/
        
        renderEncoder.setCullMode(.none)
        renderEncoder.setFrontFacing(.counterClockwise)
#if !targetEnvironment(simulator)
        renderEncoder.setDepthClipMode(.clamp)
#endif
        
        renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset:uniformBufferOffset, index: BufferIndex.uniforms.rawValue)

        return renderEncoder
    }
    
    // Actually do the video frame render.
    func renderStreamingFrame(_ whichIdx: Int, commandBuffer: MTLCommandBuffer, renderEncoder: any MTLRenderCommandEncoder, renderTargetColor: MTLTexture, renderTargetDepth: MTLTexture, viewports: [MTLViewport], viewTransforms: [simd_float4x4], sentViewTangents: [simd_float4], realViewTangents: [simd_float4], nearZ: Double, farZ: Double, rasterizationRateMap: MTLRasterizationRateMap?, framePose: simd_float4x4, simdDeviceAnchor: simd_float4x4) {
        if currentRenderColorFormat != renderTargetColor.pixelFormat && isRealityKit {
            return
        }
        
        renderEncoder.setViewports(viewports)
        
        if viewports.count > 1 {
            var viewMappings = (0..<viewports.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }
        
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: whichIdx*4, vertexCount: 4)
    }
    
    // Finish video frame encoding.
    func endRenderStreamingFrame(renderEncoder: any MTLRenderCommandEncoder) {
        renderEncoder.popDebugGroup()
        renderEncoder.endEncoding()
    }
    
    // Render an overlay on top of the video frame.
    func renderStreamingFrameOverlays(_ whichIdx: Int, commandBuffer: MTLCommandBuffer, renderTargetColor: MTLTexture, renderTargetDepth: MTLTexture, viewports: [MTLViewport], viewTransforms: [simd_float4x4], sentViewTangents: [simd_float4], realViewTangents: [simd_float4], nearZ: Double, farZ: Double, rasterizationRateMap: MTLRasterizationRateMap?, queuedFrame: QueuedFrame?, framePose: simd_float4x4, simdDeviceAnchor: simd_float4x4, drawable: LayerRenderer.Drawable?) {
        if currentRenderColorFormat != renderTargetColor.pixelFormat && isRealityKit {
            return
        }
    
        self.updateDynamicBufferState()
        
        self.updateGameStateForVideoFrame(whichIdx, drawable: drawable, viewTransforms: viewTransforms, sentViewTangents: sentViewTangents, realViewTangents: realViewTangents, nearZ: nearZ, farZ: farZ, framePose: framePose, simdDeviceAnchor: simdDeviceAnchor)
        
        if fadeInOverlayAlpha > 0.0 || WorldTracker.shared.debuggableMats.count > 0 || WorldTracker.shared.debuggableMeshes.count > 0 {
            // Not super kosher--we need the depth to be correct for the video frame box, but we can't have the view
            // outside of the video frame box be 0.0 depth or it won't get rastered by the compositor at all.
            // So we re-render the frame depth.
            renderOverlay(commandBuffer: commandBuffer, renderTargetColor: renderTargetColor, renderTargetDepth: renderTargetDepth, viewports: viewports, viewTransforms: viewTransforms, sentViewTangents: sentViewTangents, realViewTangents: realViewTangents, nearZ: nearZ, farZ: farZ, rasterizationRateMap: rasterizationRateMap, queuedFrame: queuedFrame, framePose: framePose, simdDeviceAnchor: simdDeviceAnchor)
        }
        if ALVRClientApp.gStore.settings.chaperoneDistanceCm > 0 || chaperoneSystem.hasActiveState {
            renderChaperone(commandBuffer: commandBuffer, renderTargetColor: renderTargetColor, renderTargetDepth: renderTargetDepth, viewports: viewports, rasterizationRateMap: rasterizationRateMap, simdDeviceAnchor: simdDeviceAnchor)
        }
        if !isRealityKit {
            renderStreamingFrameDepth(commandBuffer: commandBuffer, renderTargetColor: renderTargetColor, renderTargetDepth: renderTargetDepth, viewports: viewports, viewTransforms: viewTransforms, sentViewTangents: sentViewTangents, realViewTangents: realViewTangents, nearZ: nearZ, farZ: farZ, rasterizationRateMap: rasterizationRateMap, queuedFrame: queuedFrame)
        }
        if !isRealityKit && ALVRClientApp.gStore.settings.showPerformanceHud {
            renderPerformanceHud(commandBuffer: commandBuffer, renderTargetColor: renderTargetColor, renderTargetDepth: renderTargetDepth, viewports: viewports, rasterizationRateMap: rasterizationRateMap, simdDeviceAnchor: simdDeviceAnchor)
        }
    }
}


// Frame-timing diagnostic (2026-09-13). Where do the ~23 ms of ALVR's "Client System"
// (vsync_queue = presentationTime - our GPU work done) go? visionOS hands every frame three
// times: optimalInputTime (start input + rendering), renderingDeadline (our frame must be done)
// and presentationTime (photons). The renderer waits until optimalInputTime, picks the newest
// video frame and draws it. If it is done long before renderingDeadline, picking the video
// frame later would show a fresher one -- this measures how much slack there actually is before
// anything changes. Averages over 450 frames (~5 s at 90 Hz) go to the console (stdout), prefixed "frame timing".
// Measurement only: nothing here changes when frames are picked or presented.
final class FrameTimingDiag {
    static let enabled = true
    static let shared = FrameTimingDiag()

    private static let names = [
        "deadline-optimal", "present-deadline", "wake-optimal", "pick-wake", "commit-pick",
        "done-commit", "gpu exec", "slack deadline-done", "present-done (vsync_queue)",
    ]
    private let lock = NSLock()
    private var count = 0
    private var sums = [Double](repeating: 0, count: 9)
    private var mins = [Double](repeating: .greatestFiniteMagnitude, count: 9)
    private var maxs = [Double](repeating: -.greatestFiniteMagnitude, count: 9)
    private var missedDeadline = 0
    private var noNewFrame = 0

    func record(optimal: Double, deadline: Double, present: Double, wake: Double, pick: Double,
                commit: Double, done: Double, gpuStart: Double, gpuEnd: Double, newFrame: Bool) {
        let values = [
            deadline - optimal, present - deadline, wake - optimal, pick - wake, commit - pick,
            done - commit, gpuEnd - gpuStart, deadline - done, present - done,
        ]
        var line: String? = nil
        lock.lock()
        count += 1
        for i in 0..<values.count {
            sums[i] += values[i]
            mins[i] = min(mins[i], values[i])
            maxs[i] = max(maxs[i], values[i])
        }
        if done > deadline { missedDeadline += 1 }
        if !newFrame { noNewFrame += 1 }
        if count >= 450 {
            var parts: [String] = []
            for i in 0..<values.count {
                parts.append(String(format: "%@ %.2f (%.2f..%.2f)", Self.names[i],
                                    sums[i] / Double(count) * 1000.0, mins[i] * 1000.0, maxs[i] * 1000.0))
            }
            line = "frame timing (ms, avg (min..max) over \(count) frames): " + parts.joined(separator: " | ")
                + " | GPU done after deadline: \(missedDeadline) | frames without a new video frame: \(noNewFrame)"
            count = 0
            sums = [Double](repeating: 0, count: 9)
            mins = [Double](repeating: .greatestFiniteMagnitude, count: 9)
            maxs = [Double](repeating: -.greatestFiniteMagnitude, count: 9)
            missedDeadline = 0
            noNewFrame = 0
        }
        lock.unlock()
        if let line {
            pyroLog(line)
        }
    }
}


// Late frame pickup (2026-09-13), see GlobalSettings.lateFramePickup.
//
// Measured on this renderer with FrameTimingDiag: visionOS gives exactly 11.11 ms from
// optimalInputTime to renderingDeadline and a fixed 17.79 ms from the deadline to photons. The
// renderer picked its video frame right at optimalInputTime and was done ~4.5 ms before the
// deadline, so the frame shown was ~4.5 ms older than it needed to be (and a frame arriving in
// that window waited a whole vsync). With the setting on, the pick happens at
// renderingDeadline - budget:
//   budget = longest pick-to-GPU-done time of the last 90 frames + margin
// The margin starts at 1 ms, grows by 0.5 ms on every missed deadline (capped at 4 ms) and decays
// back slowly, so a load spike widens it at once and calm frames give the time back gradually.
// Until 30 frames have been measured the budget exceeds the whole render window: no extra wait.
final class LateFramePickup {
    static let shared = LateFramePickup()

    private static let window = 90
    private static let baseMargin = 0.001
    private static let maxMargin = 0.004
    private static let missStep = 0.0005
    private static let decayPerFrame = 0.000002
    private let lock = NSLock()
    private var samples = [Double](repeating: 0, count: window)
    private var filled = 0
    private var next = 0
    private var margin = baseMargin
    private var logCount = 0
    private var budgetSum = 0.0
    private var missed = 0

    // The budget decides how early the video frame is picked: pickAt = renderingDeadline -
    // budget, so every millisecond of budget is a millisecond of extra age in the picture.
    //
    // Building it from the MAXIMUM of the last 90 frames means one slow frame raises it for a
    // whole second, and every frame in that second is picked that much earlier. Measured on
    // device (2026-09-18): budget 8.3-12.8 ms against an 11.1 ms frame interval, while deadlines
    // were missed in under 2 % of frames -- so the reserve is far larger than the risk it covers.
    //
    // usePercentileBudget switches to the 95th percentile instead. The margin above it already
    // self-corrects: it grows by missStep on every missed deadline and decays slowly otherwise.
    static let usePercentileBudget = false
    private static let budgetPercentile = 0.95

    func budgetSeconds() -> Double {
        lock.lock()
        defer { lock.unlock() }
        if filled < 30 {
            return 0.0112
        }
        return workBudgetLocked() + margin
    }

    // Caller holds the lock.
    private func workBudgetLocked() -> Double {
        guard filled > 0 else { return 0.0 }
        if !Self.usePercentileBudget {
            return samples[0..<filled].max() ?? 0.0
        }
        let sorted = samples[0..<filled].sorted()
        let idx = min(sorted.count - 1, Int((Double(sorted.count) * Self.budgetPercentile).rounded(.down)))
        return sorted[idx]
    }

    // Diagnostics only: what the work distribution actually looks like, so the choice above can
    // be judged from numbers instead of assumption.
    private func workStatsLocked() -> (p50: Double, p95: Double, max: Double) {
        guard filled > 0 else { return (0, 0, 0) }
        let sorted = samples[0..<filled].sorted()
        let i50 = min(sorted.count - 1, sorted.count / 2)
        let i95 = min(sorted.count - 1, Int(Double(sorted.count) * 0.95))
        return (sorted[i50], sorted[i95], sorted[sorted.count - 1])
    }

    func record(work: Double, missedDeadline: Bool, active: Bool) {
        var line: String? = nil
        lock.lock()
        samples[next] = work
        next = (next + 1) % Self.window
        filled = min(filled + 1, Self.window)
        if missedDeadline {
            margin = min(margin + Self.missStep, Self.maxMargin)
        } else {
            margin = max(margin - Self.decayPerFrame, Self.baseMargin)
        }
        if active {
            logCount += 1
            budgetSum += workBudgetLocked() + margin
            if missedDeadline { missed += 1 }
            if logCount >= 450 {
                // p50/p95/max of the measured work: the gap between p95 and max is exactly what
                // a percentile budget would give back as picture freshness, and "missed" is the
                // price side of that trade.
                let s = workStatsLocked()
                line = String(format: "late frame pickup: budget avg %.2f ms (%@), margin now %.2f ms, work p50 %.2f p95 %.2f max %.2f ms, p95 budget would be %.2f ms, missed deadlines %ld of %ld",
                              budgetSum / Double(logCount) * 1000.0,
                              Self.usePercentileBudget ? "p95" : "max",
                              margin * 1000.0,
                              s.p50 * 1000.0, s.p95 * 1000.0, s.max * 1000.0,
                              (s.p95 + margin) * 1000.0,
                              missed, logCount)
                logCount = 0
                budgetSum = 0.0
                missed = 0
            }
        }
        lock.unlock()
        if let line {
            pyroLog(line)
        }
    }
}


// Tracking send phase (2026-09-27), see GlobalSettings.trackingSendPhase.
//
// The headset sends one pose per display cycle, and the total latency counts from that sample.
// A decoded frame then waits for the renderer's pickup ("frame buffering", 4-8 ms p50 measured
// with PyroWave): the whole chain from pose to decoded frame finished that much too early. The
// streamer cannot help (starting the game later just moves the wait into game time; it still
// renders the same pose), but the headset can: sample and send the pose that much later. Then
// the frame lands just before the pickup, from a fresher pose. The streamer's "Phase-lock frame
// pacing" follows the tracking arrival by itself.
//
// The pose goes out at optimalInputTime + phase, from this render thread (DeviceAnchor queries
// are not moved to another thread), while it waits anyway: before optimalInputTime, or before the
// late pickup. If the time falls into the frame's own work, it goes out right after. The target
// timestamp stays the presentation time of the frame the pose was planned in.
// NB: phase 0 is NOT where the pose went without this setting when Late Frame Pickup is on: that
// was after the pickup and the frame's command encoding, 3-6 ms after optimalInputTime.
//
// The phase is circular: "earlier than 0" is just before the next optimalInputTime (the pose of
// frame k then goes out at the end of its cycle), so the controller can move either way from
// wherever the session starts.
//
// It minimises the expected wait directly. Sending delta later turns every measured wait q into
// (q - delta) mod period: shorter, unless the frame then misses the pickup and waits almost a whole
// period. The total latency changes by exactly that difference (pose to pickup = chain + wait), so
// over the last 90 fresh frames it tries shifts of -3..+3 ms and takes the one with the smallest
// mean wait (a frame counts as missing when it would land within 0.25 ms of the pickup), moving at
// most 0.5 ms per window and only for a gain of more than 0.15 ms; when the gain exceeds 0.5 ms
// (after a load change) it searches the whole circle and steps up to 2 ms, then skips 30 frames
// while the streamer's vsync follows. Measured 2026-09-27:
// - run 6 (without): waits p50 0.3, p95 10.9 ms: frames right on the pickup, half missing it;
// - run 7: p50 0.9, p95 2.0 ms, total 39.4/39.5 ms, the best so far;
// - run 10, earlier rule-based version: left that point on 2.2% late frames (worth 0.24 ms against
//   a 0.5 ms step), and under heavy game load held at p50 3.5 ms because the waits spread 2.8 ms,
//   where moving on would have cost more in misses than it saved;
// - run 12 (this version with 0.5 ms steps and a +-3 ms search): under load changes phase + chain
//   length (game + encoding) stayed at 25.2 ms mod the period in every settled window, i.e. the phase
//   followed the load exactly, but took 10-15 s after each change.
final class TrackingSendPhase {
    static let shared = TrackingSendPhase()

    // Plain values, not the Timing itself: the frame's timing is only guaranteed during the frame,
    // and a pose may go out during the next frame's wait.
    private struct Pending {
        let due: Double
        let optimal: Double
        var vsyncTime: Double
        var anchorTimestamp: Double
    }

    private static let window = 90
    private static let maxStep = 0.0005
    // A clear gain (after a load change) allows a bigger step, towards 3/4 of the best shift.
    private static let bigGain = 0.0005
    private static let maxBigStep = 0.002
    // After a big step, while the streamer's vsync follows the new tracking arrival.
    private static let settleFrames = 30
    private static let searchStep = 0.00025
    // A frame landing closer than this to the pickup is counted as missing it.
    private static let safetyMargin = 0.00025
    private static let minGain = 0.00015
    // For the log only: arrived this soon after a pickup.
    private static let lateMargin = 0.002

    private var pending: Pending? = nil
    private var phase = 0.0
    private var period = 1.0 / 90.0
    private var lastOptimal = 0.0
    private var samples: [Double] = []
    private var settle = 0
    private var windows = 0
    private var sentOffsetSum = 0.0
    private var sentCount = 0
    private var directions = [String: Int]()

    private static func seconds(_ instant: LayerRenderer.Clock.Instant) -> Double {
        LayerRenderer.Clock.Instant.epoch.duration(to: instant).timeInterval
    }

    private static func instant(_ seconds: Double) -> LayerRenderer.Clock.Instant {
        LayerRenderer.Clock.Instant.epoch.advanced(by: .nanoseconds(Int64(seconds * 1e9)))
    }

    // Plans this frame's pose. A pose still pending from the last frame goes out first.
    func plan(timing: LayerRenderer.Frame.Timing) {
        if pending != nil {
            send()
        }
        let optimal = Self.seconds(timing.optimalInputTime)
        let interval = optimal - lastOptimal
        // Up to 50 ms: visionOS can hand the app only every second display frame (22.22 ms at
        // 90 Hz); the period must follow so the plan holds instead of steering on the wrong cycle.
        if lastOptimal > 0 && interval > 0.006 && interval < 0.05 {
            period += (interval - period) * 0.05
        }
        lastOptimal = optimal
        phase = wrapped(phase)
        pending = Pending(due: optimal + phase, optimal: optimal,
                          vsyncTime: Self.seconds(timing.presentationTime),
                          anchorTimestamp: Self.anchorTimestamp(timing))
    }

    private func wrapped(_ value: Double) -> Double {
        var result = value.truncatingRemainder(dividingBy: period)
        if result < 0 {
            result += period
        }
        return result
    }

    // What the render loop sends as the anchor timestamp without this setting.
    private static func anchorTimestamp(_ timing: LayerRenderer.Frame.Timing) -> Double {
        if ALVRClientApp.gStore.settings.targetHandsAtRoundtripLatency {
            return seconds(timing.presentationTime)
        }
        if #available(visionOS 2.0, *) {
            return seconds(timing.trackableAnchorTime)
        }
        return seconds(timing.renderingDeadline)
    }

    // Waits until the instant, sending the pending pose on the way if it is due before it.
    func wait(until end: LayerRenderer.Clock.Instant) {
        if let due = pending?.due, due < Self.seconds(end) {
            if due > CACurrentMediaTime() {
                LayerRenderer.Clock().wait(until: Self.instant(due))
            }
            send()
        }
        LayerRenderer.Clock().wait(until: end)
    }

    // At the old send point, after the frame's own work: takes the drawable's timing (what the
    // render loop sends without this setting) and sends if the time has come.
    func sendIfDue(drawableTiming: LayerRenderer.Frame.Timing) {
        guard pending != nil else {
            return
        }
        pending!.vsyncTime = Self.seconds(drawableTiming.presentationTime)
        pending!.anchorTimestamp = Self.anchorTimestamp(drawableTiming)
        if pending!.due <= CACurrentMediaTime() {
            send()
        }
    }

    func cancel() {
        pending = nil
        samples.removeAll()
        settle = 0
    }

    private func send() {
        guard let pending = pending else {
            return
        }
        self.pending = nil
        guard EventHandler.shared.alvrInitialized && EventHandler.shared.lastIpd != -1 else {
            return
        }
        sentOffsetSum += CACurrentMediaTime() - pending.optimal
        sentCount += 1
        _ = WorldTracker.shared.sendTracking(viewTransforms: EventHandler.shared.viewTransforms, viewFovs: EventHandler.shared.viewFovs, targetTimestamp: pending.vsyncTime, reportedTargetTimestamp: pending.vsyncTime, anchorTimestamp: pending.anchorTimestamp, delay: 0.0)
    }

    // Pickup time minus the time the frame was decoded and queued, for frames shown new.
    func recordFrameBuffering(_ wait: Double) {
        if settle > 0 {
            settle -= 1
            return
        }
        samples.append(wait)
        if samples.count < Self.window {
            return
        }
        let sorted = samples.sorted()
        samples.removeAll(keepingCapacity: true)
        let shortWait = sorted[sorted.count / 10]
        let median = sorted[sorted.count / 2]
        let late = Double(sorted.filter { $0 > period - Self.lateMargin }.count) / Double(sorted.count)

        var direction = "hold"
        var expectedNow = 0.0
        var expectedBest = 0.0
        // With the display at another rate than the stream (visionOS switched to 100 Hz under a 90 fps
        // stream in some runs), frames drift through the whole display cycle: the waits spread over a
        // full period whatever the phase. Nothing to steer then.
        let streamPeriod = DisplayRateWatch.streamPeriod()
        if abs(period - streamPeriod) > DisplayRateWatch.tolerance {
            direction = String(format: "hold (display %.0f Hz, stream %.0f fps)", 1.0 / period, 1.0 / streamPeriod)
        }
        else {
            // Mean wait if the pose went out `delta` later.
            let expectedWait = { (delta: Double) -> Double in
                var sum = 0.0
                for q in sorted {
                    var shifted = (q - delta - Self.safetyMargin).truncatingRemainder(dividingBy: self.period)
                    if shifted < 0 {
                        shifted += self.period
                    }
                    sum += shifted + Self.safetyMargin
                }
                return sum / Double(sorted.count)
            }
            expectedNow = expectedWait(0.0)
            expectedBest = expectedNow
            var bestDelta = 0.0
            // The whole circle: after a load change the best shift can be anywhere.
            let searchRange = period / 2
            var delta = -searchRange
            while delta <= searchRange + 1e-9 {
                let expected = expectedWait(delta)
                if expected < expectedBest {
                    expectedBest = expected
                    bestDelta = delta
                }
                delta += Self.searchStep
            }
            let gain = expectedNow - expectedBest
            if gain > Self.minGain {
                var step = min(max(bestDelta, -Self.maxStep), Self.maxStep)
                if gain > Self.bigGain && abs(bestDelta) > Self.maxStep {
                    step = min(max(bestDelta * 0.75, -Self.maxBigStep), Self.maxBigStep)
                    settle = Self.settleFrames
                }
                phase = wrapped(phase + step)
                direction = abs(step) > Self.maxStep ? (step > 0 ? "later (big)" : "earlier (big)") : (step > 0 ? "later" : "earlier")
            }
            else {
                expectedBest = expectedNow
            }
        }
        directions[direction, default: 0] += 1

        windows += 1
        if windows >= 5 {
            let steps = directions.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", ")
            pyroLog(String(format: "tracking send phase: %.2f ms after optimal input (sent %.2f ms after it on average), frame buffering p10 %.2f p50 %.2f ms, late %.1f%%, expected wait %.2f ms (best shift %.2f ms), period %.2f ms; last %d windows: %@",
                           phase * 1000.0, sentCount > 0 ? sentOffsetSum / Double(sentCount) * 1000.0 : 0.0,
                           shortWait * 1000.0, median * 1000.0, late * 100.0, expectedNow * 1000.0, expectedBest * 1000.0,
                           period * 1000.0, windows, steps))
            windows = 0
            sentOffsetSum = 0.0
            sentCount = 0
            directions.removeAll()
        }
    }
}


// Frame rate watch. Two things change how often visionOS hands this app a frame, and both look
// the same on the streamer (fewer tracking samples, fewer frames shown):
//  - the display rate: visionOS runs the display at 100 Hz when the passthrough cameras detect
//    50 Hz flicker from artificial light ("Passthrough_50Hz_Flicker_Detected"), whatever the app
//    asks for. Under a 90 fps stream frames then drift through the display cycle, so neither
//    pickup phase control can work (Tracking Send Phase holds while the rates differ).
//  - the app's share of it: visionOS can hand the app only every second display frame (22.22 ms
//    at 90 Hz, 20.00 ms at 100 Hz) and reproject in between. Seen in run 32 (2026-09-30) for
//    minutes at a time while our own deadlines were met; the cause is not known (GPU load,
//    thermal state?), so the log line carries the thermal state and the video filter.
// Measured from successive optimalInputTimes. A rate is reported once it has held for
// stableFrames frames in a row, so transitions and single late frames do not produce lines.
final class DisplayRateWatch {
    static let shared = DisplayRateWatch()
    static let tolerance = 0.0003

    private static let stableFrames = 90
    private static let runTolerance = 0.0005

    private var lastOptimal = 0.0
    private var runPeriod = 0.0
    private var runLength = 0
    private var reportedPeriod = 0.0
    private var streaming = false
    private var thermalObserver: NSObjectProtocol?

    // The stream's frame period: the streamer's refresh rate hint, else the display preference.
    static func streamPeriod() -> Double {
        let hint = Double(EventHandler.shared.streamEvent?.STREAMING_STARTED.refresh_rate_hint ?? 0)
        let rate = hint > 0 ? hint : Double(refreshRate)
        return rate > 0 ? 1.0 / rate : 1.0 / 90.0
    }

    static func thermalStateName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private init() {
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil
        ) { _ in
            pyroLog("Thermal state: \(DisplayRateWatch.thermalStateName())")
        }
    }

    // What a frame period means: the display at the preferred rate or at 100 Hz, and the app at
    // every display frame or every second one.
    private static func describe(_ period: Double, wanted: Double) -> String {
        let near = { (a: Double, b: Double) in abs(a - b) < 2 * runTolerance }
        if near(period, wanted) {
            return "every display frame at the preferred rate"
        }
        if near(period, 0.01) {
            return "every display frame, display at 100 Hz (visionOS detected 50 Hz flicker from artificial light; stream at 100 fps to match)"
        }
        if near(period, 2 * wanted) {
            return String(format: "every SECOND display frame at %.0f Hz: visionOS throttles the app and reprojects in between", 1.0 / wanted)
        }
        if near(period, 0.02) {
            return "every SECOND display frame, display at 100 Hz (flicker compensation) and the app throttled on top"
        }
        return "no known pattern"
    }

    func observe(optimalInputTime: LayerRenderer.Clock.Instant) {
        let optimal = LayerRenderer.Clock.Instant.epoch.duration(to: optimalInputTime).timeInterval
        let interval = optimal - lastOptimal
        lastOptimal = optimal
        let wanted = Double(refreshRate) > 0 ? 1.0 / Double(refreshRate) : Self.streamPeriod()
        guard EventHandler.shared.streamingActive else {
            // A new stream starts from the preferred rate, so a matching rate stays silent.
            reportedPeriod = wanted
            runLength = 0
            streaming = false
            return
        }
        if !streaming {
            streaming = true
            pyroLog("Thermal state at stream start: \(Self.thermalStateName())")
        }
        guard interval > 0.006 && interval < 0.05 else {
            runLength = 0
            return
        }
        if runLength > 0 && abs(interval - runPeriod) < Self.runTolerance {
            runLength += 1
            runPeriod += (interval - runPeriod) * 0.1
        }
        else {
            runPeriod = interval
            runLength = 1
        }
        guard runLength == Self.stableFrames, abs(runPeriod - reportedPeriod) > Self.runTolerance else {
            return
        }
        reportedPeriod = runPeriod
        pyroLog(String(format: "Frame rate: visionOS hands the app a frame every %.2f ms (%.1f Hz), ", runPeriod * 1000.0, 1.0 / runPeriod)
            + Self.describe(runPeriod, wanted: wanted)
            + String(format: " | preference %.0f Hz, stream %.0f fps", Double(refreshRate), 1.0 / Self.streamPeriod())
            + " | thermal \(Self.thermalStateName()), video filter \(ALVRClientApp.gStore.settings.videoFilter)")
    }
}


// Rate map and pixel density diagnostics (2026-09-30). The drawable's rasterization rate map says
// where visionOS renders at full density; with the view tangents that gives the pixels per degree
// at the view center, both for the drawable (what the headset can show) and for the stream (what
// the streamer sends, its full-resolution FFE center). Logged at stream start and whenever the
// drawable's size changes:
//   "Drawable: screen WxH ..."           sizes and granularity of the rate map
//   "Drawable eye N: FOV ... | PPD ..."  center pixels per degree and the full-rate region
//   "Stream: ... PPD ..."                the same for the stream, and stream/drawable
// Every 10 s while streaming, "Rate map: ..." says whether the map changed (sampled 10 times a
// second). A map that follows the gaze would show changes and a moving full-rate center; that
// would be the data for gaze-driven foveated encoding.
final class RateMapDiag {
    static let shared = RateMapDiag()

    private struct Axis {
        var fullStart = 0.0   // screen fraction where the full-rate span starts
        var fullEnd = 0.0
        var highShare = 0.0   // share of the screen at >= 85 % of the peak rate
        var minRate = 1.0
        var center: Double { (fullStart + fullEnd) / 2 }
    }
    private struct Eye {
        var x = Axis()
        var y = Axis()
        var physical = MTLSize(width: 0, height: 0, depth: 0)
    }

    private static let sampleEvery = 9          // frames, ~10 Hz at 90 Hz
    private static let logInterval = 10.0       // seconds

    private var reported = false
    private var lastScreen = MTLSize(width: 0, height: 0, depth: 0)
    private var lastSignature: [Double] = []
    private var frames = 0
    private var samples = 0
    private var changes = 0
    private var centerRange: [(minX: Double, maxX: Double, minY: Double, maxY: Double)] = []
    private var lastLog = 0.0

    func observe(drawable: LayerRenderer.Drawable) {
        guard EventHandler.shared.streamingActive, EventHandler.shared.lastIpd != -1,
              let vrr = drawable.rasterizationRateMaps.first else {
            reported = false
            return
        }
        frames += 1
        let screen = vrr.screenSize
        let changedSize = screen.width != lastScreen.width || screen.height != lastScreen.height
        guard !reported || changedSize || frames % Self.sampleEvery == 0 else {
            return
        }

        let eyes = (0..<min(2, vrr.layerCount)).map { Self.eye(vrr, layer: $0) }
        if !reported || changedSize {
            reported = true
            lastScreen = screen
            report(drawable: drawable, vrr: vrr, eyes: eyes)
            lastSignature = []
            resetWindow(eyes: eyes)
        }

        // What would move if the map followed the gaze: the full-rate span of each axis, per eye,
        // and the shape of the falloff around it.
        let signature = eyes.flatMap { [$0.x.fullStart, $0.x.fullEnd, $0.y.fullStart, $0.y.fullEnd,
                                        $0.x.highShare, $0.y.highShare, $0.x.minRate, $0.y.minRate,
                                        Double($0.physical.width), Double($0.physical.height)] }
            .map { ($0 * 1000).rounded() / 1000 }
        if !lastSignature.isEmpty && signature != lastSignature {
            changes += 1
        }
        lastSignature = signature
        samples += 1
        for (i, eye) in eyes.enumerated() where i < centerRange.count {
            centerRange[i].minX = min(centerRange[i].minX, eye.x.center)
            centerRange[i].maxX = max(centerRange[i].maxX, eye.x.center)
            centerRange[i].minY = min(centerRange[i].minY, eye.y.center)
            centerRange[i].maxY = max(centerRange[i].maxY, eye.y.center)
        }

        let now = CACurrentMediaTime()
        if now - lastLog >= Self.logInterval {
            let centers = centerRange.enumerated().map { (i, r) in
                String(format: "eye %ld center x %.1f-%.1f %% y %.1f-%.1f %%", i,
                       r.minX * 100, r.maxX * 100, r.minY * 100, r.maxY * 100)
            }.joined(separator: " | ")
            pyroLog("Rate map: \(changes) changes in \(samples) samples over \(Int(Self.logInterval)) s" +
                    (changes == 0 ? " (static)" : " (changing)") + " | " + centers)
            resetWindow(eyes: eyes)
        }
    }

    private func resetWindow(eyes: [Eye]) {
        lastLog = CACurrentMediaTime()
        samples = 0
        changes = 0
        centerRange = eyes.map { (minX: $0.x.center, maxX: $0.x.center, minY: $0.y.center, maxY: $0.y.center) }
    }

    // The map is separable per layer, so one row and one column describe it.
    private static func eye(_ vrr: MTLRasterizationRateMap, layer: Int) -> Eye {
        let granularity = vrr.physicalGranularity
        let physical = vrr.physicalSize(layer: layer)
        func axis(horizontal: Bool) -> Axis {
            let cellSize = horizontal ? granularity.width : granularity.height
            let cells = (horizontal ? physical.width : physical.height) / max(cellSize, 1)
            let screenSize = Double(horizontal ? vrr.screenSize.width : vrr.screenSize.height)
            var spans: [(start: Double, end: Double, rate: Double)] = []
            for j in 0..<cells {
                let p0 = Float(j * cellSize), p1 = Float((j + 1) * cellSize)
                let s0 = vrr.screenCoordinates(physicalCoordinates: MTLCoordinate2D(x: horizontal ? p0 : 0, y: horizontal ? 0 : p0), layer: layer)
                let s1 = vrr.screenCoordinates(physicalCoordinates: MTLCoordinate2D(x: horizontal ? p1 : 0, y: horizontal ? 0 : p1), layer: layer)
                let a = Double(horizontal ? s0.x : s0.y), b = Double(horizontal ? s1.x : s1.y)
                if b > a {
                    spans.append((a / screenSize, b / screenSize, Double(cellSize) / (b - a)))
                }
            }
            var result = Axis()
            guard let peak = spans.map({ $0.rate }).max(), peak > 0 else { return result }
            let full = spans.filter { $0.rate >= 0.99 * peak }
            result.fullStart = full.first?.start ?? 0
            result.fullEnd = full.last?.end ?? 0
            result.highShare = spans.filter { $0.rate >= 0.85 * peak }.reduce(0) { $0 + ($1.end - $1.start) }
            result.minRate = (spans.map { $0.rate }.min() ?? peak) / peak
            return result
        }
        return Eye(x: axis(horizontal: true), y: axis(horizontal: false), physical: physical)
    }

    // Pixels per degree at the view center of a rectilinear image `pixels` wide over the tangents.
    private static func centerPPD(pixels: Int, tan0: Float, tan1: Float) -> Double {
        let span = Double(tan0 + tan1)
        return span > 0 ? Double(pixels) / span * Double.pi / 180 : 0
    }

    private static func degrees(_ t: Float) -> Double { Double(atan(t)) * 180 / Double.pi }

    private func report(drawable: LayerRenderer.Drawable, vrr: MTLRasterizationRateMap, eyes: [Eye]) {
        let screen = vrr.screenSize
        let physical = eyes.map { "\($0.physical.width)x\($0.physical.height)" }.joined(separator: " / ")
        pyroLog("Drawable: screen \(screen.width)x\(screen.height), physical \(physical), granularity \(vrr.physicalGranularity.width)x\(vrr.physicalGranularity.height), \(drawable.views.count) views, \(drawable.rasterizationRateMaps.count) rate maps")

        let real = EventHandler.shared.realViewTangents
        let sent = EventHandler.shared.sentViewTangents
        var drawablePPD: [Double] = []
        for (i, eye) in eyes.enumerated() where i < real.count {
            let t = real[i]  // left, right, up, down
            let ppdX = Self.centerPPD(pixels: screen.width, tan0: t.x, tan1: t.y)
            let ppdY = Self.centerPPD(pixels: screen.height, tan0: t.z, tan1: t.w)
            drawablePPD.append(ppdX)
            pyroLog(String(format: "Drawable eye %ld: FOV left %.1f right %.1f up %.1f down %.1f deg | center PPD %.1f x %.1f | full rate x %.1f %% at %.1f %%, y %.1f %% at %.1f %% | >=85%% x %.1f %%, y %.1f %% | edge rate x 1/%.1f, y 1/%.1f",
                           i, Self.degrees(t.x), Self.degrees(t.y), Self.degrees(t.z), Self.degrees(t.w), ppdX, ppdY,
                           (eye.x.fullEnd - eye.x.fullStart) * 100, eye.x.center * 100,
                           (eye.y.fullEnd - eye.y.fullStart) * 100, eye.y.center * 100,
                           eye.x.highShare * 100, eye.y.highShare * 100,
                           eye.x.minRate > 0 ? 1 / eye.x.minRate : 0, eye.y.minRate > 0 ? 1 / eye.y.minRate : 0))
        }

        if let started = EventHandler.shared.streamEvent?.STREAMING_STARTED {
            let width = Int(started.view_width), height = Int(started.view_height)
            for (i, t) in sent.enumerated() where i < 2 {
                let ppdX = Self.centerPPD(pixels: width, tan0: t.x, tan1: t.y)
                let ppdY = Self.centerPPD(pixels: height, tan0: t.z, tan1: t.w)
                let ratio = i < drawablePPD.count && drawablePPD[i] > 0 ? ppdX / drawablePPD[i] : 0
                pyroLog(String(format: "Stream eye %ld: %ldx%ld over FOV left %.1f right %.1f up %.1f down %.1f deg | center PPD %.1f x %.1f (stream/drawable %.2f)",
                               i, width, height, Self.degrees(t.x), Self.degrees(t.y), Self.degrees(t.z), Self.degrees(t.w), ppdX, ppdY, ratio))
            }
        }
    }
}
