//
//  PyroWaveDecoder.swift
//
//  Decodes PyroWave (https://github.com/Themaister/pyrowave) frames on the GPU with the
//  vendored Metal port in PyroWave/, packet by packet as they arrive.
//
//  The streamer sends every frame as many small packets, in stripe order (see
//  PyroWave/pyrowave_stripes.hpp): once the packets of the first k stripes are in, the decoder
//  has every coefficient it needs for the top rows of the picture. So each packet is parsed on
//  the network thread the moment it arrives, and each time more stripes are complete, a serial
//  worker encodes the dequantization and inverse wavelet work for exactly the rows those stripes
//  make computable. When the last packet arrives only the last stripe is left to decode.
//
//  The rest of the client renders VideoToolbox output, i.e. biplanar YCbCr CVPixelBuffers.
//  Luma is decoded straight into the luma plane of an IOSurface-backed 10-bit biplanar full
//  range pixel buffer; Cb and Cr are decoded into private planes and interleaved into its
//  chroma plane stripe by stripe. The finished buffer goes through the same frame queue,
//  reprojection and foveation paths as a VideoToolbox frame.
//
//  Packet loss: a missing packet stops the stripe pipeline at that point (later stripes may
//  depend on what it carried); the frame is then finished when its last packet or the next
//  frame arrives, with the missing blocks decoded as zero, which only blurs them.
//

import CoreMedia
import CoreVideo
import Foundation
import Metal
import QuartzCore

// The PyroWave, renderer timing and connection lines of the app, also written to
// Documents/pyrowave_debug.log (the previous run's in pyrowave_debug.prev.log), since stdout is
// only visible with a console attached. Copy it off the headset with:
//   xcrun devicectl device copy from --device <UDID> --domain-type appDataContainer \
//     --domain-identifier <bundle id> --source Documents/pyrowave_debug.log --destination .
func pyroLog(_ message: String) {
    print(message)
    PyroWaveLogFile.shared.append(message)
}

final class PyroWaveLogFile {
    static let shared = PyroWaveLogFile()

    // One background queue, one open handle: a line every few seconds must not cost the render
    // or network thread a file open and close.
    private let queue = DispatchQueue(label: "PyroWaveLogFile", qos: .utility)
    private var handle: FileHandle?
    private var bytesWritten = 0
    private static let maxBytes = 32 << 20
    private let formatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter
    }()

    private init() {
        queue.async {
            guard let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
                return
            }
            let url = documents.appendingPathComponent("pyrowave_debug.log")
            let previous = documents.appendingPathComponent("pyrowave_debug.prev.log")
            try? FileManager.default.removeItem(at: previous)
            try? FileManager.default.moveItem(at: url, to: previous)
            FileManager.default.createFile(atPath: url.path, contents: nil)
            self.handle = try? FileHandle(forWritingTo: url)
        }
    }

    func append(_ message: String) {
        let now = Date()
        queue.async {
            guard let handle = self.handle, self.bytesWritten < Self.maxBytes else {
                return
            }
            let line = "\(self.formatter.string(from: now)) \(message)\n"
            if let data = line.data(using: .utf8) {
                handle.write(data)
                self.bytesWritten += data.count
            }
        }
    }
}

final class PyroWaveDecoder {
    // PyroWaveStripes::StreamConfig, sent by the streamer as the decoder config.
    struct StreamConfig {
        static let magic: UInt32 = 0x5752_5950 // "PYRW"
        static let version: UInt32 = 2
        static let size = 32

        let width: Int
        let height: Int
        let chroma444: Bool
        let stripeHeight: Int
        let maxFrameBytes: Int

        init?(bytes: UnsafeBufferPointer<UInt8>) {
            guard bytes.count >= StreamConfig.size else {
                pyroLog("PyroWave: decoder config is \(bytes.count) bytes, expected \(StreamConfig.size)")
                return nil
            }
            let raw = UnsafeRawBufferPointer(bytes)
            func u32(_ index: Int) -> UInt32 {
                return UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: 4 * index, as: UInt32.self))
            }
            guard u32(0) == StreamConfig.magic, u32(1) == StreamConfig.version else {
                pyroLog("PyroWave: decoder config has magic \(u32(0)) version \(u32(1)), expected \(StreamConfig.magic) version \(StreamConfig.version). Streamer and client are from different versions.")
                return nil
            }
            // Only BT.709 full range SDR (color 0) exists so far.
            guard u32(5) == 0 else {
                pyroLog("PyroWave: unsupported color mode \(u32(5))")
                return nil
            }
            width = Int(u32(2))
            height = Int(u32(3))
            chroma444 = u32(4) == 1
            stripeHeight = Int(u32(6))
            maxFrameBytes = Int(u32(7))
            guard width > 0, height > 0, chroma444 || (width % 2 == 0 && height % 2 == 0),
                  stripeHeight > 0, stripeHeight % 32 == 0, maxFrameBytes > 0 else {
                pyroLog("PyroWave: invalid stream config \(width)x\(height), stripes of \(stripeHeight) rows, \(maxFrameBytes) bytes")
                return nil
            }
        }
    }

    // PyroWaveStripes::PacketPrefix, at the start of every video packet. The PyroWave data
    // (dataBytes long) follows; anything after it is zero padding.
    struct PacketPrefix {
        static let size = 16

        let stripe: Int
        let stripeCount: Int
        let index: Int
        let count: Int
        let dataBytes: Int

        init?(_ packet: UnsafeRawBufferPointer) {
            guard packet.count > PacketPrefix.size else {
                return nil
            }
            stripe = Int(UInt16(littleEndian: packet.loadUnaligned(fromByteOffset: 0, as: UInt16.self)))
            stripeCount = Int(UInt16(littleEndian: packet.loadUnaligned(fromByteOffset: 2, as: UInt16.self)))
            index = Int(UInt32(littleEndian: packet.loadUnaligned(fromByteOffset: 4, as: UInt32.self)))
            count = Int(UInt32(littleEndian: packet.loadUnaligned(fromByteOffset: 8, as: UInt32.self)))
            dataBytes = Int(UInt32(littleEndian: packet.loadUnaligned(fromByteOffset: 12, as: UInt32.self)))
            guard count > 0, index < count, stripe < stripeCount, dataBytes <= packet.count - PacketPrefix.size else {
                return nil
            }
        }
    }

    // One frame being received and decoded. The receive-side fields are only touched on the
    // network thread, the decode-side fields only on the worker queue.
    private final class Frame {
        let timestamp: UInt64
        let handle: pyrowave_frame
        let pixelBuffer: CVPixelBuffer
        // The CVMetalTextures must live as long as GPU work writes through them.
        let lumaTexture: CVMetalTexture
        let chromaTexture: CVMetalTexture
        let luma: MTLTexture
        let chroma: MTLTexture
        let packetCount: Int
        let firstPacketTime: CFTimeInterval

        // Receive side.
        var stripeOfPacket: [UInt16]
        var received: [Bool]
        var receivedCount = 0
        var corruptPackets = 0
        // Packets [0, contiguous) have all arrived.
        var contiguous = 0
        var stripesSubmitted = 0
        var lastPacketTime: CFTimeInterval = 0

        // Decode side.
        var chromaRowsInterleaved = 0

        init(timestamp: UInt64, handle: pyrowave_frame, pixelBuffer: CVPixelBuffer, lumaTexture: CVMetalTexture,
             chromaTexture: CVMetalTexture, luma: MTLTexture, chroma: MTLTexture, packetCount: Int) {
            self.timestamp = timestamp
            self.handle = handle
            self.pixelBuffer = pixelBuffer
            self.lumaTexture = lumaTexture
            self.chromaTexture = chromaTexture
            self.luma = luma
            self.chroma = chroma
            self.packetCount = packetCount
            firstPacketTime = CACurrentMediaTime()
            stripeOfPacket = [UInt16](repeating: 0, count: packetCount)
            received = [Bool](repeating: false, count: packetCount)
        }
    }

    // Tags the pixel buffers this decoder produces, so the renderer can tell them apart from
    // VideoToolbox output.
    private static let frameAttachmentKey = "ALVRPyroWaveFrame" as CFString

    // Upper bound on pixel buffers in use at once: the frame queue, the frame being rendered,
    // and the frames being received. Past it a frame is dropped instead of allocating more.
    private static let maxPixelBuffers = 10

    // A frame that lost more than this share of its packets is dropped instead of shown.
    private static let minimumReceivedRatio = 0.9

    let config: StreamConfig
    // Describes the frames for the renderer's YCbCr to RGB transform (EventHandler.videoFormat).
    let formatDescription: CMFormatDescription

    private let onFrameDecoded: (CVPixelBuffer, UInt64, CFTimeInterval) -> Void
    private let mtlDevice: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let worker = DispatchQueue(label: "PyroWave decode", qos: .userInteractive)
    private let interleavePipeline: MTLComputePipelineState
    private let textureCache: CVMetalTextureCache
    private let pixelBufferPool: CVPixelBufferPool
    private let chromaWidth: Int
    private let chromaHeight: Int
    private let cbTexture: MTLTexture
    private let crTexture: MTLTexture
    private var device: pyrowave_device?
    private var decoder: pyrowave_decoder?
    private var stripeCount = 0

    // Network thread.
    private var current: Frame?
    private var newestFinishedTimestamp: UInt64 = 0

    // Statistics, from the network thread and from Metal's completion threads.
    private let statsLock = NSLock()
    private var statFramesShown = 0
    private var statFramesDropped = 0
    private var statFramesIncomplete = 0
    private var statPacketsLost = 0
    private var statTailMs = 0.0
    private var statTailMaxMs = 0.0
    private var statReceiveMs = 0.0
    private var lastReportTime = CACurrentMediaTime()

    static func isSupported() -> Bool {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return false
        }
        return pyrowave_device_is_supported(Unmanaged.passUnretained(device as AnyObject).toOpaque())
    }

    static func isPyroWaveFrame(_ pixelBuffer: CVPixelBuffer) -> Bool {
        return CVBufferCopyAttachment(pixelBuffer, frameAttachmentKey, nil) != nil
    }

    // onFrameDecoded(pixelBuffer, timestamp, time of the frame's first packet) runs on a Metal
    // completion thread once a frame is fully decoded.
    init?(configBytes: UnsafeBufferPointer<UInt8>, onFrameDecoded: @escaping (CVPixelBuffer, UInt64, CFTimeInterval) -> Void) {
        guard let config = StreamConfig(bytes: configBytes) else {
            return nil
        }
        self.config = config
        self.onFrameDecoded = onFrameDecoded
        chromaWidth = config.chroma444 ? config.width : config.width / 2
        chromaHeight = config.chroma444 ? config.height : config.height / 2

        guard let mtlDevice = MTLCreateSystemDefaultDevice(),
              let commandQueue = mtlDevice.makeCommandQueue(),
              let library = mtlDevice.makeDefaultLibrary(),
              let interleaveFunction = library.makeFunction(name: "pyrowaveInterleaveChroma"),
              let interleavePipeline = try? mtlDevice.makeComputePipelineState(function: interleaveFunction)
        else {
            pyroLog("PyroWave: Metal setup failed")
            return nil
        }
        self.mtlDevice = mtlDevice
        self.commandQueue = commandQueue
        commandQueue.label = "PyroWave decode"
        self.interleavePipeline = interleavePipeline

        var textureCache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(nil, nil, mtlDevice, nil, &textureCache) == kCVReturnSuccess,
              let textureCache else {
            pyroLog("PyroWave: CVMetalTextureCacheCreate failed")
            return nil
        }
        self.textureCache = textureCache

        // 10 bits of the 16 bit containers are what CoreVideo defines for this format; the
        // decoder writes full 16 bit values and the renderer samples them as r16Unorm/rg16Unorm.
        let pixelFormat = config.chroma444
            ? kCVPixelFormatType_444YpCbCr10BiPlanarFullRange
            : kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
        let pixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
            kCVPixelBufferWidthKey as String: config.width,
            kCVPixelBufferHeightKey as String: config.height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ]
        let poolAttributes: [String: Any] = [kCVPixelBufferPoolMinimumBufferCountKey as String: 4]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(nil, poolAttributes as CFDictionary, pixelBufferAttributes as CFDictionary, &pool) == kCVReturnSuccess,
              let pool else {
            pyroLog("PyroWave: CVPixelBufferPoolCreate failed")
            return nil
        }
        pixelBufferPool = pool

        let chromaDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Unorm, width: chromaWidth, height: chromaHeight, mipmapped: false)
        chromaDescriptor.usage = [.shaderRead, .shaderWrite]
        chromaDescriptor.storageMode = .private
        guard let cbTexture = mtlDevice.makeTexture(descriptor: chromaDescriptor),
              let crTexture = mtlDevice.makeTexture(descriptor: chromaDescriptor) else {
            pyroLog("PyroWave: chroma plane allocation failed")
            return nil
        }
        cbTexture.label = "PyroWave Cb"
        crTexture.label = "PyroWave Cr"
        self.cbTexture = cbTexture
        self.crTexture = crTexture

        let extensions: [CFString: Any] = [
            kCMFormatDescriptionExtension_YCbCrMatrix: kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2,
            kCMFormatDescriptionExtension_ColorPrimaries: kCMFormatDescriptionColorPrimaries_ITU_R_709_2,
            kCMFormatDescriptionExtension_TransferFunction: kCMFormatDescriptionTransferFunction_ITU_R_709_2,
            kCMFormatDescriptionExtension_FullRangeVideo: true,
            kCMFormatDescriptionExtension_BitsPerComponent: 10,
        ]
        var formatDescription: CMFormatDescription?
        guard CMVideoFormatDescriptionCreate(allocator: nil, codecType: pixelFormat,
                                             width: Int32(config.width), height: Int32(config.height),
                                             extensions: extensions as CFDictionary,
                                             formatDescriptionOut: &formatDescription) == noErr,
              let formatDescription else {
            pyroLog("PyroWave: CMVideoFormatDescriptionCreate failed")
            return nil
        }
        self.formatDescription = formatDescription

        // Compiles the decode pipelines, so this takes a moment; it happens once per stream.
        var deviceInfo = pyrowave_device_create_info(
            mtl_device: Unmanaged.passUnretained(mtlDevice as AnyObject).toOpaque(),
            message_callback: { _, message in
                if let message {
                    pyroLog("PyroWave: \(String(cString: message))")
                }
            },
            message_userdata: nil)
        var result = pyrowave_device_create(&deviceInfo, &device)
        guard result == PYROWAVE_SUCCESS else {
            pyroLog("PyroWave: device creation failed: \(String(cString: pyrowave_result_to_string(result)))")
            return nil
        }

        var decoderInfo = pyrowave_decoder_create_info(
            device: device,
            width: Int32(config.width),
            height: Int32(config.height),
            chroma: config.chroma444 ? PYROWAVE_CHROMA_SUBSAMPLING_444 : PYROWAVE_CHROMA_SUBSAMPLING_420)
        result = pyrowave_decoder_create(&decoderInfo, &decoder)
        guard result == PYROWAVE_SUCCESS else {
            pyroLog("PyroWave: decoder creation failed: \(String(cString: pyrowave_result_to_string(result)))")
            return nil
        }

        result = pyrowave_decoder_enable_progressive(decoder, Int32(config.stripeHeight), config.maxFrameBytes)
        guard result == PYROWAVE_SUCCESS else {
            pyroLog("PyroWave: progressive decoding unavailable: \(String(cString: pyrowave_result_to_string(result)))")
            return nil
        }
        stripeCount = Int(pyrowave_decoder_stripe_count(decoder))

        pyroLog("PyroWave: decoder ready, \(config.width)x\(config.height) \(config.chroma444 ? "4:4:4" : "4:2:0"), \(stripeCount) stripes of \(config.stripeHeight) rows")
    }

    deinit {
        // Queued decode steps and command buffer handlers hold strong references to this
        // decoder, so it only goes away once they are done. PyroWave also waits for the GPU
        // before it frees anything.
        if let decoder {
            pyrowave_decoder_destroy(decoder)
        }
        if let device {
            pyrowave_device_destroy(device)
        }
    }

    // Called on the network receive thread for every video packet of the stream.
    func receive(packet: UnsafeRawBufferPointer, timestamp: UInt64) {
        guard let prefix = PacketPrefix(packet) else {
            pyroLog("PyroWave: dropping a malformed packet (\(packet.count) bytes)")
            return
        }

        if let frame = current, frame.timestamp != timestamp {
            if timestamp < frame.timestamp {
                // A straggler of a frame that was already given up on.
                return
            }
            // The next frame started before this one completed: some of its packets were lost.
            finish(frame)
        }
        if timestamp <= newestFinishedTimestamp {
            return
        }

        if current == nil {
            guard prefix.stripeCount == stripeCount else {
                pyroLog("PyroWave: packet says \(prefix.stripeCount) stripes, the decoder has \(stripeCount). Streamer and client disagree on the stream config.")
                newestFinishedTimestamp = timestamp
                return
            }
            guard let frame = beginFrame(timestamp: timestamp, packetCount: prefix.count) else {
                noteDropped()
                // Ignore the rest of this frame's packets.
                newestFinishedTimestamp = timestamp
                return
            }
            current = frame
        }
        guard let frame = current else {
            return
        }
        guard prefix.count == frame.packetCount, !frame.received[prefix.index] else {
            return
        }

        let data = UnsafeRawBufferPointer(rebasing: packet[PacketPrefix.size..<(PacketPrefix.size + prefix.dataBytes)])
        if pyrowave_frame_push_packet(frame.handle, data.baseAddress, data.count) != PYROWAVE_SUCCESS {
            frame.corruptPackets += 1
        }
        frame.received[prefix.index] = true
        frame.stripeOfPacket[prefix.index] = UInt16(prefix.stripe)
        frame.receivedCount += 1
        frame.lastPacketTime = CACurrentMediaTime()

        while frame.contiguous < frame.packetCount && frame.received[frame.contiguous] {
            frame.contiguous += 1
        }

        if frame.receivedCount == frame.packetCount {
            finish(frame)
            return
        }

        // Packets go out in stripe order, so every stripe before the one of the last contiguous
        // packet is complete (that stripe itself may still have packets on the way).
        let completeStripes = frame.contiguous == 0 ? 0 : Int(frame.stripeOfPacket[frame.contiguous - 1])
        if completeStripes > frame.stripesSubmitted {
            frame.stripesSubmitted = completeStripes
            worker.async { [self] in
                decodeStep(frame, completeStripes: completeStripes, final: false)
            }
        }
    }

    private func beginFrame(timestamp: UInt64, packetCount: Int) -> Frame? {
        guard let decoder else {
            return nil
        }
        var handle: pyrowave_frame?
        guard pyrowave_decoder_progressive_begin(decoder, &handle) == PYROWAVE_SUCCESS, let handle else {
            pyroLog("PyroWave: all frames are still decoding, dropping one")
            return nil
        }

        var pixelBufferOut: CVPixelBuffer?
        let auxAttributes = [kCVPixelBufferPoolAllocationThresholdKey as String: PyroWaveDecoder.maxPixelBuffers] as CFDictionary
        guard CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(nil, pixelBufferPool, auxAttributes, &pixelBufferOut) == kCVReturnSuccess,
              let pixelBuffer = pixelBufferOut else {
            pyroLog("PyroWave: no free pixel buffer, dropping a frame")
            pyrowave_frame_end(handle)
            return nil
        }
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(pixelBuffer, PyroWaveDecoder.frameAttachmentKey, kCFBooleanTrue, .shouldNotPropagate)

        guard let lumaTexture = makeWritableTexture(pixelBuffer, plane: 0, format: .r16Unorm),
              let chromaTexture = makeWritableTexture(pixelBuffer, plane: 1, format: .rg16Unorm),
              let luma = CVMetalTextureGetTexture(lumaTexture),
              let chroma = CVMetalTextureGetTexture(chromaTexture) else {
            pyrowave_frame_end(handle)
            return nil
        }

        return Frame(timestamp: timestamp, handle: handle, pixelBuffer: pixelBuffer, lumaTexture: lumaTexture,
                     chromaTexture: chromaTexture, luma: luma, chroma: chroma, packetCount: packetCount)
    }

    // Network thread: no more packets will be taken for this frame.
    private func finish(_ frame: Frame) {
        if current === frame {
            current = nil
        }
        newestFinishedTimestamp = max(newestFinishedTimestamp, frame.timestamp)

        let lost = frame.packetCount - frame.receivedCount
        let usable = Double(frame.receivedCount) >= PyroWaveDecoder.minimumReceivedRatio * Double(frame.packetCount)
        if lost > 0 || frame.corruptPackets > 0 {
            statsLock.withLock {
                statFramesIncomplete += 1
                statPacketsLost += lost
            }
        }

        worker.async { [self] in
            if usable {
                decodeStep(frame, completeStripes: stripeCount, final: true)
            } else {
                // Earlier steps may still be running on the GPU; the frame pool waits for them
                // before it reuses this frame.
                pyrowave_frame_end(frame.handle)
                noteDropped()
            }
        }
        reportIfDue()
    }

    // Worker queue.
    private func decodeStep(_ frame: Frame, completeStripes: Int, final: Bool) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            if final {
                pyrowave_frame_end(frame.handle)
                noteDropped()
            }
            return
        }
        commandBuffer.label = final ? "PyroWave last stripes" : "PyroWave stripes"

        var planes = pyrowave_gpu_buffers(planes: (
            Unmanaged.passUnretained(frame.luma as AnyObject).toOpaque(),
            Unmanaged.passUnretained(cbTexture as AnyObject).toOpaque(),
            Unmanaged.passUnretained(crTexture as AnyObject).toOpaque()))
        var progress = pyrowave_progress()
        let result = pyrowave_frame_decode(frame.handle, Unmanaged.passUnretained(commandBuffer as AnyObject).toOpaque(),
                                           &planes, Int32(completeStripes), final, &progress)
        if result != PYROWAVE_SUCCESS {
            pyroLog("PyroWave: decode step failed: \(String(cString: pyrowave_result_to_string(result)))")
        }

        // Interleave the chroma rows this step finished into the pixel buffer's chroma plane.
        let chromaRows = Int(min(progress.plane_rows.1, progress.plane_rows.2))
        if result == PYROWAVE_SUCCESS, chromaRows > frame.chromaRowsInterleaved,
           let encoder = commandBuffer.makeComputeCommandEncoder() {
            var firstRow = UInt32(frame.chromaRowsInterleaved)
            encoder.label = "PyroWave interleave chroma"
            encoder.setComputePipelineState(interleavePipeline)
            encoder.setTexture(cbTexture, index: 0)
            encoder.setTexture(crTexture, index: 1)
            encoder.setTexture(frame.chroma, index: 2)
            encoder.setBytes(&firstRow, length: MemoryLayout<UInt32>.size, index: 0)
            let threadWidth = interleavePipeline.threadExecutionWidth
            let threadHeight = max(1, interleavePipeline.maxTotalThreadsPerThreadgroup / threadWidth)
            encoder.dispatchThreads(MTLSize(width: chromaWidth, height: chromaRows - frame.chromaRowsInterleaved, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: threadWidth, height: threadHeight, depth: 1))
            encoder.endEncoding()
            frame.chromaRowsInterleaved = chromaRows
        }

        if !final {
            // The frame (its pixel buffer above all) must outlive every command buffer that
            // writes into it, also when it is dropped before its last step.
            commandBuffer.addCompletedHandler { _ in
                withExtendedLifetime(frame) {}
            }
        } else {
            let lastPacketTime = frame.lastPacketTime
            let ok = result == PYROWAVE_SUCCESS
            commandBuffer.addCompletedHandler { [self] buffer in
                if ok && buffer.status == .completed {
                    let now = CACurrentMediaTime()
                    statsLock.withLock {
                        statFramesShown += 1
                        let tailMs = (now - lastPacketTime) * 1000.0
                        statTailMs += tailMs
                        statTailMaxMs = max(statTailMaxMs, tailMs)
                        statReceiveMs += (lastPacketTime - frame.firstPacketTime) * 1000.0
                    }
                    onFrameDecoded(frame.pixelBuffer, frame.timestamp, frame.firstPacketTime)
                } else {
                    pyroLog("PyroWave: GPU decode failed: \(String(describing: buffer.error))")
                    noteDropped()
                }
            }
        }
        commandBuffer.commit()
        if final {
            pyrowave_frame_end(frame.handle)
        }
    }

    private func makeWritableTexture(_ pixelBuffer: CVPixelBuffer, plane: Int, format: MTLPixelFormat) -> CVMetalTexture? {
        let attributes = [kCVMetalTextureUsage as String: MTLTextureUsage([.shaderRead, .shaderWrite]).rawValue] as CFDictionary
        var texture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, pixelBuffer, attributes, format,
            CVPixelBufferGetWidthOfPlane(pixelBuffer, plane),
            CVPixelBufferGetHeightOfPlane(pixelBuffer, plane),
            plane, &texture)
        if status != kCVReturnSuccess {
            pyroLog("PyroWave: CVMetalTextureCacheCreateTextureFromImage(plane \(plane)) failed: \(status)")
            return nil
        }
        return texture
    }

    private func noteDropped() {
        statsLock.withLock {
            statFramesDropped += 1
        }
    }

    // Network thread.
    private func reportIfDue() {
        let now = CACurrentMediaTime()
        guard now - lastReportTime >= 5.0 else {
            return
        }
        statsLock.withLock {
            let shown = max(statFramesShown, 1)
            pyroLog(String(format: "PyroWave: %d frames shown, %d dropped, %d incomplete (%d packets lost) in %.1f s; receive %.2f ms, last packet to decoded %.2f ms avg %.2f ms max",
                         statFramesShown, statFramesDropped, statFramesIncomplete, statPacketsLost, now - lastReportTime,
                         statReceiveMs / Double(shown), statTailMs / Double(shown), statTailMaxMs))
            statFramesShown = 0
            statFramesDropped = 0
            statFramesIncomplete = 0
            statPacketsLost = 0
            statTailMs = 0
            statTailMaxMs = 0
            statReceiveMs = 0
        }
        lastReportTime = now
    }
}
