// PyroWave decoder bench (Mac, Apple silicon): GPU time of decoding one frame at the stream's
// size with the GPU otherwise idle, for the upstream whole-frame path and for the ALVR stripe
// progressive path split into a given number of steps, one command buffer per step as in
// PyroWaveDecoder.swift.
//
//   decbench <source.bin> [width] [height] [bpp] [iterations]
//
// source.bin: 8 bit luma of any picture, read as <srcW>x<srcH> from the env SRC_W/SRC_H (default
// 2432x1728, one eye of a StreamRecorder frame) and tiled (mirrored) to width x height. Content
// matters little at a fixed rate: the rate control fills the budget either way.
// PYROWAVE_PRECISION=0/1/2 selects the decoder precision as upstream documents it.

#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>
#include <cstring>
#include "pyrowave_metal.h"
#include "pyrowave_stripes.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void *opaque(id object) { return (__bridge void *)object; }

static id<MTLTexture> plane(id<MTLDevice> device, int w, int h) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Unorm
                                                                                 width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModePrivate;
    return [device newTextureWithDescriptor:d];
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: decbench <source.bin> [width] [height] [bpp] [iterations]\n");
            return 1;
        }
        const int W = argc > 2 ? atoi(argv[2]) : 7040;
        const int H = argc > 3 ? atoi(argv[3]) : 2496;
        const double bpp = argc > 4 ? atof(argv[4]) : 2.0;
        const int iterations = argc > 5 ? atoi(argv[5]) : 30;
        const int srcW = getenv("SRC_W") ? atoi(getenv("SRC_W")) : 2432;
        const int srcH = getenv("SRC_H") ? atoi(getenv("SRC_H")) : 1728;

        std::vector<uint8_t> src(size_t(srcW) * srcH);
        FILE *f = fopen(argv[1], "rb");
        if (!f || fread(src.data(), 1, src.size(), f) != src.size()) {
            fprintf(stderr, "cannot read %d x %d bytes from %s\n", srcW, srcH, argv[1]);
            return 1;
        }
        fclose(f);
        auto mirror = [](int v, int n) { int p = v % (2 * n); return p < n ? p : 2 * n - 1 - p; };
        const int CW = W / 2, CH = H / 2;
        std::vector<uint8_t> y(size_t(W) * H), cb(size_t(CW) * CH), cr(size_t(CW) * CH);
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                y[size_t(r) * W + c] = src[size_t(mirror(r, srcH)) * srcW + mirror(c, srcW)];
        // Some chroma structure from the luma, so the chroma bands are not empty.
        for (int r = 0; r < CH; r++)
            for (int c = 0; c < CW; c++) {
                const int v = y[size_t(2 * r) * W + 2 * c];
                cb[size_t(r) * CW + c] = uint8_t(128 + (v - 128) / 4);
                cr[size_t(r) * CW + c] = uint8_t(128 - (v - 128) / 5);
            }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        printf("device %s, %dx%d 4:2:0, %.2f bpp, %d iterations, PYROWAVE_PRECISION=%s\n",
               device.name.UTF8String, W, H, bpp, iterations, getenv("PYROWAVE_PRECISION") ?: "default");

        pyrowave_device_create_info deviceInfo = {};
        deviceInfo.mtl_device = opaque(device);
        pyrowave_device pwDevice = nullptr;
        if (pyrowave_device_create(&deviceInfo, &pwDevice) != PYROWAVE_SUCCESS) return 1;

        pyrowave_encoder_create_info encInfo = {};
        encInfo.device = pwDevice;
        encInfo.width = W;
        encInfo.height = H;
        encInfo.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
        pyrowave_encoder encoder = nullptr;
        if (pyrowave_encoder_create(&encInfo, &encoder) != PYROWAVE_SUCCESS) return 1;

        pyrowave_cpu_buffer input = {};
        input.data[0] = y.data(); input.data[1] = cb.data(); input.data[2] = cr.data();
        input.row_stride_in_bytes[0] = W; input.row_stride_in_bytes[1] = CW; input.row_stride_in_bytes[2] = CW;
        input.plane_size_in_bytes[0] = y.size(); input.plane_size_in_bytes[1] = cb.size(); input.plane_size_in_bytes[2] = cr.size();
        input.width = W;
        input.height = H;
        input.format = PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;
        const size_t maxBytes = size_t(bpp * double(W) * H / 8.0);
        pyrowave_rate_control rate = {};
        rate.maximum_bitstream_size = maxBytes;
        if (pyrowave_encoder_encode_cpu_synchronous(encoder, &input, &rate) != PYROWAVE_SUCCESS) return 1;

        id<MTLTexture> outY = plane(device, W, H), outCb = plane(device, CW, CH), outCr = plane(device, CW, CH);
        // OUTPUT=pb: luma into plane 0 of a 10 bit biplanar CVPixelBuffer (IOSurface), as
        // PyroWaveDecoder.swift does, instead of a private texture.
        CVMetalTextureCacheRef textureCache = nullptr;
        CVPixelBufferRef pixelBuffer = nullptr;
        CVMetalTextureRef lumaRef = nullptr;
        if (getenv("OUTPUT") && strcmp(getenv("OUTPUT"), "pb") == 0) {
            CVMetalTextureCacheCreate(nullptr, nullptr, device, nullptr, &textureCache);
            NSDictionary *attributes = @{
                (id)kCVPixelBufferMetalCompatibilityKey: @YES,
                (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
            };
            CVPixelBufferCreate(nullptr, W, H, kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
                                (__bridge CFDictionaryRef)attributes, &pixelBuffer);
            CVMetalTextureCacheCreateTextureFromImage(nullptr, textureCache, pixelBuffer, nullptr,
                                                      MTLPixelFormatR16Unorm, W, H, 0, &lumaRef);
            outY = CVMetalTextureGetTexture(lumaRef);
            printf("output: luma into a CVPixelBuffer plane\n");
        }
        pyrowave_gpu_buffers planes = {};
        planes.planes[0] = opaque(outY); planes.planes[1] = opaque(outCb); planes.planes[2] = opaque(outCr);

        // --- Upstream path: the whole frame in one decode call.
        {
            const size_t boundary = 1 << 16;
            size_t packetCount = 0;
            pyrowave_encoder_compute_num_packets(encoder, boundary, &packetCount);
            std::vector<pyrowave_packet> packets(packetCount);
            std::vector<uint8_t> bitstream(maxBytes * 2 + (1 << 20));
            size_t written = 0;
            pyrowave_encoder_packetize(encoder, packets.data(), boundary, &written, bitstream.data(), bitstream.size());
            size_t coded = 0;
            for (size_t i = 0; i < written; i++) coded += packets[i].size;
            printf("coded %.0f KB (limit %.0f KB)\n", coded / 1024.0, maxBytes / 1024.0);

            pyrowave_decoder_create_info decInfo = {};
            decInfo.device = pwDevice;
            decInfo.width = W;
            decInfo.height = H;
            decInfo.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
            pyrowave_decoder decoder = nullptr;
            if (pyrowave_decoder_create(&decInfo, &decoder) != PYROWAVE_SUCCESS) return 1;
            std::vector<double> gpu;
            for (int it = 0; it < iterations + 3; it++) {
                pyrowave_decoder_clear(decoder);
                for (size_t i = 0; i < written; i++)
                    pyrowave_decoder_push_packet(decoder, bitstream.data() + packets[i].offset, packets[i].size);
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                if (pyrowave_decoder_decode_gpu_buffer(decoder, opaque(cmd), &planes) != PYROWAVE_SUCCESS) return 1;
                [cmd commit];
                [cmd waitUntilCompleted];
                if (it >= 3) gpu.push_back((cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0);
            }
            printf("upstream whole frame : GPU %.2f ms median (min %.2f)\n", median(gpu), *std::min_element(gpu.begin(), gpu.end()));
            pyrowave_decoder_destroy(decoder);
        }

        // --- ALVR stripe progressive path.
        const void *words = nullptr, *meta = nullptr;
        size_t wordBytes = 0, metaBytes = 0;
        if (pyrowave_encoder_get_mapped_raw_bitstream(encoder, &words, &wordBytes, &meta, &metaBytes) != PYROWAVE_SUCCESS) return 1;
        PyroWaveStripes::Geometry g;
        const int stripeHeight = 64;
        g.init(W, H, false, stripeHeight);
        std::vector<uint8_t> out;
        std::vector<uint32_t> sizes;
        const size_t packetBytes = getenv("PACKET_BYTES") ? atoi(getenv("PACKET_BYTES")) : 28 * 1024;
        const char *problem = PyroWaveStripes::build_packets(
            g, g.stripe_blocks(), static_cast<const uint32_t *>(words), wordBytes / 4,
            static_cast<const PyroWaveStripes::BlockMeta *>(meta), metaBytes / sizeof(PyroWaveStripes::BlockMeta),
            packetBytes, false, out, sizes);
        if (problem) {
            fprintf(stderr, "build_packets: %s\n", problem);
            return 1;
        }
        printf("stripe packets: %zu, %d stripes of %d rows\n", sizes.size(), g.stripe_count(), stripeHeight);

        pyrowave_decoder_create_info decInfo = {};
        decInfo.device = pwDevice;
        decInfo.width = W;
        decInfo.height = H;
        decInfo.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
        pyrowave_decoder decoder = nullptr;
        if (pyrowave_decoder_create(&decInfo, &decoder) != PYROWAVE_SUCCESS) return 1;
        if (pyrowave_decoder_enable_progressive(decoder, stripeHeight, maxBytes + (1 << 20)) != PYROWAVE_SUCCESS) return 1;
        const int stripes = pyrowave_decoder_stripe_count(decoder);

        const int stepCounts[] = { 1, 5, 10, 20, stripes };
        for (int steps : stepCounts) {
            std::vector<double> sum, span;
            for (int it = 0; it < iterations + 3; it++) {
                pyrowave_frame frame = nullptr;
                if (pyrowave_decoder_progressive_begin(decoder, &frame) != PYROWAVE_SUCCESS) return 1;
                size_t offset = 0;
                for (uint32_t size : sizes) {
                    const auto *prefix = reinterpret_cast<const PyroWaveStripes::PacketPrefix *>(out.data() + offset);
                    pyrowave_frame_push_packet(frame, out.data() + offset + sizeof(PyroWaveStripes::PacketPrefix), prefix->data_bytes);
                    offset += size;
                }
                std::vector<id<MTLCommandBuffer>> cmds;
                for (int s = 1; s <= steps; s++) {
                    const int complete = s == steps ? stripes : std::max(1, s * stripes / steps);
                    id<MTLCommandBuffer> cmd = [queue commandBuffer];
                    pyrowave_progress progress = {};
                    if (pyrowave_frame_decode(frame, opaque(cmd), &planes, complete, s == steps, &progress) != PYROWAVE_SUCCESS) return 1;
                    [cmd commit];
                    cmds.push_back(cmd);
                }
                [cmds.back() waitUntilCompleted];
                pyrowave_frame_end(frame);
                double total = 0;
                for (id<MTLCommandBuffer> c : cmds) {
                    [c waitUntilCompleted];
                    total += (c.GPUEndTime - c.GPUStartTime) * 1000.0;
                }
                if (it >= 3) {
                    sum.push_back(total);
                    span.push_back((cmds.back().GPUEndTime - cmds.front().GPUStartTime) * 1000.0);
                }
            }
            printf("progressive %2d steps : GPU %.2f ms median (sum over command buffers), first start to last end %.2f ms\n",
                   steps, median(sum), median(span));
        }
        pyrowave_decoder_destroy(decoder);
        pyrowave_encoder_destroy(encoder);
        pyrowave_device_destroy(pwDevice);
    }
    return 0;
}
