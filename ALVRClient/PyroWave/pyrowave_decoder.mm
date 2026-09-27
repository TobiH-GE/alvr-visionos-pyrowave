// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Metal backend for the PyroWave decoder. Mirrors the compute path of
// pyrowave_decoder.cpp; the fragment iDWT path is not ported. The device object
// and the wavelet pyramid it shares with the encoder live in
// pyrowave_common.mm.

#include "pyrowave_common.hpp"

#include <atomic>
#include <memory>
#include <string.h>
#include <vector>

// ALVR extension: stripe progressive decoding, see pyrowave_stripes.hpp and the end of this file.
#include "pyrowave_stripes.hpp"

using namespace PyroWave;

namespace
{
// Push constant layouts. These must match the Registers structs SPIRV-Cross
// emitted into shaders/metal/*.metal. MSL gives int2 an 8 byte alignment, so
// the dequant struct is padded out to 24 bytes even though only 20 are used.
struct DequantPush
{
	int32_t resolution[2];
	int32_t output_layer;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
	// ALVR: first block row of a partial dispatch (see pyrowave_frame_decode). 0 for a full one.
	int32_t block_row_offset;
};
static_assert(sizeof(DequantPush) == 24, "DequantPush layout mismatch.");

struct IdwtPush
{
	int32_t resolution[2];
	float inv_resolution[2];
	// ALVR: first threadgroup of a partial dispatch (see pyrowave_frame_decode). 0 for a full one.
	int32_t tile_offset[2];
};
static_assert(sizeof(IdwtPush) == 24, "IdwtPush layout mismatch.");

// How many decodes a caller may have in flight before acquiring a slot waits for the
// oldest. Two would do for the intended display loop; four is headroom at ~1 MB per
// slot at 4K.
constexpr size_t UploadSlotCount = 4;

struct UploadSlot
{
	id<MTLBuffer> offsets;
	id<MTLBuffer> payload;
	id<MTLCommandBuffer> consumer;

	void reclaim()
	{
		if (!consumer)
			return;
		// Returns immediately unless the caller really is UploadSlotCount ahead.
		[consumer waitUntilCompleted];
		consumer = nil;
	}

	// No destructor: ARC releases both buffers, and Metal keeps anything a command
	// buffer still references alive on its own.
};
}

// ALVR extension: one pooled frame of stripe progressive decoding. The payload and offsets
// live in shared memory buffers the parser writes directly, so the GPU reads what arrived
// without another copy.
struct pyrowave_frame_opaque
{
	pyrowave_decoder decoder = nullptr;

	id<MTLBuffer> offsets; // uint32 per 32x32 block, UINT32_MAX for "not received"
	id<MTLBuffer> payload; // block headers and payloads, back to back
	size_t payload_capacity_words = 0;
	size_t payload_words = 0;

	std::atomic<int> received_blocks{0};
	std::atomic<int> total_blocks{0};
	uint32_t sequence = UINT32_MAX;
	bool logged_overflow = false;

	// Decode progress, per component and input level: block rows dequantized (all bands of
	// the level), iDWT tile rows done.
	int dequant_rows[PyroWaveStripes::Components][PyroWaveStripes::Levels] = {};
	int idwt_tiles[PyroWaveStripes::Components][PyroWaveStripes::Levels] = {};
	bool finished = false;

	std::atomic<bool> in_use{false};
	// The last command buffer that reads this frame's buffers.
	id<MTLCommandBuffer> consumer;
};

namespace
{
constexpr int ProgressiveFrameCount = 4;
}

struct pyrowave_decoder_opaque
{
	pyrowave_device device = nullptr;

	// ALVR extension: set by pyrowave_decoder_enable_progressive.
	PyroWaveStripes::Geometry stripes;
	bool progressive = false;
	pyrowave_frame_opaque frames[ProgressiveFrameCount];

	BlockLayout layout;
	BitstreamParser parser;
	WaveletPyramid wavelet;

	// Fixed size on purpose: a vector that appended whenever no slot was free grew
	// without bound when a caller submitted faster than the GPU drained.
	UploadSlot upload_slots[UploadSlotCount];
	size_t next_upload_slot = 0;
};

namespace
{
// Overallocates so that a steadily sized stream stops reallocating.
bool ensure_buffer(pyrowave_device device, __strong id<MTLBuffer> *buffer, size_t size)
{
	if (*buffer && (*buffer).length >= size)
		return true;

	size_t allocate = size * 2;
	if (allocate < 64 * 1024)
		allocate = 64 * 1024;

	*buffer = [device->mtl newBufferWithLength:allocate options:MTLResourceStorageModeShared];
	if (!*buffer)
	{
		device->log("Failed to allocate a %zu byte upload buffer.", allocate);
		return false;
	}

	return true;
}

UploadSlot *acquire_upload_slot(pyrowave_decoder decoder, size_t offsets_size, size_t payload_size)
{
	UploadSlot *slot = &decoder->upload_slots[decoder->next_upload_slot];
	decoder->next_upload_slot = (decoder->next_upload_slot + 1) % UploadSlotCount;

	// Applies back pressure rather than allocating another slot.
	slot->reclaim();

	if (!ensure_buffer(decoder->device, &slot->offsets, offsets_size) ||
	    !ensure_buffer(decoder->device, &slot->payload, payload_size))
		return nullptr;

	return slot;
}

void encode_dequant(pyrowave_decoder decoder, id<MTLComputeCommandEncoder> enc, UploadSlot *slot)
{
	auto &layout = decoder->layout;

	[enc setComputePipelineState:decoder->device->dequant_pipeline];
	// The u8/u16/u32 aliases of the payload collapse into a single binding in MSL.
	[enc setBuffer:slot->payload offset:0 atIndex:0];
	[enc setBuffer:slot->offsets offset:0 atIndex:2];

	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0 && layout.chroma == ChromaSubsampling::Chroma420)
				continue;

			[enc setTexture:decoder->wavelet.component_layer_views[component][level] atIndex:0];

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				DequantPush push = {};
				push.resolution[0] = layout.level_width(level);
				push.resolution[1] = layout.level_height(level);
				push.output_layer = band;
				push.block_offset_32x32 = layout.block_meta[component][level][band].block_offset_32x32;
				push.block_stride_32x32 = layout.block_meta[component][level][band].block_stride_32x32;
				[enc setBytes:&push length:sizeof(push) atIndex:1];

				[enc dispatchThreadgroups:MTLSizeMake((push.resolution[0] + 31) / 32,
				                                     (push.resolution[1] + 31) / 32, 1)
				      threadsPerThreadgroup:MTLSizeMake(DequantThreadgroupSize, 1, 1)];
			}
		}
	}
}

void encode_idwt_dispatch(pyrowave_decoder decoder, id<MTLComputeCommandEncoder> enc,
                          const IdwtPush &push, id<MTLTexture> input, id<MTLTexture> output,
                          bool dc_shift)
{
	[enc setComputePipelineState:decoder->device->idwt_pipeline[dc_shift ? 1 : 0]];
	[enc setBytes:&push length:sizeof(push) atIndex:0];
	[enc setTexture:input atIndex:0];
	[enc setTexture:output atIndex:1];
	[enc setSamplerState:decoder->device->mirror_repeat_sampler atIndex:0];
	[enc dispatchThreadgroups:MTLSizeMake((push.resolution[0] + 15) / 16,
	                                     (push.resolution[1] + 15) / 16, 1)
	      threadsPerThreadgroup:MTLSizeMake(IdwtThreadgroupSize, 1, 1)];
}

void encode_idwt(pyrowave_decoder decoder, id<MTLComputeCommandEncoder> enc,
                 id<MTLTexture> const planes[3])
{
	auto &layout = decoder->layout;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;

	for (int input_level = DecompositionLevels - 1; input_level >= 0; input_level--)
	{
		// Levels are a dependent chain: this one reads the LL band the previous
		// one produced. Within a level the three components are independent, so
		// the encoder runs concurrently and only the level boundaries barrier.
		if (input_level != DecompositionLevels - 1)
			[enc memoryBarrierWithScope:MTLBarrierScopeTextures];

		IdwtPush push = {};
		// The shader transposes on load, so resolution is swapped here.
		push.resolution[0] = layout.level_height(input_level);
		push.resolution[1] = layout.level_width(input_level);
		push.inv_resolution[0] = 1.0f / float(push.resolution[0]);
		push.inv_resolution[1] = 1.0f / float(push.resolution[1]);

		if (input_level == 0)
		{
			// Final level writes the output planes directly. Under 420 the chroma
			// planes were already finished one level earlier.
			const int components = chroma_420 ? 1 : NumComponents;
			for (int c = 0; c < components; c++)
			{
				encode_idwt_dispatch(decoder, enc, push,
				                     decoder->wavelet.component_layer_views[c][input_level],
				                     planes[c], true);
			}
		}
		else
		{
			for (int c = 0; c < NumComponents; c++)
			{
				const bool final_chroma = chroma_420 && c != 0 && input_level == 1;
				id<MTLTexture> output = final_chroma ?
				                       planes[c] :
				                       decoder->wavelet.component_ll_views[c][input_level - 1];

				encode_idwt_dispatch(decoder, enc, push,
				                     decoder->wavelet.component_layer_views[c][input_level],
				                     output, final_chroma);
			}
		}
	}
}

bool validate_plane(pyrowave_device device, id<MTLTexture> texture, int index, int width, int height)
{
	if (!texture)
	{
		device->log("Output plane %d is NULL.", index);
		return false;
	}

	if (texture.textureType != MTLTextureType2D)
	{
		device->log("Output plane %d must be MTLTextureType2D.", index);
		return false;
	}

	if (int(texture.width) != width || int(texture.height) != height)
	{
		device->log("Output plane %d is %ux%u, expected %dx%d.",
		            index, unsigned(texture.width), unsigned(texture.height), width, height);
		return false;
	}

	if ((texture.usage & MTLTextureUsageShaderWrite) == 0)
	{
		device->log("Output plane %d was not created with MTLTextureUsageShaderWrite.", index);
		return false;
	}

	return true;
}
}

//////
// Public API

pyrowave_result pyrowave_decoder_create(const pyrowave_decoder_create_info *info, pyrowave_decoder *decoder)
{
	if (!info || !decoder || !info->device)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (info->chroma != PYROWAVE_CHROMA_SUBSAMPLING_420 &&
	    info->chroma != PYROWAVE_CHROMA_SUBSAMPLING_444)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	const bool chroma_420 = info->chroma == PYROWAVE_CHROMA_SUBSAMPLING_420;
	if (chroma_420 && ((info->width & 1) != 0 || (info->height & 1) != 0))
	{
		info->device->log("420 subsampling requires even dimensions, got %dx%d.",
		                  info->width, info->height);
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	auto created = std::unique_ptr<pyrowave_decoder_opaque>(new (std::nothrow) pyrowave_decoder_opaque);
	if (!created)
		return PYROWAVE_ERROR_OUT_OF_HOST_MEMORY;

	created->device = info->device;

	if (!created->layout.init(info->width, info->height,
	                          chroma_420 ? ChromaSubsampling::Chroma420 : ChromaSubsampling::Chroma444))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	created->parser.init(&created->layout);

	if (!created->wavelet.init(created->device, created->layout))
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	*decoder = created.release();
	return PYROWAVE_SUCCESS;
}

void pyrowave_decoder_destroy(pyrowave_decoder decoder)
{
	delete decoder;
}

void pyrowave_decoder_clear(pyrowave_decoder decoder)
{
	if (decoder)
		decoder->parser.clear();
}

pyrowave_result pyrowave_decoder_push_packet(pyrowave_decoder decoder, const void *data, size_t size)
{
	if (!decoder || (!data && size != 0))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	if (!decoder->parser.push_packet(data, size))
		return PYROWAVE_ERROR_CORRUPT_BITSTREAM;

	return PYROWAVE_SUCCESS;
}

bool pyrowave_decoder_decode_is_ready(pyrowave_decoder decoder, bool allow_partial_frame)
{
	return decoder && decoder->parser.decode_is_ready(allow_partial_frame);
}

bool pyrowave_decoder_decode_is_ready_with_sideband(pyrowave_decoder decoder, bool allow_partial_frame,
                                                    int num_pristine_bands, float minimum_packet_ratio,
                                                    const uint32_t *active_block_mask, size_t word_count)
{
	// num_pristine_bands is range checked only by an assert in has_pristine_bands(),
	// matching the Vulkan C API. It indexes block_meta[..][DecompositionLevels - band][..],
	// so a large enough value walks off the front of the array with NDEBUG.
	if (!decoder)
		return false;

	return decoder->parser.decode_is_ready(allow_partial_frame, num_pristine_bands, minimum_packet_ratio,
	                                       active_block_mask, word_count);
}

pyrowave_result pyrowave_decoder_decode_gpu_buffer(pyrowave_decoder decoder,
                                                   pyrowave_mtl_command_buffer command_buffer,
                                                   const pyrowave_gpu_buffers *buffers)
{
	if (!decoder || !command_buffer || !buffers)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto &layout = decoder->layout;
	auto *device = decoder->device;

	id<MTLTexture> planes[3];
	const int chroma_width = layout.chroma == ChromaSubsampling::Chroma420 ?
	                         layout.width / 2 : layout.width;
	const int chroma_height = layout.chroma == ChromaSubsampling::Chroma420 ?
	                          layout.height / 2 : layout.height;

	for (int i = 0; i < 3; i++)
	{
		planes[i] = (__bridge id<MTLTexture>)(buffers->planes[i]);
		const int expected_width = i == 0 ? layout.width : chroma_width;
		const int expected_height = i == 0 ? layout.height : chroma_height;
		if (!validate_plane(device, planes[i], i, expected_width, expected_height))
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	const auto &offsets = decoder->parser.dequant_offsets();
	const auto &payload = decoder->parser.payload();

	const size_t offsets_size = offsets.size() * sizeof(uint32_t);
	// The dequant shader can read slightly past the end of the payload, so pad.
	const size_t payload_size = payload.size() * sizeof(uint32_t) + 16;

	auto *slot = acquire_upload_slot(decoder, offsets_size, payload_size);
	if (!slot)
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;

	if (offsets_size)
		memcpy(slot->offsets.contents, offsets.data(), offsets_size);
	if (!payload.empty())
		memcpy(slot->payload.contents, payload.data(), payload.size() * sizeof(uint32_t));

	auto *cmd = (__bridge id<MTLCommandBuffer>)(command_buffer);

	// Every dequant dispatch writes a distinct (component, level, band) region of
	// the pyramid and none reads another's output, so they can all run at once. A
	// serial encoder would barrier between all ~42, but the cost is underutilization
	// rather than barrier latency: each dispatch is far too small to fill the GPU on
	// its own, which is why this pays at low resolution and not at 1080p 4:4:4.
	auto dequant_enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
	if (!dequant_enc)
		return PYROWAVE_ERROR_GENERIC;

	dequant_enc.label = @("pyrowave dequant");
	encode_dequant(decoder, dequant_enc, slot);
	[dequant_enc endEncoding];

	// The iDWT is a dependent chain across levels, but the three components within
	// a level are independent, so this is also concurrent with explicit barriers
	// at the level boundaries only. Ordering against the dequant work above comes
	// from the encoder boundary, which Metal tracks automatically.
	auto idwt_enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
	if (!idwt_enc)
		return PYROWAVE_ERROR_GENERIC;

	idwt_enc.label = @("pyrowave idwt");
	encode_idwt(decoder, idwt_enc, planes);
	[idwt_enc endEncoding];

	// Retained until this slot comes round again, so its buffers cannot be rewritten
	// while the GPU is still reading them.
	slot->consumer = cmd;

	decoder->parser.mark_frame_decoded();
	return PYROWAVE_SUCCESS;
}

//////
// ALVR extension: stripe progressive decoding. See pyrowave_metal.h.

namespace
{
void encode_dequant_rows(pyrowave_decoder decoder, id<MTLComputeCommandEncoder> enc,
                         int component, int level, int first_row, int rows)
{
	auto &layout = decoder->layout;
	[enc setTexture:decoder->wavelet.component_layer_views[component][level] atIndex:0];

	for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
	{
		const auto &meta = layout.block_meta[component][level][band];
		DequantPush push = {};
		push.resolution[0] = layout.level_width(level);
		push.resolution[1] = layout.level_height(level);
		push.output_layer = band;
		push.block_offset_32x32 = meta.block_offset_32x32;
		push.block_stride_32x32 = meta.block_stride_32x32;
		push.block_row_offset = first_row;
		[enc setBytes:&push length:sizeof(push) atIndex:1];
		[enc dispatchThreadgroups:MTLSizeMake(meta.block_stride_32x32, rows, 1)
		      threadsPerThreadgroup:MTLSizeMake(DequantThreadgroupSize, 1, 1)];
	}
}
}

pyrowave_result pyrowave_decoder_enable_progressive(pyrowave_decoder decoder, int stripe_height, size_t max_frame_bytes)
{
	if (!decoder || decoder->progressive)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto &layout = decoder->layout;
	if (!decoder->stripes.init(layout.width, layout.height,
	                           layout.chroma == ChromaSubsampling::Chroma444, stripe_height))
	{
		decoder->device->log("Invalid stripe height %d.", stripe_height);
		return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}
	if (decoder->stripes.block_count() != layout.block_count_32x32)
	{
		decoder->device->log("Stripe geometry disagrees with the block layout (%d vs %d blocks).",
		                     decoder->stripes.block_count(), layout.block_count_32x32);
		return PYROWAVE_ERROR_GENERIC;
	}

	// The rate control target bounds the coded blocks; the headers PyroWave adds on top are
	// small. Twice the target plus the dequant shader's read-ahead padding.
	const size_t payload_bytes = max_frame_bytes * 2 + 64 * 1024;
	for (auto &frame : decoder->frames)
	{
		frame.decoder = decoder;
		frame.offsets = [decoder->device->mtl newBufferWithLength:size_t(layout.block_count_32x32) * sizeof(uint32_t)
		                                                  options:MTLResourceStorageModeShared];
		frame.payload = [decoder->device->mtl newBufferWithLength:payload_bytes + 16
		                                                  options:MTLResourceStorageModeShared];
		if (!frame.offsets || !frame.payload)
			return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;
		frame.payload_capacity_words = payload_bytes / sizeof(uint32_t);
	}

	decoder->progressive = true;
	return PYROWAVE_SUCCESS;
}

int pyrowave_decoder_stripe_count(pyrowave_decoder decoder)
{
	return decoder && decoder->progressive ? decoder->stripes.stripe_count() : 0;
}

pyrowave_result pyrowave_decoder_progressive_begin(pyrowave_decoder decoder, pyrowave_frame *frame_out)
{
	if (!decoder || !decoder->progressive || !frame_out)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	for (auto &frame : decoder->frames)
	{
		bool expected = false;
		if (!frame.in_use.compare_exchange_strong(expected, true))
			continue;

		// The previous frame in this slot has ended, but the GPU may still be reading it.
		// Returns at once unless the caller is a whole pool ahead of the GPU.
		if (frame.consumer)
		{
			[frame.consumer waitUntilCompleted];
			frame.consumer = nil;
		}

		memset(frame.offsets.contents, 0xff, frame.offsets.length);
		frame.payload_words = 0;
		frame.received_blocks = 0;
		frame.total_blocks = 0;
		frame.sequence = UINT32_MAX;
		frame.logged_overflow = false;
		memset(frame.dequant_rows, 0, sizeof(frame.dequant_rows));
		memset(frame.idwt_tiles, 0, sizeof(frame.idwt_tiles));
		frame.finished = false;

		*frame_out = &frame;
		return PYROWAVE_SUCCESS;
	}

	return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;
}

pyrowave_result pyrowave_frame_push_packet(pyrowave_frame frame, const void *data, size_t size)
{
	if (!frame || (!data && size != 0))
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto *decoder = frame->decoder;

	// The parser lives in pyrowave_stripes.hpp, where it is tested against PyroWave's own; it
	// writes straight into the frame's GPU buffers.
	PyroWaveStripes::ParseTarget target;
	target.offsets = static_cast<uint32_t *>(frame->offsets.contents);
	target.payload = static_cast<uint32_t *>(frame->payload.contents);
	target.payload_capacity_words = frame->payload_capacity_words;
	target.payload_words = frame->payload_words;
	target.sequence = frame->sequence;
	target.received_blocks = frame->received_blocks;
	target.total_blocks = frame->total_blocks;

	const auto result = PyroWaveStripes::parse_packet(decoder->stripes, decoder->layout.block_count_32x32,
	                                                  target, data, size);

	frame->payload_words = target.payload_words;
	frame->sequence = target.sequence;
	frame->received_blocks = target.received_blocks;
	frame->total_blocks = target.total_blocks;

	switch (result)
	{
	case PyroWaveStripes::ParseOk:
		return PYROWAVE_SUCCESS;
	case PyroWaveStripes::ParseOverflow:
		if (!frame->logged_overflow)
			decoder->device->log("Frame exceeds the payload buffer, dropping blocks.");
		frame->logged_overflow = true;
		return PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY;
	default:
		return PYROWAVE_ERROR_CORRUPT_BITSTREAM;
	}
}

pyrowave_result pyrowave_frame_decode(pyrowave_frame frame, pyrowave_mtl_command_buffer command_buffer,
                                      const pyrowave_gpu_buffers *buffers, int complete_stripes, bool final,
                                      pyrowave_progress *progress)
{
	if (!frame || !command_buffer || !buffers || frame->finished)
		return PYROWAVE_ERROR_INVALID_ARGUMENT;

	auto *decoder = frame->decoder;
	auto &layout = decoder->layout;
	const auto &g = decoder->stripes;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;

	id<MTLTexture> planes[3];
	for (int i = 0; i < 3; i++)
	{
		planes[i] = (__bridge id<MTLTexture>)(buffers->planes[i]);
		const int expected_width = i == 0 || !chroma_420 ? layout.width : layout.width / 2;
		const int expected_height = i == 0 || !chroma_420 ? layout.height : layout.height / 2;
		if (!validate_plane(decoder->device, planes[i], i, expected_width, expected_height))
			return PYROWAVE_ERROR_INVALID_ARGUMENT;
	}

	const int stripe_count = g.stripe_count();
	complete_stripes = std::max(0, std::min(complete_stripes, stripe_count));
	const int luma_rows = final ? g.aligned_height :
	                      complete_stripes == 0 ? 0 : g.luma_rows_through_stripe(complete_stripes - 1);

	// Targets for this call.
	int target_rows[PyroWaveStripes::Components][PyroWaveStripes::Levels];
	int target_tiles[PyroWaveStripes::Components][PyroWaveStripes::Levels];
	bool any_dequant = false, any_idwt = false;
	for (int c = 0; c < PyroWaveStripes::Components; c++)
	{
		int tiles[PyroWaveStripes::Levels], rows[PyroWaveStripes::Levels];
		g.requirements(c, luma_rows, tiles, rows);
		for (int level = 0; level < PyroWaveStripes::Levels; level++)
		{
			target_rows[c][level] = PyroWaveStripes::Geometry::div_ceil(rows[level], PyroWaveStripes::BlockSize);
			target_tiles[c][level] = tiles[level];
			any_dequant |= target_rows[c][level] > frame->dequant_rows[c][level];
			any_idwt |= target_tiles[c][level] > frame->idwt_tiles[c][level];
		}
	}

	auto plane_rows = [&](int c, const int tiles[PyroWaveStripes::Levels]) {
		const int f = g.final_level(c);
		const int plane_height = c == 0 || !chroma_420 ? layout.height : layout.height / 2;
		return std::min(tiles[f] * PyroWaveStripes::IdwtOutputRowsPerTile, plane_height);
	};

	if (progress)
		for (int c = 0; c < 3; c++)
			progress->previous_plane_rows[c] = plane_rows(c, frame->idwt_tiles[c]);

	auto *cmd = (__bridge id<MTLCommandBuffer>)(command_buffer);

	if (any_dequant)
	{
		// Dequant dispatches of different bands and levels touch disjoint texels.
		auto enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
		if (!enc)
			return PYROWAVE_ERROR_GENERIC;
		enc.label = @("pyrowave dequant (stripes)");
		[enc setComputePipelineState:decoder->device->dequant_pipeline];
		[enc setBuffer:frame->payload offset:0 atIndex:0];
		[enc setBuffer:frame->offsets offset:0 atIndex:2];

		for (int level = 0; level < PyroWaveStripes::Levels; level++)
			for (int c = 0; c < PyroWaveStripes::Components; c++)
			{
				if (!g.has_level(c, level))
					continue;
				const int done = frame->dequant_rows[c][level];
				const int target = target_rows[c][level];
				if (target > done)
					encode_dequant_rows(decoder, enc, c, level, done, target - done);
				frame->dequant_rows[c][level] = std::max(done, target);
			}
		[enc endEncoding];
	}

	if (any_idwt)
	{
		// Levels form a dependent chain, components within a level do not; the dequant work
		// above is ordered by the encoder boundary.
		auto enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
		if (!enc)
			return PYROWAVE_ERROR_GENERIC;
		enc.label = @("pyrowave idwt (stripes)");

		for (int level = PyroWaveStripes::Levels - 1; level >= 0; level--)
		{
			if (level != PyroWaveStripes::Levels - 1)
				[enc memoryBarrierWithScope:MTLBarrierScopeTextures];

			IdwtPush push = {};
			// The shader transposes on load: dimension 0 of the dispatch walks rows.
			push.resolution[0] = layout.level_height(level);
			push.resolution[1] = layout.level_width(level);
			push.inv_resolution[0] = 1.0f / float(push.resolution[0]);
			push.inv_resolution[1] = 1.0f / float(push.resolution[1]);

			for (int c = 0; c < PyroWaveStripes::Components; c++)
			{
				if (!g.has_level(c, level))
					continue;
				const int done = frame->idwt_tiles[c][level];
				const int target = target_tiles[c][level];
				if (target <= done)
					continue;

				push.tile_offset[0] = done;
				push.tile_offset[1] = 0;

				const bool writes_plane = level == g.final_level(c);
				id<MTLTexture> output = writes_plane ? planes[c] : decoder->wavelet.component_ll_views[c][level - 1];

				[enc setComputePipelineState:decoder->device->idwt_pipeline[writes_plane ? 1 : 0]];
				[enc setBytes:&push length:sizeof(push) atIndex:0];
				[enc setTexture:decoder->wavelet.component_layer_views[c][level] atIndex:0];
				[enc setTexture:output atIndex:1];
				[enc setSamplerState:decoder->device->mirror_repeat_sampler atIndex:0];
				[enc dispatchThreadgroups:MTLSizeMake(target - done, g.idwt_tile_columns(level), 1)
				      threadsPerThreadgroup:MTLSizeMake(IdwtThreadgroupSize, 1, 1)];
				frame->idwt_tiles[c][level] = target;
			}
		}
		[enc endEncoding];
	}

	if (any_dequant || any_idwt)
		frame->consumer = cmd;
	if (final)
		frame->finished = true;

	if (progress)
		for (int c = 0; c < 3; c++)
			progress->plane_rows[c] = plane_rows(c, frame->idwt_tiles[c]);

	return PYROWAVE_SUCCESS;
}

void pyrowave_frame_get_block_counts(pyrowave_frame frame, int *received_blocks, int *total_blocks)
{
	if (received_blocks)
		*received_blocks = frame ? frame->received_blocks.load() : 0;
	if (total_blocks)
		*total_blocks = frame ? frame->total_blocks.load() : 0;
}

void pyrowave_frame_end(pyrowave_frame frame)
{
	if (frame)
		frame->in_use = false;
}
