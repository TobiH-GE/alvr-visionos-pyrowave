# Vendored PyroWave Metal port

Copied from `metal/` of https://github.com/Themaister/pyrowave at commit
`89f7e47d4abbf650c91fae766728af866c5e32a0` (MIT, see `LICENSE`; upstream's notes on the port
are in `UPSTREAM-README.md`).

The streamer must run a PyroWave built from the same commit: the bitstream is what the two
sides share, and upstream still calls it a draft. See `deps/windows/pyrowave/README.md` in the
streamer repository.

Files taken: the `.mm`, `.cpp`, `.hpp` and `.h` sources and `shaders/pyrowave_msl.h`, which
holds the Metal shaders as source strings compiled at runtime. The `.metal` files and
`transpile.sh` are left out on purpose: added to the app target, Xcode would compile them into
the app's default library next to `Shaders.metal`.

## Local changes (ALVR stripe progressive decoding)

Upstream decodes a frame only once all of it has arrived. For ALVR the streamer sends each
frame as many packets in stripe order, and this copy decodes it stripe by stripe while it is
still arriving. Everything added is marked `ALVR` in the sources:

- `pyrowave_stripes.hpp` (new): the stripe schedule, identical to `PyroWaveStripes.h` in the
  streamer. It says which coefficient rows the first N rows of the picture need at every level.
- `pyrowave_metal.h`: the `pyrowave_frame` API at the end (`pyrowave_decoder_enable_progressive`,
  `pyrowave_decoder_progressive_begin`, `pyrowave_frame_push_packet`, `pyrowave_frame_decode`,
  `pyrowave_frame_end`).
- `pyrowave_decoder.mm`: its implementation at the end of the file. It adds `block_row_offset`
  (previously padding) to `DequantPush` and `tile_offset` to `IdwtPush`; both are 0 on the
  upstream code paths, which behave exactly as before.
- `shaders/pyrowave_msl.h`: the three iDWT variants and the dequant kernel add those offsets to
  the threadgroup position, so a dispatch can cover a range of rows.

To update, copy the same set of files from a newer commit, reapply the changes above, update
the hash here and in the streamer, and rebuild both.

`PyroWaveDecoder.swift` (one level up) is the glue to the rest of the client.
