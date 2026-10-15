# TODO

## Frame interpolation for HEVC, H.264 and AV1 (not only PyroWave)

Goal: 45 fps from the streamer, 90 Hz on the headset with in-between frames for every codec, so
streams over Wi-Fi (HEVC) get the same halved game and encoder load as PyroWave on a cable.
Assessed 2026-10-10 from the code. **Done 2026-10-12 for NVENC** (HEVC tested at 6016x2144 and
7040x2496): the motion packets go ahead of the frame as extra video packets marked 0xffff, not as
a packet type of their own (step 2 below), and the client core skips them in its video handling.
Still open: AMF and software encoding (call FrameMotion like VideoEncoderNVENC does), compressing
the motion data (raw ~122 Mbit/s at 7040x2496), the headset's own estimate and "Choose Motion
per Pixel" for VideoToolbox frames.

Already independent of the codec:

- NVIDIA Optical Flow and the motion choice on the streamer work on the composed frame in SteamVR,
  before it is encoded.
- Head poses, head rotation per cell, the foveation mapping, the interpolation shader and the
  content clock (alpha) work on the decoded picture, whatever decoded it.
- The client already offers 45 (and 50) fps to the streamer for every codec.

To do, by repository:

1. Streamer (`ALVR-PyroWave`): move the motion module (`PyroWaveMotion`: Submit, Collect,
   motion choice) out of `VideoEncoderPyroWave` into the generic encoder path (`CEncoder`, after
   the frame copy), so it runs for NVENC/AMF/software encoding too.
2. Transport: today vectors and choice travel as extra PyroWave packets in the video stream
   (`PyroWaveStripes::MotionStripe`). For HEVC the client core expects one packet per frame, so
   extra packets would reach the decoder as broken frames. Give the motion data a packet type of
   its own in the ALVR protocol (server core and client core); older clients must ignore it.
   (SEI user data inside the HEVC stream would work too, but the frame would then wait for the
   vectors, ~10-17 ms after the copy: more latency.)
3. Client core (`alvr-client-core-pyrowave`): pass the new packets on to the app.
4. Client (`alvr-visionos-pyrowave`):
   - Move `FrameExtrapolation` out of `PyroWaveDecoder` one level up (EventHandler/Renderer),
     fed by both decoders; drop the `isPyroWaveFrame` checks around the interpolation and the
     hold-back.
   - VideoToolbox hands HEVC frames over in Apple's private packed formats, drawn by
     `videoFrameFragmentShader_SecretYpCbCrFormats` (samples as RGB): add the interpolation
     there (sampling at shifted positions works the same).
   - "Choose Motion per Pixel" on the headset and the headset's own motion estimate use
     PyroWave's lowpass bands. Not needed with the streamer's choice; for the headset's choice
     make a half resolution luma of the HEVC frame (one small pass, < 1 ms). The own estimate
     without streamer vectors is optional for HEVC (more expensive without the free pyramid).
5. Test with HEVC at 45 fps, first on Ethernet, then over Wi-Fi.

Risks:

- The HEVC decoder may buffer differently; whether frame times and motion data line up shows
  only in a test (hold-back counts, alpha on first display).
- HEVC artefacts at low bitrates may mislead the headset's choice (the streamer's works on the
  original frames).
- Touches all three repositories including a protocol change.
