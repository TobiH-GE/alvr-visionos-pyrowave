//
//  FrameExtrapolation.metal
//
//  Motion estimation between two decoded PyroWave frames and extrapolation of the newer one
//  half a frame (or any fraction) ahead, so that a 45 fps stream can be shown with motion at
//  90 Hz. See FrameExtrapolation.swift.
//
//  The search runs on PyroWave's own lowpass (LL) bands, which the decoder produces anyway on
//  its way to the full picture: level 3 is 1/16 of the picture, level 0 is 1/2. Coarse to fine:
//  a full search at the coarsest level, then at every finer level the best of the coarser
//  vectors around the cell, refined by +-2 texels. Every cell is 4x4 texels of its level and is
//  matched with the 8x8 window around it; at level 0 a cell is 8x8 picture pixels, i.e. 4x4
//  vectors per 32x32 PyroWave block.
//
//  The picture holds both eyes side by side; nothing reads across the middle.
//
//  Motion vectors are stored in picture pixels per frame interval of the source pair: the
//  content at x in the newer frame was at x - m(x) in the older one.
//

#include <metal_stdlib>
using namespace metal;

struct FXSearchParams {
    int2 levelSize;          // texels of this level holding picture, both eyes
    int eyeWidth;            // texels per eye at this level
    int cellsPerEye;
    int2 gridSize;           // cells: 2 * cellsPerEye by rows
    int parentCellsPerEye;   // 0 at the coarsest level (full search)
    int2 parentGridSize;
    int searchRadius;        // full search radius, or the refinement radius (<= 2)
    float pixelsPerTexel;    // picture pixels per texel of this level, 2 << level
    float valueScale;        // texel value to 8 bit units
    float lambda;            // cost per texel of distance from the predictor, in SAD units
    int subpixel;            // fit a parabola around the best integer vector
    float2 predictedShift;   // motion expected everywhere, picture pixels, added to the rotation field's
    int rotationCellsPerEye; // the rotation field (motion from head rotation alone) is on the finest grid
    int rotationCellScale;   // finest cells per cell of this level, per axis
    int debugStop;           // profiling only: 1 return after the window, 2 after the predictors, 3 after the patch
    float confidenceRatio;   // > 0: a refined vector must beat the predictor's SAD by this share ...
    float confidenceMinGain; // ... and by at least this much (8 bit units over the 8x8 window), else the predictor stays
};

constant int FXCell = 4;
constant int FXWindow = 8;
constant int FXWindowOffset = -2; // window starts 2 texels before the cell

constant int FXCoarseMaxRadius = 8;
constant int FXRefineMaxRadius = 2;
constant int FXCellsPerGroup = 4;

// The 2x2 texels from p (top left) on, times the value scale, as (0,0) (1,0) (0,1) (1,1). The block
// is moved, not mirrored, to stay inside the eye: only cells at the eye edge see the difference.
static inline float4 fx_gather(texture2d<float, access::sample> t, int2 p, int eye, constant FXSearchParams &prm) {
    const int x0 = eye * prm.eyeWidth;
    p = int2(clamp(p.x, x0, x0 + prm.eyeWidth - 2), clamp(p.y, 0, prm.levelSize.y - 2));
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest);
    // gather returns (0,1) (1,1) (1,0) (0,0) of the 2x2 footprint around the coordinate.
    const float4 g = t.gather(s, float2(p + 1) / float2(t.get_width(), t.get_height()));
    return float4(g.w, g.z, g.x, g.y) * prm.valueScale;
}

// One SIMD group per cell. cur = newer frame, prev = older frame, both at this level.
//
// The SADs of the final search all come from threadgroup memory: the window of the cell and
// the patch of the older frame the search can reach. Only the predictors of a refinement level
// read the older frame directly, and only the distinct ones.
template <int MaxRadius>
static void fx_motion_search(texture2d<float, access::sample> cur, texture2d<float, access::sample> prev,
                             texture2d<float, access::read> parentMV, texture2d<float, access::write> outMV,
                             texture2d<float, access::read> rotation,
                             constant FXSearchParams &prm, uint2 requestedCell, uint lane,
                             threadgroup float *win, threadgroup float *patch, threadgroup float *costs) {
    // Groups at the right edge carry cells past the grid: they still pass every barrier.
    const bool valid = int(requestedCell.x) < prm.gridSize.x;
    const uint2 cell = uint2(min(int(requestedCell.x), prm.gridSize.x - 1), requestedCell.y);

    const int eye = int(cell.x) / prm.cellsPerEye;
    const int localCell = int(cell.x) - eye * prm.cellsPerEye;
    const int2 cellOrigin = int2(eye * prm.eyeWidth + localCell * FXCell, int(cell.y) * FXCell);
    const int2 origin = cellOrigin + int2(FXWindowOffset);

    if (lane < 16) {
        const int bx = int(lane) % 4, by = int(lane) / 4;
        const float4 g = fx_gather(cur, origin + int2(2 * bx, 2 * by), eye, prm);
        win[(2 * by) * FXWindow + 2 * bx] = g.x;
        win[(2 * by) * FXWindow + 2 * bx + 1] = g.y;
        win[(2 * by + 1) * FXWindow + 2 * bx] = g.z;
        win[(2 * by + 1) * FXWindow + 2 * bx + 1] = g.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (prm.debugStop == 1) { if (lane == 0 && valid) outMV.write(float4(win[0]), cell); return; }

    // Displacement d (texels): cur(x) ~ prev(x + d). Motion m = -d.
    // The motion head rotation alone explains (from the two frames' poses) is where the search starts.
    const int rotationCell = min(localCell * prm.rotationCellScale + prm.rotationCellScale / 2, prm.rotationCellsPerEye - 1);
    const int rotationRow = min(int(cell.y) * prm.rotationCellScale + prm.rotationCellScale / 2, int(rotation.get_height()) - 1);
    const float2 expected = prm.predictedShift + rotation.read(uint2(eye * prm.rotationCellsPerEye + rotationCell, rotationRow)).rg;
    const int2 predictor = int2(round(-expected / prm.pixelsPerTexel));
    const bool coarsest = prm.parentCellsPerEye == 0;
    const int radius = min(prm.searchRadius, MaxRadius);
    int2 own = predictor;
    int2 center = predictor;
    float predictorSad = INFINITY;

    if (!coarsest) {
        // Candidates: the parent cell and its 8 neighbours (same eye), and the predictor.
        const int localParent = min(localCell / 2, prm.parentCellsPerEye - 1);
        const int parentRow = min(int(cell.y) / 2, prm.parentGridSize.y - 1);
        own = int2(round(-parentMV.read(uint2(eye * prm.parentCellsPerEye + localParent, parentRow)).rg / prm.pixelsPerTexel));
        int2 d = predictor;
        if (lane < 9) {
            int px = clamp(localParent + int(lane % 3) - 1, 0, prm.parentCellsPerEye - 1);
            int py = clamp(parentRow + int(lane / 3) - 1, 0, prm.parentGridSize.y - 1);
            d = int2(round(-parentMV.read(uint2(eye * prm.parentCellsPerEye + px, py)).rg / prm.pixelsPerTexel));
        }
        // The whole group evaluates every candidate: two candidates at a time, 16 lanes each, one
        // 2x2 gather per lane, and all five gathers of a lane are independent so they are in flight
        // together. One candidate after another, waiting for its reads and its sum, and one read per
        // texel, both cost more than the whole search below (measured).
        const int block = int(lane) & 15;
        const int bx = block % 4, by = block / 4;
        const float4 w = float4(win[(2 * by) * FXWindow + 2 * bx], win[(2 * by) * FXWindow + 2 * bx + 1],
                                win[(2 * by + 1) * FXWindow + 2 * bx], win[(2 * by + 1) * FXWindow + 2 * bx + 1]);
        const int2 blockOrigin = origin + int2(2 * bx, 2 * by);
        // Neighbouring parents mostly agree: only distinct candidates are evaluated, two per pass.
        bool duplicate = lane >= 10;
        for (ushort j = 0; j < 9; j++) {
            const int2 other = int2(simd_shuffle(d.x, j), simd_shuffle(d.y, j));
            duplicate = duplicate || (j < lane && all(other == d));
        }
        uint unique = uint(static_cast<simd_vote::vote_t>(simd_ballot(!duplicate)));
        float bestPredictorCost = INFINITY;
        // Raw SAD of the predictor (the motion head rotation alone explains), for the confidence test.
        predictorSad = INFINITY;
        while (unique != 0) {
            const ushort ja = ushort(ctz(unique));
            unique &= unique - 1;
            const ushort jb = unique != 0 ? ushort(ctz(unique)) : ja;
            unique &= unique - 1;
            const ushort j = (lane >> 4) != 0 ? jb : ja;
            const int2 dj = int2(simd_shuffle(d.x, j), simd_shuffle(d.y, j));
            const float4 diff = fabs(w - fx_gather(prev, blockOrigin + dj, eye, prm));
            float s = diff.x + diff.y + diff.z + diff.w;
            s += simd_shuffle_xor(s, 8);
            s += simd_shuffle_xor(s, 4);
            s += simd_shuffle_xor(s, 2);
            s += simd_shuffle_xor(s, 1);
            const float cost = s + prm.lambda * float(abs(dj.x - own.x) + abs(dj.y - own.y));
            // Lane 0 holds candidate ja, lane 16 candidate jb.
            const float costA = simd_shuffle(cost, ushort(0)), costB = simd_shuffle(cost, ushort(16));
            const int2 da = int2(simd_shuffle(dj.x, ushort(0)), simd_shuffle(dj.y, ushort(0)));
            const int2 db = int2(simd_shuffle(dj.x, ushort(16)), simd_shuffle(dj.y, ushort(16)));
            if (all(da == predictor)) predictorSad = simd_shuffle(s, ushort(0));
            if (all(db == predictor)) predictorSad = simd_shuffle(s, ushort(16));
            if (costA < bestPredictorCost) { bestPredictorCost = costA; center = da; }
            if (costB < bestPredictorCost) { bestPredictorCost = costB; center = db; }
        }
    }

    if (prm.debugStop == 2) { if (lane == 0 && valid) outMV.write(float4(float2(center), 0, 0), cell); return; }

    // The patch of the older frame every candidate of the search below reads.
    const int side = 2 * radius + 1;
    const int patchSize = FXWindow + 2 * radius;
    const int2 patchOrigin = origin + center - int2(radius);
    const int halfPatch = patchSize / 2;
    for (int i = int(lane); i < halfPatch * halfPatch; i += 32) {
        const int px = 2 * (i % halfPatch), py = 2 * (i / halfPatch);
        const float4 g = fx_gather(prev, patchOrigin + int2(px, py), eye, prm);
        patch[py * patchSize + px] = g.x;
        patch[py * patchSize + px + 1] = g.y;
        patch[(py + 1) * patchSize + px] = g.z;
        patch[(py + 1) * patchSize + px + 1] = g.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (prm.debugStop == 3) { if (lane == 0 && valid) outMV.write(float4(patch[lane]), cell); return; }

    float bestCost = INFINITY;
    float bestSad = INFINITY;
    int bestIndex = 0;
    for (int c = int(lane); c < side * side; c += 32) {
        const int ox = c % side, oy = c / side;
        float s = 0.0f;
        for (int y = 0; y < FXWindow; y++) {
            for (int x = 0; x < FXWindow; x++) {
                s += fabs(win[y * FXWindow + x] - patch[(y + oy) * patchSize + x + ox]);
            }
        }
        const int2 d = center + int2(ox - radius, oy - radius);
        const float cost = s + prm.lambda * float(abs(d.x - own.x) + abs(d.y - own.y));
        if (c < 32) costs[c] = cost;
        if (cost < bestCost) {
            bestCost = cost;
            bestSad = s;
            bestIndex = c;
        }
    }
    float minCost = simd_min(bestCost);
    int winnerIndex = simd_min(bestCost == minCost ? bestIndex : 0x7fffffff);
    const float winnerSad = simd_shuffle(bestSad, ushort(winnerIndex & 31));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0 && valid) {
        const int bx = winnerIndex % side, by = winnerIndex / side;
        int2 d = center + int2(bx - radius, by - radius);
        float2 sub = float2(0.0f);
        // Flat or ambiguous areas (a menu background around text) match almost anything; a vector
        // there is noise that drags the edges around it. Keep the predictor unless the vector is
        // clearly better.
        const bool confident = prm.confidenceRatio <= 0.0f || coarsest || !isfinite(predictorSad) ||
            predictorSad - winnerSad >= max(prm.confidenceMinGain, prm.confidenceRatio * predictorSad);
        if (!confident) {
            d = predictor;
        }
        // The parabola needs the neighbours' costs, kept for searches of up to 32 candidates.
        if (confident && prm.subpixel != 0 && side * side <= 32) {
            if (bx > 0 && bx < side - 1) {
                float l = costs[by * side + bx - 1], c = costs[by * side + bx], r = costs[by * side + bx + 1];
                float den = l - 2.0f * c + r;
                if (den > 1e-3f) sub.x = clamp(0.5f * (l - r) / den, -0.5f, 0.5f);
            }
            if (by > 0 && by < side - 1) {
                float u = costs[(by - 1) * side + bx], c = costs[by * side + bx], b = costs[(by + 1) * side + bx];
                float den = u - 2.0f * c + b;
                if (den > 1e-3f) sub.y = clamp(0.5f * (u - b) / den, -0.5f, 0.5f);
            }
        }
        // Not confident: exactly the motion head rotation explains, so nothing is left over once the
        // rotation is subtracted (the rounded predictor would leave up to half a texel of noise).
        const float2 m = confident ? -(float2(d) + sub) * prm.pixelsPerTexel : expected;
        outMV.write(float4(m, 0.0f, 0.0f), cell);
    }
}

// One SIMD group per cell, FXCellsPerGroup cells side by side per threadgroup (a threadgroup of
// a single SIMD group per cell made the per-group fixed cost dominate). Threadgroup memory is
// sized for the search radius: the coarse search needs a 24x24 patch, a refinement 12x12.
kernel void fxMotionSearchCoarse(texture2d<float, access::sample> cur [[texture(0)]],
                                 texture2d<float, access::sample> prev [[texture(1)]],
                                 texture2d<float, access::read> parentMV [[texture(2)]],
                                 texture2d<float, access::write> outMV [[texture(3)]],
                                 texture2d<float, access::read> rotation [[texture(4)]],
                                 constant FXSearchParams &prm [[buffer(0)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint simd [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float win[FXCellsPerGroup][FXWindow * FXWindow];
    threadgroup float patch[FXCellsPerGroup][(FXWindow + 2 * FXCoarseMaxRadius) * (FXWindow + 2 * FXCoarseMaxRadius)];
    threadgroup float costs[FXCellsPerGroup][32];
    fx_motion_search<FXCoarseMaxRadius>(cur, prev, parentMV, outMV, rotation, prm, uint2(group.x * FXCellsPerGroup + simd, group.y), lane,
                         win[simd], patch[simd], costs[simd]);
}

kernel void fxMotionSearchRefine(texture2d<float, access::sample> cur [[texture(0)]],
                                 texture2d<float, access::sample> prev [[texture(1)]],
                                 texture2d<float, access::read> parentMV [[texture(2)]],
                                 texture2d<float, access::write> outMV [[texture(3)]],
                                 texture2d<float, access::read> rotation [[texture(4)]],
                                 constant FXSearchParams &prm [[buffer(0)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint simd [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float win[FXCellsPerGroup][FXWindow * FXWindow];
    threadgroup float patch[FXCellsPerGroup][(FXWindow + 2 * FXRefineMaxRadius) * (FXWindow + 2 * FXRefineMaxRadius)];
    threadgroup float costs[FXCellsPerGroup][32];
    fx_motion_search<FXRefineMaxRadius>(cur, prev, parentMV, outMV, rotation, prm, uint2(group.x * FXCellsPerGroup + simd, group.y), lane,
                         win[simd], patch[simd], costs[simd]);
}

// 3x3 component-wise median of the vector field, within each eye, minus the motion head rotation
// explains: the renderer already turns every frame to the current head pose, so only the rest of the
// motion (the scene's own, and parallax) is extrapolated.
kernel void fxMedian(texture2d<float, access::read> inMV [[texture(0)]],
                     texture2d<float, access::write> outMV [[texture(1)]],
                     texture2d<float, access::read> rotation [[texture(2)]],
                     constant int &cellsPerEye [[buffer(0)]],
                     constant float &deadzone [[buffer(1)]],
                     uint2 cell [[thread_position_in_grid]]) {
    const int w = int(inMV.get_width()), h = int(inMV.get_height());
    if (int(cell.x) >= w || int(cell.y) >= h) return;
    const int eye = int(cell.x) / cellsPerEye;
    const int x0 = eye * cellsPerEye, x1 = x0 + cellsPerEye - 1;
    float xs[9], ys[9];
    int n = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 q = int2(clamp(int(cell.x) + dx, x0, x1), clamp(int(cell.y) + dy, 0, h - 1));
            float2 m = inMV.read(uint2(q)).rg;
            xs[n] = m.x; ys[n] = m.y; n++;
        }
    }
    // Partial selection sort up to the median.
    for (int i = 0; i <= 4; i++) {
        for (int j = i + 1; j < 9; j++) {
            if (xs[j] < xs[i]) { float t = xs[i]; xs[i] = xs[j]; xs[j] = t; }
            if (ys[j] < ys[i]) { float t = ys[i]; ys[i] = ys[j]; ys[j] = t; }
        }
    }
    // Motion below the dead zone is noise more often than motion: text on a panel that only moves
    // with the head stays exactly where the reprojection puts it instead of trembling.
    float2 scene = float2(xs[4], ys[4]) - rotation.read(cell).rg;
    if (length(scene) < deadzone) scene = float2(0.0f);
    // ba: the motion head rotation alone explains here, for blending with the previous frame,
    // which the renderer has to map through it.
    outMV.write(float4(scene, rotation.read(cell).rg), cell);
}

// MARK: - Motion explained by head rotation

struct FXRotationParams {
    float3x3 rotation;       // older frame's eye from the newer frame's eye: R_older^T * R_newer
    float4 tangentsLeft;     // stream view tangents of each eye as the renderer uses them:
    float4 tangentsRight;    //   left, right, up, down (all positive)
    float2 frameSize;        // picture pixels, both eyes
    float2 eyeSizeRatio;     // FFR EYE_SIZE_RATIO, (1, 1) without foveation
    float cellPixels;        // picture pixels per cell of the field
    int cellsPerEye;
    int lutSize;             // entries per foveation table, 0 without foveation
    int pad;
};

// Linear lookup in one of the four foveation tables (FrameExtrapolation.swift builds them from the
// renderer's decompressAxisAlignedCoord): 0/1 = x/y undistorted -> compressed, 2/3 = the inverse.
static inline float fx_lut(device const float *lut, int table, int size, float x) {
    const float f = clamp(x, 0.0f, 1.0f) * float(size - 1);
    const int i = min(int(f), size - 2);
    const device float *t = lut + table * size;
    return mix(t[i], t[i + 1], f - float(i));
}

// For every cell centre of the newer frame, where the same view direction was in the older frame
// when only the head turned; the motion that explains, newer minus older position, per cell. It is
// the same mapping the renderer uses to draw a frame (foveation, eye tangents, the frame's pose),
// so subtracting it leaves the motion the renderer does not already handle.
kernel void fxRotationField(texture2d<float, access::write> out [[texture(0)]],
                            constant FXRotationParams &prm [[buffer(0)]],
                            device const float *lut [[buffer(1)]],
                            uint2 cell [[thread_position_in_grid]]) {
    if (cell.x >= out.get_width() || cell.y >= out.get_height()) return;
    const int eye = int(cell.x) / prm.cellsPerEye;
    const int local = int(cell.x) - eye * prm.cellsPerEye;
    const float eyeWidth = prm.frameSize.x * 0.5f;
    const float2 p = float2(float(eye) * eyeWidth + (float(local) + 0.5f) * prm.cellPixels,
                            (float(cell.y) + 0.5f) * prm.cellPixels);
    const float2 t = p / prm.frameSize;
    // The renderer's TextureToEyeUV: the right eye's foveation is mirrored.
    const float2 c = float2(eye == 0 ? t.x * 2.0f : (1.0f - t.x) * 2.0f, t.y);
    float2 e = c;
    if (prm.lutSize > 0) {
        const float2 cn = c / prm.eyeSizeRatio;
        e = float2(fx_lut(lut, 2, prm.lutSize, cn.x), fx_lut(lut, 3, prm.lutSize, cn.y));
    }
    const float4 tan = eye == 0 ? prm.tangentsLeft : prm.tangentsRight;
    const float u = eye == 0 ? e.x : 1.0f - e.x;
    const float3 d = float3(mix(-tan.x, tan.y, u), mix(tan.z, -tan.w, e.y), -1.0f);
    const float3 dOld = prm.rotation * d;
    if (dOld.z > -1e-3f) {
        out.write(float4(0.0f), cell);
        return;
    }
    const float2 tanOld = dOld.xy / -dOld.z;
    const float uOld = (tanOld.x + tan.x) / (tan.x + tan.y);
    const float vOld = (tan.z - tanOld.y) / (tan.z + tan.w);
    float2 eOld = float2(eye == 0 ? uOld : 1.0f - uOld, vOld);
    float2 cOld = eOld;
    if (prm.lutSize > 0) {
        // Outside the frame the tables clamp: the motion saturates at the frame edge.
        cOld = float2(fx_lut(lut, 0, prm.lutSize, eOld.x), fx_lut(lut, 1, prm.lutSize, eOld.y)) * prm.eyeSizeRatio;
    }
    const float2 tOld = float2(eye == 0 ? cOld.x * 0.5f : 1.0f - cOld.x * 0.5f, cOld.y);
    out.write(float4(p - tOld * prm.frameSize, 0.0f, 0.0f), cell);
}

// Cells the streamer marked unreliable (flat: optical flow has nothing to match there) take the
// mean motion of the reliable cells nearest to them, within up to 6 cells: the motion of a flat
// area does not show, but its neighbours' keeps the warp from dragging their edges into it.
// motion: rg = motion, b = 1 if reliable. Cells with no reliable neighbour keep head rotation only.
kernel void fxFillUnreliable(texture2d<float, access::read> motion [[texture(0)]],
                             texture2d<float, access::read> rotation [[texture(1)]],
                             texture2d<float, access::write> filled [[texture(2)]],
                             constant int &cellsPerEye [[buffer(0)]],
                             uint2 cell [[thread_position_in_grid]]) {
    const int w = int(motion.get_width()), h = int(motion.get_height());
    if (int(cell.x) >= w || int(cell.y) >= h) return;
    const float4 own = motion.read(cell);
    if (own.b > 0.5f) {
        filled.write(float4(own.rg, 0.0f, 0.0f), cell);
        return;
    }
    const int eye = int(cell.x) / cellsPerEye;
    const int x0 = eye * cellsPerEye, x1 = x0 + cellsPerEye - 1;
    for (int r = 1; r <= 6; r++) {
        float2 sum = float2(0.0f);
        float n = 0.0f;
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                if (max(abs(dx), abs(dy)) != r) continue;
                const int2 q = int2(int(cell.x) + dx, int(cell.y) + dy);
                if (q.x < x0 || q.x > x1 || q.y < 0 || q.y >= h) continue;
                const float4 m = motion.read(uint2(q));
                if (m.b > 0.5f) {
                    // Their motion relative to their own rotation, moved onto this cell's rotation.
                    sum += m.rg - rotation.read(uint2(q)).rg;
                    n += 1.0f;
                }
            }
        }
        if (n > 0.0f) {
            filled.write(float4(sum / n + rotation.read(cell).rg, 0.0f, 0.0f), cell);
            return;
        }
    }
    filled.write(float4(rotation.read(cell).rg, 0.0f, 0.0f), cell);
}

// Frame interpolation, "choose motion per pixel": for every 8x8 pixels (FX_SELECT_BLOCK) of the
// in-between half way (both eyes), WHICH motion (previous -> newer frame) the two frames agree best
// under around it: 0 the streamer's field, 1 head rotation only (world-fixed content), 2 none
// (content fixed to the view, like a HUD), 3-6 the field one cell left/right/up/down. Compared on
// the frames' PyroWave lowpass level 0 (half resolution) at 3x3 points 4 picture pixels apart:
// at 7040x2496 the full resolution luma with 4x4 blocks cost ~20 ms of GPU per frame (2026-10-09).
// Only the choice is stored (.r); the renderer evaluates the chosen motion per pixel (mode 2 with
// both frames, mode 3 with the newer frame only), and choice 0 is the plain path. Storing the
// vector instead (constant per block) tore text apart on panels at fast head turns.
// Tuned with tools/frameflow/recplay (branch frame-interpolation-archive; modes selk/sel1,
// SEL_HALF=1 SEL_BLOCK=8 SEL_WIN=4 SEL_STEP=4 SEL_REL=0.35 SEL_ABS=0.04 SEL_MINCONTRAST=0.03) on
// three recordings: HUD, menu, fast-turn panel.
#ifndef FX_SELECT_BLOCK
#include "ShaderTypes.h"
#endif
struct FXSelectParams {
    float2 frameSize;
    float cellPixels;
    int cellsPerEye;
    int gridRows;
    int iterations;
    int candidates;      // 3-7 (app settings "Candidates")
    int samplesPerSide;  // 2 or 3 (app setting "Samples": 2x2 or 3x3)
};

static float4 fx_select_field(texture2d<float> motion, constant FXSelectParams &p, float2 y, int eye) {
    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);
    const float eyeWidth = p.frameSize.x * 0.5;
    const float2 grid = float2(2 * p.cellsPerEye, p.gridRows);
    float gx = clamp((y.x - float(eye) * eyeWidth) / p.cellPixels, 0.5, float(p.cellsPerEye) - 0.5) + float(eye * p.cellsPerEye);
    float gy = clamp(y.y / p.cellPixels, 0.5, grid.y - 0.5);
    return motion.sample(linearSampler, float2(gx, gy) / grid);
}

kernel void fxSelectMotion(texture2d<float> curY [[texture(0)]],
                           texture2d<float> prevY [[texture(1)]],
                           texture2d<float> motion [[texture(2)]],
                           texture2d<float, access::write> selected [[texture(3)]],
                           constant FXSelectParams &p [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= selected.get_width() || gid.y >= selected.get_height()) {
        return;
    }
    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);
    const float2 llSize = float2(curY.get_width(), curY.get_height());
    const float2 y = float2(gid) * float(FX_SELECT_BLOCK) + float(FX_SELECT_BLOCK) * 0.5;
    const float eyeWidth = p.frameSize.x * 0.5;
    const int eye = y.x >= eyeWidth ? 1 : 0;
    const float2 lo = float2(float(eye) * eyeWidth + 0.5, 0.5);
    const float2 hi = float2(float(eye + 1) * eyeWidth - 0.5, p.frameSize.y - 0.5);
    const float a = 0.5;
    const float2 rot = fx_select_field(motion, p, y, eye).ba;
    float2 x = y;
    for (int i = 0; i < p.iterations; i++) {
        x = y + a * fx_select_field(motion, p, x, eye).rg;
    }
    const float c = p.cellPixels;
    float2 cand[7];
    cand[0] = fx_select_field(motion, p, x, eye).rg + rot;
    cand[1] = rot;
    cand[2] = float2(0.0);
    cand[3] = fx_select_field(motion, p, y + float2(-c, 0), eye).rg + rot;
    cand[4] = fx_select_field(motion, p, y + float2(c, 0), eye).rg + rot;
    cand[5] = fx_select_field(motion, p, y + float2(0, -c), eye).rg + rot;
    cand[6] = fx_select_field(motion, p, y + float2(0, c), eye).rg + rot;
    float best = 0.0;
    int chosen = 0;
    // The first p.candidates of the seven: 3 = the streamer's field, head rotation only and none.
    // The neighbour cells (3-6) made no visible difference on the recordings (2026-10-09, recplay
    // SEL_NCAND=3); without them, and with 2x2 instead of 3x3 points, the choice costs about a fifth.
    // Points spread over +-4 picture pixels (+-2 level 0 texels): 2x2 at -4/+4, 3x3 at -4/0/+4.
    const int step = p.samplesPerSide >= 3 ? 4 : 8;
    for (int k = 0; k < p.candidates; k++) {
        const float2 t = cand[k];
        const float2 xc = clamp(y + a * (t - rot), lo, hi);
        const float2 pc = clamp(xc - t, lo, hi);
        float cost = 0.0;
        float lmin = 1.0, lmax = 0.0;
        for (int oy = -4; oy <= 4; oy += step) {
            for (int ox = -4; ox <= 4; ox += step) {
                const float2 o = float2(ox, oy);
                // Level 0 is half the picture size (its texture may be aligned wider).
                const float lc = curY.sample(linearSampler, (xc + o) * 0.5 / llSize).r;
                cost += abs(lc - prevY.sample(linearSampler, (pc + o) * 0.5 / llSize).r);
                lmin = min(lmin, lc);
                lmax = max(lmax, lc);
            }
        }
        // Another candidate only where it fits clearly better than the current choice (the
        // streamer's own vector first), and only where the samples show structure: on a flat area
        // every candidate fits, and "no motion" won next to text, cutting off descenders.
        if (k == 0 || (lmax - lmin >= 0.03 && cost < best * 0.35 - 0.04)) {
            best = cost;
            chosen = k;
        }
    }
    selected.write(float4(float(chosen), 0, 0, 0), gid);
}
