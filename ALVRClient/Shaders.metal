//
//  Shaders.metal
//

// File for Metal kernel and shader functions

#include <metal_stdlib>
#include <simd/simd.h>

// Including header shared between this Metal shader code and Swift/C code executing Metal API commands
#import "ShaderTypes.h"

using namespace metal;

typedef struct
{
    float3 position [[attribute(VertexAttributePosition)]];
    float2 texCoord [[attribute(VertexAttributeTexcoord)]];
} Vertex;

typedef struct
{
    float3 position [[attribute(VertexAttributePosition)]];
} VertexPosOnly;

typedef struct
{
    float4 position [[position]];
    float4 viewPosition;
    float4 color;
    float2 texCoord;
    float planeDoProximity;
} ColorInOutPlane;

typedef struct
{
    float4 position [[position]];
    float2 texCoord;
} ColorInOut;

vertex ColorInOutPlane vertexShader(Vertex in [[stage_in]],
                               ushort amp_id [[amplification_id]],
                               constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                               constant PlaneUniform & planeUniform [[ buffer(BufferIndexPlaneUniforms) ]])
{
    ColorInOutPlane out;

    Uniforms uniforms = uniformsArray.uniforms[amp_id];
    
    float4 position = float4(in.position, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * planeUniform.planeTransform * position;
    out.viewPosition = uniforms.modelViewMatrix * planeUniform.planeTransform * position;
    out.texCoord = in.texCoord;
    out.color = planeUniform.planeColor;
    out.planeDoProximity = planeUniform.planeDoProximity;

    return out;
}

fragment float4 fragmentShader(ColorInOutPlane in [[stage_in]])
{
    float4 color = in.color;
    if (in.planeDoProximity >= 0.5) {
        float cameraDistance = ((-in.viewPosition.z / in.viewPosition.w));
        float cameraX = (in.viewPosition.x);
        float cameraY = (in.viewPosition.y);
        float distFromCenterOfCamera = clamp((2.0 - sqrt(cameraX*cameraX+cameraY*cameraY)) / 2.0, 0.0, 0.9);
        cameraDistance = clamp((1.5 - sqrt(cameraDistance))/1.5, 0.0, 1.0);
        
        color *= pow(distFromCenterOfCamera * cameraDistance, 2.2);
        color.a = in.color.a;
    }
    
    if (color.a <= 0.0) {
        discard_fragment();
        return float4(0.0, 0.0, 0.0, 0.0);
    }
    return color;
}

vertex ColorInOut hudVertexShader(Vertex in [[stage_in]],
                                  ushort amp_id [[amplification_id]],
                                  constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]],
                                  constant PlaneUniform & planeUniform [[ buffer(BufferIndexPlaneUniforms) ]])
{
    ColorInOut out;
    Uniforms uniforms = uniformsArray.uniforms[amp_id];
    float4 position = float4(in.position, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * planeUniform.planeTransform * position;
    out.texCoord = in.texCoord;
    return out;
}

fragment half4 hudFragmentShader(ColorInOut in [[stage_in]],
                                 texture2d<half> in_tex [[texture(TextureIndexColor)]])
{
    constexpr sampler s(mag_filter::linear, min_filter::linear);
    return in_tex.sample(s, in.texCoord);
}

// from ALVR

// FFR_COMMON_SHADER_FORMAT

constant bool FFR_ENABLED [[ function_constant(ALVRFunctionConstantFfrEnabled) ]];
constant uint2 TARGET_RESOLUTION [[ function_constant(ALVRFunctionConstantFfrCommonShaderTargetResolution) ]];
constant uint2 OPTIMIZED_RESOLUTION [[ function_constant(ALVRFunctionConstantFfrCommonShaderOptimizedResolution) ]];
constant float2 EYE_SIZE_RATIO [[ function_constant(ALVRFunctionConstantFfrCommonShaderEyeSizeRatio) ]];
constant float2 CENTER_SIZE [[ function_constant(ALVRFunctionConstantFfrCommonShaderCenterSize) ]];
constant float2 CENTER_SHIFT [[ function_constant(ALVRFunctionConstantFfrCommonShaderCenterShift) ]];
constant float2 EDGE_RATIO [[ function_constant(ALVRFunctionConstantFfrCommonShaderEdgeRatio) ]];
constant bool CHROMAKEY_ENABLED [[ function_constant(ALVRFunctionConstantChromaKeyEnabled) ]];
constant float3 CHROMAKEY_COLOR [[ function_constant(ALVRFunctionConstantChromaKeyColor) ]];
constant float2 CHROMAKEY_LERP_DIST_RANGE [[ function_constant(ALVRFunctionConstantChromaKeyLerpDistRange) ]];
constant bool REALITYKIT_ENABLED [[ function_constant(ALVRFunctionConstantRealityKitEnabled) ]];
constant float2 VRR_SCREEN_SIZE [[ function_constant(ALVRFunctionConstantVRRScreenSize) ]];
constant float2 VRR_PHYS_SIZE [[ function_constant(ALVRFunctionConstantVRRPhysSize) ]];
constant float ENCODING_GAMMA [[ function_constant(ALVRFunctionConstantEncodingGamma) ]];
constant float4 ENCODING_YUV_TRANSFORM_0 [[ function_constant(ALVRFunctionConstantEncodingYUVTransform0) ]];
constant float4 ENCODING_YUV_TRANSFORM_1 [[ function_constant(ALVRFunctionConstantEncodingYUVTransform1) ]];
constant int VIDEO_FILTER [[ function_constant(ALVRFunctionConstantVideoFilter) ]]; // 0 bilinear, 1 bicubic, 2 FSR
constant float VIDEO_SHARPEN [[ function_constant(ALVRFunctionConstantVideoSharpen) ]];
constant float4 ENCODING_YUV_TRANSFORM_2 [[ function_constant(ALVRFunctionConstantEncodingYUVTransform2) ]];
constant float4 ENCODING_YUV_TRANSFORM_3 [[ function_constant(ALVRFunctionConstantEncodingYUVTransform3) ]];

float2 TextureToEyeUV(float2 textureUV, bool isRightEye) {
    // flip distortion horizontally for right eye
    // left: x * 2; right: (1 - x) * 2
    return float2((textureUV.x + float(isRightEye) * (1. - 2. * textureUV.x)) * 2., textureUV.y);
}

float2 EyeToTextureUV(float2 eyeUV, bool isRightEye) {
    // left: x / 2; right 1 - (x / 2)
    return float2(eyeUV.x * 0.5 + float(isRightEye) * (1. - eyeUV.x), eyeUV.y);
}

// DECOMPRESS_AXIS_ALIGNED_FRAGMENT_SHADER
float2 decompressAxisAlignedCoord(float2 uv) {
    bool isRightEye = uv.x > 0.5;
    float2 eyeUV = TextureToEyeUV(uv, isRightEye);

    const float2 c0 = (1. - CENTER_SIZE) * 0.5;
    const float2 c1 = (EDGE_RATIO - 1.) * c0 * (CENTER_SHIFT + 1.) / EDGE_RATIO;
    const float2 c2 = (EDGE_RATIO - 1.) * CENTER_SIZE + 1.;

    const float2 loBound = c0 * (CENTER_SHIFT + 1.);
    const float2 hiBound = c0 * (CENTER_SHIFT - 1.) + 1.;
    float2 underBound = float2(eyeUV.x < loBound.x, eyeUV.y < loBound.y);
    float2 inBound = float2(loBound.x < eyeUV.x && eyeUV.x < hiBound.x,
                        loBound.y < eyeUV.y && eyeUV.y < hiBound.y);
    float2 overBound = float2(eyeUV.x > hiBound.x, eyeUV.y > hiBound.y);

    float2 center = (eyeUV - c1) * EDGE_RATIO / c2;

    const float2 loBoundC = c0 * (CENTER_SHIFT + 1.) / c2;
    const float2 hiBoundC = c0 * (CENTER_SHIFT - 1.) / c2 + 1.;

    float2 leftEdge = (-(c1 + c2 * loBoundC) / loBoundC +
                    sqrt(((c1 + c2 * loBoundC) / loBoundC) * ((c1 + c2 * loBoundC) / loBoundC) +
                        4. * c2 * (1. - EDGE_RATIO) / (EDGE_RATIO * loBoundC) * eyeUV)) /
                    (2. * c2 * (1. - EDGE_RATIO)) * (EDGE_RATIO * loBoundC);
    float2 rightEdge =
        (-(c2 - EDGE_RATIO * c1 - 2. * EDGE_RATIO * c2 + c2 * EDGE_RATIO * (1. - hiBoundC) +
        EDGE_RATIO) /
            (EDGE_RATIO * (1. - hiBoundC)) +
        sqrt(((c2 - EDGE_RATIO * c1 - 2. * EDGE_RATIO * c2 + c2 * EDGE_RATIO * (1. - hiBoundC) +
                EDGE_RATIO) /
            (EDGE_RATIO * (1. - hiBoundC))) *
                ((c2 - EDGE_RATIO * c1 - 2. * EDGE_RATIO * c2 +
                    c2 * EDGE_RATIO * (1. - hiBoundC) + EDGE_RATIO) /
                (EDGE_RATIO * (1. - hiBoundC))) -
            4. * ((c2 * EDGE_RATIO - c2) * (c1 - hiBoundC + hiBoundC * c2) /
                        (EDGE_RATIO * (1. - hiBoundC) * (1. - hiBoundC)) -
                    eyeUV * (c2 * EDGE_RATIO - c2) / (EDGE_RATIO * (1. - hiBoundC))))) /
        (2. * c2 * (EDGE_RATIO - 1.)) * (EDGE_RATIO * (1. - hiBoundC));

    // todo: idk why these clamps are necessary
    float2 uncompressedUV = clamp(underBound * leftEdge, float2(0, 0), float2(1, 1)) + clamp(inBound * center, float2(0, 0), float2(1, 1)) + clamp(overBound * rightEdge, float2(0, 0), float2(1, 1));
    return EyeToTextureUV(uncompressedUV * EYE_SIZE_RATIO, isRightEye);
}

// VERTEX_SHADER

ColorInOut videoFrameVertexShaderCommon(ushort vertexID [[vertex_id]],
                               int which,
                               matrix_float4x4 projectionMatrix,
                               matrix_float4x4 modelViewMatrixFrame,
                               simd_float4 tangents)
{
    ColorInOut out;

    float2 uv = float2(float((vertexID << ushort(1)) & 2u) * 0.5, 1.0 - (float(vertexID & ushort(2)) * 0.5));
    float4 position = float4((uv * float2(2.0, -2.0)) + float2(-1.0, 1.0), -500.0, 1.0);

    if (position.x < 1.0) {
        position.x *= tangents[0] * 500.0;
    }
    else {
        position.x *= tangents[1] * 500.0;
    }
    if (position.y < 1.0) {
        position.y *= tangents[3] * 500.0;
    }
    else {
        position.y *= tangents[2] * 500.0;
    }
    out.position = projectionMatrix * modelViewMatrixFrame * position;
    if (which == 0) {
        out.texCoord = float2((uv.x * 0.5), uv.y);
    } else {
        out.texCoord = float2((uv.x * 0.5) + 0.5,  uv.y);
    }

    return out;
}

vertex ColorInOut videoFrameVertexShader(ushort vertexID [[vertex_id]],
                               ushort amp_id [[amplification_id]],
                               constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]])
{
    int which = (vertexID >= 4 ? 1 : 0) + amp_id;
    Uniforms uniforms = uniformsArray.uniforms[which];
    
    return videoFrameVertexShaderCommon(vertexID, which, uniforms.projectionMatrix, uniforms.modelViewMatrixFrame, uniforms.tangents);
}

float3 NonlinearToLinearRGB(float3 color) {
    const float DIV12 = 1. / 12.92;
    const float DIV1 = 1. / 1.055;
    const float THRESHOLD = 0.04045;
    const float3 GAMMA = float3(2.4);
        
    float3 condition = float3(color.r < THRESHOLD, color.g < THRESHOLD, color.b < THRESHOLD);
    float3 lowValues = color * DIV12;
    float3 highValues = pow((color + 0.055) * DIV1, GAMMA);
    return condition * lowValues + (1.0 - condition) * highValues;
}

float3 EncodingNonlinearToLinearRGB(float3 color, float gamma) {
    float3 ret;
    ret.r = color.r < 0.0 ? color.r : pow(color.r, gamma);
    ret.g = color.g < 0.0 ? color.g : pow(color.g, gamma);
    ret.b = color.b < 0.0 ? color.b : pow(color.b, gamma);
    return ret;
}

half3 NonlinearToLinearRGB_half(half3 color) {
    const half DIV12 = 1. / 12.92;
    const half DIV1 = 1. / 1.055;
    const half THRESHOLD = 0.04045;
    const half GAMMA = 2.4;
      
    return half3(color.r < THRESHOLD ? (color.r * DIV12) : pow((color.r + 0.055h) * DIV1, GAMMA), color.g < THRESHOLD ? (color.g * DIV12) : pow((color.g + 0.055h) * DIV1, GAMMA), color.b < THRESHOLD ? (color.b * DIV12) : pow((color.b + 0.055h) * DIV1, GAMMA));
}

half3 EncodingNonlinearToLinearRGB_half(half3 color, half gamma) {
    half3 ret;
    ret.r = color.r < 0.0 ? color.r : pow(color.r, gamma);
    ret.g = color.g < 0.0 ? color.g : pow(color.g, gamma);
    ret.b = color.b < 0.0 ? color.b : pow(color.b, gamma);
    return ret;
}

half colorclose_hsv(half3 hsv, half3 keyHsv, half2 tol)
{
    // Saturation or value too low, don't consider it at all.
    if (hsv.b < 0.001 || hsv.g < 0.001) {
        return 1.0;
    }
    half3 weights = half3(4., 1., 2.);
    half tmp = length(weights * (keyHsv - hsv));
    if (tmp < tol.x)
      return 0.0;
   	else if (tmp < tol.y)
      return (tmp - tol.x)/(tol.y - tol.x);
   	else
      return 1.0;
}

half3 rgb2hsv(half3 rgb) {
    half Cmax = max(rgb.r, max(rgb.g, rgb.b));
    half Cmin = min(rgb.r, min(rgb.g, rgb.b));
    half delta = Cmax - Cmin;
    half3 hsv = half3(0., 0., Cmax);

    if(Cmax > Cmin) {
        hsv.y = delta / Cmax;

        if(rgb.r == Cmax) {
            hsv.x = (rgb.g - rgb.b) / delta;
        } else {
            if (rgb.g == Cmax) {
                hsv.x = 2. + (rgb.b - rgb.r) / delta;
            } else {
                hsv.x = 4. + (rgb.r - rgb.g) / delta;
            }
        }
        hsv.x = fract(hsv.x / 6.);
    }
    
    return hsv;
}

half4 videoFrameFragmentShader_common(half3 color_in) {
    half3 color = EncodingNonlinearToLinearRGB_half(color_in, ENCODING_GAMMA);
    
    half4 colorOut = half4(color.rgb, 1.0);
    if (CHROMAKEY_ENABLED) {
        half4 chromaKeyHSV = half4(rgb2hsv(half3(CHROMAKEY_COLOR)), 1.0);
        half4 newHSV = half4(rgb2hsv(color.rgb), 1.0);
        half mask = colorclose_hsv(newHSV.rgb, chromaKeyHSV.rgb, half2(CHROMAKEY_LERP_DIST_RANGE));
        if (!REALITYKIT_ENABLED && mask <= 0.0) {
            discard_fragment();
        }
        
        colorOut = half4((colorOut.rgb * mask) - (half3(CHROMAKEY_COLOR) * (1.0 - mask)), colorOut.a * mask);
        return colorOut;
        //return float4(color.rgb, mask);
    }
    else {
        return colorOut;
    }
    
    // Brighten the scene to examine blocking artifacts/smearing
    //color = pow(color, 1.0 / 2.4);

    /*const float3x3 linearToDisplayP3 = {
        float3(1.2249, -0.0420, -0.0197),
        float3(-0.2247, 1.0419, -0.0786),
        float3(0.0, 0.0, 1.0979),
    };*/
    
    //technically not accurate, since sRGB is below 1.0, but it makes colors pop a bit
    //color = linearToDisplayP3 * color;
}

// Catmull-Rom bicubic sampling from nine bilinear reads (the separable weights of the outer
// taps fold into the linear filter). Sharper than bilinear; the negative lobes can overshoot
// slightly, so the result is clamped to the texture's range.
static half4 sampleCatmullRom(texture2d<half> tex, sampler s, float2 uv) {
    float2 texSize = float2(tex.get_width(), tex.get_height());
    float2 samplePos = uv * texSize;
    float2 texPos1 = floor(samplePos - 0.5f) + 0.5f;
    float2 f = samplePos - texPos1;

    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);

    float2 w12 = w1 + w2;
    float2 offset12 = w2 / w12;

    float2 texPos0 = (texPos1 - 1.0f) / texSize;
    float2 texPos3 = (texPos1 + 2.0f) / texSize;
    float2 texPos12 = (texPos1 + offset12) / texSize;

    float4 result = 0.0f;
    result += float4(tex.sample(s, float2(texPos0.x, texPos0.y))) * w0.x * w0.y;
    result += float4(tex.sample(s, float2(texPos12.x, texPos0.y))) * w12.x * w0.y;
    result += float4(tex.sample(s, float2(texPos3.x, texPos0.y))) * w3.x * w0.y;
    result += float4(tex.sample(s, float2(texPos0.x, texPos12.y))) * w0.x * w12.y;
    result += float4(tex.sample(s, float2(texPos12.x, texPos12.y))) * w12.x * w12.y;
    result += float4(tex.sample(s, float2(texPos3.x, texPos12.y))) * w3.x * w12.y;
    result += float4(tex.sample(s, float2(texPos0.x, texPos3.y))) * w0.x * w3.y;
    result += float4(tex.sample(s, float2(texPos12.x, texPos3.y))) * w12.x * w3.y;
    result += float4(tex.sample(s, float2(texPos3.x, texPos3.y))) * w3.x * w3.y;
    return half4(clamp(result, 0.0f, 1.0f));
}

static half4 sampleVideo(texture2d<half> tex, sampler s, float2 uv) {
    return VIDEO_FILTER >= 1 ? sampleCatmullRom(tex, s, uv) : tex.sample(s, uv);
}

// One texel of the red channel, clamped to the texture like address::clamp_to_edge.
static float easuTexel(texture2d<half> tex, float2 p, float2 maxPos) {
    return float(tex.read(uint2(clamp(p, 0.0f, maxPos))).r);
}

// Direction and edge length from one texel's cross of neighbours (FsrEasuSetF): a is above, b left,
// c the texel itself, d right, e below. w is the texel's bilinear weight at the sample point.
static void easuSet(thread float2 &dir, thread float &len, float w,
                    float a, float b, float c, float d, float e) {
    float dc = d - c;
    float cb = c - b;
    float lenX = max(abs(dc), abs(cb));
    lenX = lenX > 0.0f ? 1.0f / lenX : 0.0f;
    float dirX = d - b;
    dir.x += dirX * w;
    lenX = saturate(abs(dirX) * lenX);
    len += lenX * lenX * w;

    float ec = e - c;
    float ca = c - a;
    float lenY = max(abs(ec), abs(ca));
    lenY = lenY > 0.0f ? 1.0f / lenY : 0.0f;
    float dirY = e - a;
    dir.y += dirY * w;
    lenY = saturate(abs(dirY) * lenY);
    len += lenY * lenY * w;
}

// One tap of the EASU kernel (FsrEasuTapF): a Lanczos-2 approximation stretched along the edge
// direction and squeezed across it.
static void easuTap(thread float &aC, thread float &aW, float2 off, float2 dir, float2 len2,
                    float lob, float clp, float c) {
    float2 v = float2(off.x * dir.x + off.y * dir.y, -off.x * dir.y + off.y * dir.x) * len2;
    float d2 = min(dot(v, v), clp);
    float wB = 2.0f / 5.0f * d2 - 1.0f;
    float wA = lob * d2 - 1.0f;
    wB *= wB;
    wA *= wA;
    wB = 25.0f / 16.0f * wB - (25.0f / 16.0f - 1.0f);
    float w = wA * wB;
    aC += c * w;
    aW += w;
}

// AMD FidelityFX Super Resolution 1 edge adaptive spatial upsampling (EASU) of one channel, the
// luma. Twelve texels around the sample point, in the layout
//       b c
//     e f g h
//     i j k l
//       n o
// give a local edge direction and length; the kernel then follows the edge, so diagonal and curved
// edges stay smooth where bilinear and bicubic filters staircase. The result is clamped to the four
// nearest texels, which keeps the negative lobes from ringing. Reads texels directly rather than
// through the sampler, as the original does with gathers.
static half easuLuma(texture2d<half> tex, float2 uv) {
    float2 texSize = float2(tex.get_width(), tex.get_height());
    float2 maxPos = texSize - 1.0f;
    float2 pp = uv * texSize - 0.5f;
    float2 fp = floor(pp);
    pp -= fp;

    float b = easuTexel(tex, fp + float2(0.0f, -1.0f), maxPos);
    float c = easuTexel(tex, fp + float2(1.0f, -1.0f), maxPos);
    float e = easuTexel(tex, fp + float2(-1.0f, 0.0f), maxPos);
    float f = easuTexel(tex, fp, maxPos);
    float g = easuTexel(tex, fp + float2(1.0f, 0.0f), maxPos);
    float h = easuTexel(tex, fp + float2(2.0f, 0.0f), maxPos);
    float i = easuTexel(tex, fp + float2(-1.0f, 1.0f), maxPos);
    float j = easuTexel(tex, fp + float2(0.0f, 1.0f), maxPos);
    float k = easuTexel(tex, fp + float2(1.0f, 1.0f), maxPos);
    float l = easuTexel(tex, fp + float2(2.0f, 1.0f), maxPos);
    float n = easuTexel(tex, fp + float2(0.0f, 2.0f), maxPos);
    float o = easuTexel(tex, fp + float2(1.0f, 2.0f), maxPos);

    float2 dir = 0.0f;
    float len = 0.0f;
    easuSet(dir, len, (1.0f - pp.x) * (1.0f - pp.y), b, e, f, g, j);
    easuSet(dir, len, pp.x * (1.0f - pp.y), c, f, g, h, k);
    easuSet(dir, len, (1.0f - pp.x) * pp.y, f, i, j, k, n);
    easuSet(dir, len, pp.x * pp.y, g, j, k, l, o);

    float dirR = dir.x * dir.x + dir.y * dir.y;
    bool zero = dirR < 1.0f / 32768.0f;
    dirR = zero ? 1.0f : dirR;
    dir.x = zero ? 1.0f : dir.x;
    dir *= rsqrt(dirR);
    len = len * 0.5f;
    len *= len;
    float stretch = (dir.x * dir.x + dir.y * dir.y) / max(abs(dir.x), abs(dir.y));
    float2 len2 = float2(1.0f + (stretch - 1.0f) * len, 1.0f - 0.5f * len);
    float lob = 0.5f + ((1.0f / 4.0f - 0.04f) - 0.5f) * len;
    float clp = 1.0f / lob;

    float aC = 0.0f;
    float aW = 0.0f;
    easuTap(aC, aW, float2(0.0f, -1.0f) - pp, dir, len2, lob, clp, b);
    easuTap(aC, aW, float2(1.0f, -1.0f) - pp, dir, len2, lob, clp, c);
    easuTap(aC, aW, float2(-1.0f, 1.0f) - pp, dir, len2, lob, clp, i);
    easuTap(aC, aW, float2(0.0f, 1.0f) - pp, dir, len2, lob, clp, j);
    easuTap(aC, aW, float2(0.0f, 0.0f) - pp, dir, len2, lob, clp, f);
    easuTap(aC, aW, float2(-1.0f, 0.0f) - pp, dir, len2, lob, clp, e);
    easuTap(aC, aW, float2(1.0f, 1.0f) - pp, dir, len2, lob, clp, k);
    easuTap(aC, aW, float2(2.0f, 1.0f) - pp, dir, len2, lob, clp, l);
    easuTap(aC, aW, float2(2.0f, 0.0f) - pp, dir, len2, lob, clp, h);
    easuTap(aC, aW, float2(1.0f, 0.0f) - pp, dir, len2, lob, clp, g);
    easuTap(aC, aW, float2(1.0f, 2.0f) - pp, dir, len2, lob, clp, o);
    easuTap(aC, aW, float2(0.0f, 2.0f) - pp, dir, len2, lob, clp, n);

    float mn = min(min(f, g), min(j, k));
    float mx = max(max(f, g), max(j, k));
    return half(clamp(aC / aW, mn, mx));
}

// Contrast adaptive sharpening (after AMD FidelityFX CAS) of one channel: the center value
// against its four neighbours one texel away in the video texture. The sharpening weight shrinks
// where the neighbourhood already has high contrast, so edges do not ring and flat areas are
// left alone. strength 0...1 goes from CAS's weakest (-1/8) to its strongest (-1/5) lobe.
static half sharpenChannel(texture2d<half> tex, sampler s, float2 uv, half center, float strength) {
    float2 t = 1.0f / float2(tex.get_width(), tex.get_height());
    half n = tex.sample(s, uv + float2(0.0f, -t.y)).r;
    half so = tex.sample(s, uv + float2(0.0f, t.y)).r;
    half e = tex.sample(s, uv + float2(t.x, 0.0f)).r;
    half w = tex.sample(s, uv + float2(-t.x, 0.0f)).r;
    half mn = min(center, min(min(n, so), min(e, w)));
    half mx = max(center, max(max(n, so), max(e, w)));
    half amp = sqrt(saturate(min(mn, 1.0h - mx) / max(mx, 0.001h)));
    half weight = amp * (-1.0h / mix(8.0h, 5.0h, half(saturate(strength))));
    return saturate((center + (n + so + e + w) * weight) / (1.0h + 4.0h * weight));
}

// FrameExtrapolation: where a frame extrapolated fx.alpha source intervals ahead shows the decoded
// frame's picture at uv (frame texture coordinates, foveated): p with p + alpha * m(p) = uv, found by
// fixed point iteration, never across the eye boundary. The same lookup as fxWarpPlane in
// FrameExtrapolation.metal, which the offline test bench measures.
float2 frameExtrapolationCoord(texture2d<float> motion, constant FrameExtrapolationUniform &fx, float2 uv) {
    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);
    const float2 y = uv * fx.frameSize;
    const float eyeWidth = fx.frameSize.x * 0.5;
    const int eye = y.x >= eyeWidth ? 1 : 0;
    const float2 grid = float2(2 * fx.cellsPerEye, fx.gridRows);
    float2 p = y;
    for (int i = 0; i < fx.iterations; i++) {
        float gx = clamp((p.x - float(eye) * eyeWidth) / fx.cellPixels, 0.5, float(fx.cellsPerEye) - 0.5) + float(eye * fx.cellsPerEye);
        float gy = clamp(p.y / fx.cellPixels, 0.5, grid.y - 0.5);
        p = y - fx.alpha * motion.sample(linearSampler, float2(gx, gy) / grid).rg;
    }
    p.x = clamp(p.x, float(eye) * eyeWidth + 0.5, float(eye + 1) * eyeWidth - 0.5);
    p.y = clamp(p.y, 0.5, fx.frameSize.y - 0.5);
    return p / fx.frameSize;
}

// The video frame at sampleCoord as Y, Cb, Cr, with the video filter and sharpening of the setting.
static half4 sampleVideoFrame(texture2d<half> in_tex_y, texture2d<half> in_tex_uv, sampler colorSampler, float2 sampleCoord) {
    // FSR: EASU on luma, where the detail is; chroma is stored at half resolution and gets the
    // bicubic filter.
    half4 ySample = VIDEO_FILTER == 2 ? half4(easuLuma(in_tex_y, sampleCoord))
                                      : sampleVideo(in_tex_y, colorSampler, sampleCoord);
    half4 uvSample = sampleVideo(in_tex_uv, colorSampler, sampleCoord);
    if (VIDEO_SHARPEN > 0.0f) {
        // Luma only: sharpening chroma adds color fringes and little perceived detail.
        ySample.r = sharpenChannel(in_tex_y, colorSampler, sampleCoord, ySample.r, VIDEO_SHARPEN);
    }
    return half4(ySample.r, uvSample.rg, 1.0f);
}

// Motion field at picture position p (bilinear between cell centres, inside the eye).
static float4 frameInterpolationField(texture2d<float> motion, constant FrameExtrapolationUniform &fx, float2 p, int eye) {
    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);
    const float eyeWidth = fx.frameSize.x * 0.5;
    const float2 grid = float2(2 * fx.cellsPerEye, fx.gridRows);
    float gx = clamp((p.x - float(eye) * eyeWidth) / fx.cellPixels, 0.5, float(fx.cellsPerEye) - 0.5) + float(eye * fx.cellsPerEye);
    float gy = clamp(p.y / fx.cellPixels, 0.5, grid.y - 0.5);
    return motion.sample(linearSampler, float2(gx, gy) / grid);
}

// "Predict with Streamer Vectors" (mode 5): where the frame shown (n) is sampled so that it looks
// fx.alpha of the way to the next frame (n+1), in its own view, along the next frame's field. That
// field lives in n+1's picture: a point c there was at c - f.ba - f.rg in n (.ba the head rotation,
// .rg the scene motion). With head rotation left to the reprojection, the content at y in the
// prediction came from p = y - alpha * rg(c) in n, where c = p + ba(c) + rg(c) = y + ba(c) +
// (1 - alpha) * rg(c), found by fixed point iteration.
static float2 frameInterpolationPredictCoord(texture2d<float> motion, constant FrameExtrapolationUniform &fx, float2 uv) {
    const float2 y = uv * fx.frameSize;
    const float eyeWidth = fx.frameSize.x * 0.5;
    const int eye = y.x >= eyeWidth ? 1 : 0;
    const float2 lo = float2(float(eye) * eyeWidth + 0.5, 0.5);
    const float2 hi = float2(float(eye + 1) * eyeWidth - 0.5, fx.frameSize.y - 0.5);
    const float t = fx.alpha;
    float4 f = frameInterpolationField(motion, fx, y, eye);
    for (int i = 0; i < 2; i++) {
        f = frameInterpolationField(motion, fx, y + f.ba + (1.0 - t) * f.rg, eye);
    }
    return clamp(y - t * f.rg, lo, hi) / fx.frameSize;
}

// Frame interpolation from both frames (setting "Interpolation: Blend Both Frames"), after the
// approach of OFXR Bridge's frame_synthesis.hlsl (ideas only, no code): the in-between at uv is the
// newer frame moved back and the previous frame moved forward along the scene motion, mixed by how
// far between them it lies. Where the two disagree (wrong motion, occlusion) it fades to a plain
// mix of both frames without scene motion, and where nothing moves (text, still panels) it is just
// the newer frame as decoded, so still detail looks exactly as on the real frames. The previous
// frame is mapped through the head rotation between the two (field .ba).
static half4 frameInterpolationBlend(texture2d<float> motion, constant FrameExtrapolationUniform &fx, float2 uv,
                                     texture2d<half> curY, texture2d<half> curUV,
                                     texture2d<half> prevY, texture2d<half> prevUV, sampler colorSampler) {
    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);
    const float2 y = uv * fx.frameSize;
    const float eyeWidth = fx.frameSize.x * 0.5;
    const int eye = y.x >= eyeWidth ? 1 : 0;
    const float2 lo = float2(float(eye) * eyeWidth + 0.5, 0.5);
    const float2 hi = float2(float(eye + 1) * eyeWidth - 0.5, fx.frameSize.y - 0.5);
    // How far back towards the previous frame: 0 = the newer frame, 1 = the previous one.
    const float a = -fx.alpha;

    // Still: the same picture in both frames once head rotation is mapped out, here and 2 pixels
    // to either side (text runs horizontally; a moving edge can cross a flat centre), and no
    // scene motion in the field here. Without the field's say a thin line moving ~20 px was torn
    // apart: at the place it belongs in the in-between both frames are flat and agree (seen on
    // streamer dumps, 2026-10-13). The field sample is the one read for the rotation anyway.
    //
    // Kept cheap as measured on streamer dumps against the real middle frame (2026-10-15): one
    // still tap at the centre, one iteration less with the field read here as its first step, no
    // extra field read at the end, and the fallback read only where it is used, keep the quality
    // (within 0.03 dB) at 30-40 % fewer video and 50-60 % fewer field samples; confidence x3
    // instead of x6 was 0.05-1 dB better.
    const float4 fieldHere = frameInterpolationField(motion, fx, y, eye);
    const float2 rotationHere = fieldHere.ba;
    const float2 yPrev = clamp(y - rotationHere, lo, hi);
    if (abs(fieldHere.r) + abs(fieldHere.g) < 0.75) {
        const float c = float(curY.sample(linearSampler, uv).r);
        const float q = float(prevY.sample(linearSampler, yPrev / fx.frameSize).r);
        if (abs(c - q) <= 3.0 / 255.0) {
            return sampleVideoFrame(curY, curUV, colorSampler, uv);
        }
    }

    // Where the content seen at y is in the newer frame (x) and in the previous one.
    float4 f = fieldHere;
    float2 x = y + a * f.rg;
    for (int i = 2; i < fx.iterations; i++) {
        f = frameInterpolationField(motion, fx, x, eye);
        x = y + a * f.rg;
    }
    const float2 xCur = clamp(x, lo, hi);
    const float2 xPrev = clamp(x - f.ba - f.rg, lo, hi);
    const half4 cur = sampleVideoFrame(curY, curUV, colorSampler, xCur / fx.frameSize);
    const half4 prev = half4(prevY.sample(linearSampler, xPrev / fx.frameSize).r,
                             prevUV.sample(linearSampler, xPrev / fx.frameSize).rg, 1.0h);
    const half4 moved = mix(cur, prev, half(a));
    const half confidence = saturate(1.0h - abs(cur.r - prev.r) * 3.0h);
    if (confidence >= 1.0h) {
        return moved;
    }

    // The fallback: both frames without scene motion, mixed.
    const half4 curStill = half4(curY.sample(linearSampler, uv).r, curUV.sample(linearSampler, uv).rg, 1.0h);
    const half4 prevStill = half4(prevY.sample(linearSampler, yPrev / fx.frameSize).r,
                                  prevUV.sample(linearSampler, yPrev / fx.frameSize).rg, 1.0h);
    const half4 stable = mix(curStill, prevStill, half(a));
    return mix(stable, moved, confidence);
}

// Frame interpolation, "choose motion per pixel" without Blend Both Frames (mode 3): where the
// newer frame is sampled for the in-between at uv, moved back along the motion chosen for this
// 4x4 block. Choice 0, the streamer's field, is frameExtrapolationCoord as is; 2 (none) keeps HUD
// elements fixed to the view where they are. One video sample per pixel, like the plain move.
static float2 frameInterpolationSelectedCoord(texture2d<float> motion, texture2d<float> selected,
                                              constant FrameExtrapolationUniform &fx, float2 uv) {
    const float2 y = uv * fx.frameSize;
    const uint2 cell = min(uint2(y / float(FX_SELECT_BLOCK)), uint2(selected.get_width() - 1, selected.get_height() - 1));
    const int chosen = int(selected.read(cell).r + 0.5);
    if (chosen == 0) {
        return frameExtrapolationCoord(motion, fx, uv);
    }
    const float eyeWidth = fx.frameSize.x * 0.5;
    const int eye = y.x >= eyeWidth ? 1 : 0;
    const float2 lo = float2(float(eye) * eyeWidth + 0.5, 0.5);
    const float2 hi = float2(float(eye + 1) * eyeWidth - 0.5, fx.frameSize.y - 0.5);
    const float a = -fx.alpha;
    const float2 rotation = frameInterpolationField(motion, fx, y, eye).ba;
    const float c = fx.cellPixels;
    const float2 offset = chosen == 3 ? float2(-c, 0) : chosen == 4 ? float2(c, 0) : chosen == 5 ? float2(0, -c) : float2(0, c);
    const float2 t = chosen == 1 ? rotation
                   : chosen == 2 ? float2(0.0)
                   : frameInterpolationField(motion, fx, y + offset, eye).rg + rotation;
    return clamp(y + a * (t - rotation), lo, hi) / fx.frameSize;
}

// Frame interpolation with "choose motion per pixel": the motion chosen for this 4x4 block
// (fxSelectMotion in FrameExtrapolation.metal, made once per frame), evaluated here per pixel so
// it stays smooth across blocks. Choice 0, the streamer's field, is frameInterpolationBlend as is;
// for the others the newer frame moved back and the previous one forward along the chosen motion,
// mixed by how far between them the display lies.
static half4 frameInterpolationSelected(texture2d<float> motion, texture2d<float> selected, constant FrameExtrapolationUniform &fx,
                                        float2 uv, texture2d<half> curY, texture2d<half> curUV,
                                        texture2d<half> prevY, texture2d<half> prevUV, sampler colorSampler) {
    constexpr sampler linearSampler(filter::linear, address::clamp_to_edge);
    const float2 y = uv * fx.frameSize;
    const uint2 cell = min(uint2(y / float(FX_SELECT_BLOCK)), uint2(selected.get_width() - 1, selected.get_height() - 1));
    const int chosen = int(selected.read(cell).r + 0.5);
    if (chosen == 0) {
        return frameInterpolationBlend(motion, fx, uv, curY, curUV, prevY, prevUV, colorSampler);
    }
    const float eyeWidth = fx.frameSize.x * 0.5;
    const int eye = y.x >= eyeWidth ? 1 : 0;
    const float2 lo = float2(float(eye) * eyeWidth + 0.5, 0.5);
    const float2 hi = float2(float(eye + 1) * eyeWidth - 0.5, fx.frameSize.y - 0.5);
    const float a = -fx.alpha;
    const float2 rotation = frameInterpolationField(motion, fx, y, eye).ba;
    const float c = fx.cellPixels;
    const float2 offset = chosen == 3 ? float2(-c, 0) : chosen == 4 ? float2(c, 0) : chosen == 5 ? float2(0, -c) : float2(0, c);
    const float2 t = chosen == 1 ? rotation
                   : chosen == 2 ? float2(0.0)
                   : frameInterpolationField(motion, fx, y + offset, eye).rg + rotation;
    const float2 xCur = clamp(y + a * (t - rotation), lo, hi);
    const float2 xPrev = clamp(xCur - t, lo, hi);
    const half4 cur = sampleVideoFrame(curY, curUV, colorSampler, xCur / fx.frameSize);
    const half4 prev = half4(prevY.sample(linearSampler, xPrev / fx.frameSize).r,
                             prevUV.sample(linearSampler, xPrev / fx.frameSize).rg, 1.0h);
    return mix(cur, prev, half(a));
}

fragment half4 videoFrameFragmentShader_YpCbCrBiPlanar(ColorInOut in [[stage_in]], texture2d<half> in_tex_y [[texture(0)]], texture2d<half> in_tex_uv [[texture(1)]],
                                                       texture2d<float> fxMotion [[texture(2)]],
                                                       texture2d<half> prev_tex_y [[texture(3)]],
                                                       texture2d<half> prev_tex_uv [[texture(4)]],
                                                       texture2d<float> fxSelected [[texture(5)]],
                                                       constant FrameExtrapolationUniform &fx [[buffer(BufferIndexFrameExtrapolation)]]) {

    float2 sampleCoord;
    if (FFR_ENABLED) {
        sampleCoord = decompressAxisAlignedCoord(in.texCoord);
    } else {
        sampleCoord = in.texCoord;
    }

    constexpr sampler colorSampler(mip_filter::none,
                                   mag_filter::linear,
                                   min_filter::linear,
                                   address::clamp_to_edge);
    half4 ycbcr;
    if (fx.alpha > 0.0f && fx.blendPrevious == 5) {
        ycbcr = sampleVideoFrame(in_tex_y, in_tex_uv, colorSampler, frameInterpolationPredictCoord(fxMotion, fx, sampleCoord));
    } else if (fx.alpha < 0.0f && fx.blendPrevious == 2) {
        ycbcr = frameInterpolationSelected(fxMotion, fxSelected, fx, sampleCoord, in_tex_y, in_tex_uv, prev_tex_y, prev_tex_uv, colorSampler);
    } else if (fx.alpha != 0.0f && fx.blendPrevious == 3) {
        ycbcr = sampleVideoFrame(in_tex_y, in_tex_uv, colorSampler, frameInterpolationSelectedCoord(fxMotion, fxSelected, fx, sampleCoord));
    } else if (fx.alpha < 0.0f && fx.blendPrevious != 0) {
        ycbcr = frameInterpolationBlend(fxMotion, fx, sampleCoord, in_tex_y, in_tex_uv, prev_tex_y, prev_tex_uv, colorSampler);
    } else {
        if (fx.alpha != 0.0f) {
            sampleCoord = frameExtrapolationCoord(fxMotion, fx, sampleCoord);
        }
        ycbcr = sampleVideoFrame(in_tex_y, in_tex_uv, colorSampler, sampleCoord);
    }
    
    const matrix_half4x4 transform = matrix_half4x4(
        half4(ENCODING_YUV_TRANSFORM_0),
        half4(ENCODING_YUV_TRANSFORM_1),
        half4(ENCODING_YUV_TRANSFORM_2),
        half4(ENCODING_YUV_TRANSFORM_3)
    );
    
    half3 rgb_uncorrect = half3((transform * ycbcr).rgb);
    half3 color = NonlinearToLinearRGB_half(rgb_uncorrect);
    
    return videoFrameFragmentShader_common(color);
}

// Apple's private formats (VideoToolbox frames): one texture per frame, or per eye's half with
// "Encode eyes separately" (right != left), sampled as linear RGB; a filter at the eye boundary
// only ever sees its own eye.
static half3 sampleSecret(texture2d<half> left, texture2d<half> right, bool split, sampler s, float2 uv) {
    if (split) {
        const bool r = uv.x >= 0.5;
        const float2 local = float2((uv.x - (r ? 0.5 : 0.0)) * 2.0, uv.y);
        return (r ? sampleVideo(right, s, local) : sampleVideo(left, s, local)).rgb;
    }
    return sampleVideo(left, s, uv).rgb;
}

// Luma for the comparisons of the blend: of the linear RGB these formats give, square-rooted so
// that its thresholds mean about what they mean on the gamma encoded planes of frameInterpolationBlend.
static half secretLuma(half3 c) {
    return sqrt(max(dot(c, half3(0.2126h, 0.7152h, 0.0722h)), 0.0h));
}

// frameInterpolationBlend for the private formats: the same in-between of both frames (still test
// with the field's say, motion back and forward, confidence, fallback without scene motion), on
// their RGB.
static half3 frameInterpolationBlendSecret(texture2d<float> motion, constant FrameExtrapolationUniform &fx, float2 uv,
                                           texture2d<half> curL, texture2d<half> curR,
                                           texture2d<half> prevL, texture2d<half> prevR, bool split, sampler colorSampler) {
    const float2 y = uv * fx.frameSize;
    const float eyeWidth = fx.frameSize.x * 0.5;
    const int eye = y.x >= eyeWidth ? 1 : 0;
    const float2 lo = float2(float(eye) * eyeWidth + 0.5, 0.5);
    const float2 hi = float2(float(eye + 1) * eyeWidth - 0.5, fx.frameSize.y - 0.5);
    const float a = -fx.alpha;

    // The same reduced sampling as frameInterpolationBlend (measured on streamer dumps there).
    const float4 fieldHere = frameInterpolationField(motion, fx, y, eye);
    const float2 rotationHere = fieldHere.ba;
    const float2 yPrev = clamp(y - rotationHere, lo, hi);
    if (abs(fieldHere.r) + abs(fieldHere.g) < 0.75) {
        const half3 curHere = sampleSecret(curL, curR, split, colorSampler, uv);
        const half q = secretLuma(sampleSecret(prevL, prevR, split, colorSampler, yPrev / fx.frameSize));
        if (abs(secretLuma(curHere) - q) <= 3.0h / 255.0h) {
            return curHere;
        }
    }

    float4 f = fieldHere;
    float2 x = y + a * f.rg;
    for (int i = 2; i < fx.iterations; i++) {
        f = frameInterpolationField(motion, fx, x, eye);
        x = y + a * f.rg;
    }
    const float2 xCur = clamp(x, lo, hi);
    const float2 xPrev = clamp(x - f.ba - f.rg, lo, hi);
    const half3 cur = sampleSecret(curL, curR, split, colorSampler, xCur / fx.frameSize);
    const half3 prev = sampleSecret(prevL, prevR, split, colorSampler, xPrev / fx.frameSize);
    const half3 moved = mix(cur, prev, half(a));
    const half confidence = saturate(1.0h - abs(secretLuma(cur) - secretLuma(prev)) * 3.0h);
    if (confidence >= 1.0h) {
        return moved;
    }

    const half3 stable = mix(sampleSecret(curL, curR, split, colorSampler, uv),
                             sampleSecret(prevL, prevR, split, colorSampler, yPrev / fx.frameSize), half(a));
    return mix(stable, moved, confidence);
}

// frameInterpolationSelected for the private formats: choice 0 (the streamer's field) is the
// blend above, the others both frames along the chosen motion, mixed.
static half3 frameInterpolationSelectedSecret(texture2d<float> motion, texture2d<float> selected, constant FrameExtrapolationUniform &fx,
                                              float2 uv, texture2d<half> curL, texture2d<half> curR,
                                              texture2d<half> prevL, texture2d<half> prevR, bool split, sampler colorSampler) {
    const float2 y = uv * fx.frameSize;
    const uint2 cell = min(uint2(y / float(FX_SELECT_BLOCK)), uint2(selected.get_width() - 1, selected.get_height() - 1));
    const int chosen = int(selected.read(cell).r + 0.5);
    if (chosen == 0) {
        return frameInterpolationBlendSecret(motion, fx, uv, curL, curR, prevL, prevR, split, colorSampler);
    }
    const float eyeWidth = fx.frameSize.x * 0.5;
    const int eye = y.x >= eyeWidth ? 1 : 0;
    const float2 lo = float2(float(eye) * eyeWidth + 0.5, 0.5);
    const float2 hi = float2(float(eye + 1) * eyeWidth - 0.5, fx.frameSize.y - 0.5);
    const float a = -fx.alpha;
    const float2 rotation = frameInterpolationField(motion, fx, y, eye).ba;
    const float c = fx.cellPixels;
    const float2 offset = chosen == 3 ? float2(-c, 0) : chosen == 4 ? float2(c, 0) : chosen == 5 ? float2(0, -c) : float2(0, c);
    const float2 t = chosen == 1 ? rotation
                   : chosen == 2 ? float2(0.0)
                   : frameInterpolationField(motion, fx, y + offset, eye).rg + rotation;
    const float2 xCur = clamp(y + a * (t - rotation), lo, hi);
    const float2 xPrev = clamp(xCur - t, lo, hi);
    return mix(sampleSecret(curL, curR, split, colorSampler, xCur / fx.frameSize),
               sampleSecret(prevL, prevR, split, colorSampler, xPrev / fx.frameSize), half(a));
}

// The in-between of a frame and the one before it, made ahead on the render queue as soon as the
// frame, its field and its choice are there (FrameExtrapolation.tryPrecompute), at the alpha of its
// first display. The display then only samples it (mode 4): interpolating in the render pass cost
// the in-between displays 7-7.7 ms of GPU against 2-3 for the others, and most deadline misses
// (2026-10-14). Stream space, as the render pass samples it; sqrt-encoded linear RGB in 10 bits.
kernel void fxPrecomputeInBetween(texture2d<half> curL [[texture(0)]],
                                  texture2d<half> curR [[texture(1)]],
                                  texture2d<half> prevL [[texture(2)]],
                                  texture2d<half> prevR [[texture(3)]],
                                  texture2d<float> motion [[texture(4)]],
                                  texture2d<float> selected [[texture(5)]],
                                  texture2d<half, access::write> out [[texture(6)]],
                                  constant FrameExtrapolationUniform &fx [[buffer(0)]],
                                  constant int &splitEyes [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    constexpr sampler colorSampler(mip_filter::none, mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    const float2 uv = (float2(gid) + 0.5) / float2(out.get_width(), out.get_height());
    const bool split = splitEyes != 0;
    const half3 c = fx.blendPrevious == 2
        ? frameInterpolationSelectedSecret(motion, selected, fx, uv, curL, curR, prevL, prevR, split, colorSampler)
        : frameInterpolationBlendSecret(motion, fx, uv, curL, curR, prevL, prevR, split, colorSampler);
    out.write(half4(sqrt(max(c, half3(0.0h))), 1.0h), gid);
}

fragment half4 videoFrameFragmentShader_SecretYpCbCrFormats(ColorInOut in [[stage_in]], texture2d<half> in_tex_y [[texture(0)]],
                                                           texture2d<float> fxMotion [[texture(2)]],
                                                           texture2d<half> prev_tex [[texture(3)]],
                                                           texture2d<half> prev_right_tex [[texture(4)]],
                                                           texture2d<float> fxSelected [[texture(5)]],
                                                           texture2d<half> right_tex_y [[texture(6)]],
                                                           texture2d<half> precomputed [[texture(7)]],
                                                           constant FrameExtrapolationUniform &fx [[buffer(BufferIndexFrameExtrapolation)]],
                                                           constant int &splitEyes [[buffer(BufferIndexSplitEyes)]]) {

    float2 sampleCoord;
    if (FFR_ENABLED) {
        sampleCoord = decompressAxisAlignedCoord(in.texCoord);
    } else {
        sampleCoord = in.texCoord;
    }

    constexpr sampler colorSampler(mip_filter::none,
                                   mag_filter::linear,
                                   min_filter::linear,
                                   address::clamp_to_edge);

    // Frame interpolation for VideoToolbox frames (HEVC, H.264, AV1): with Blend Both Frames the
    // in-between of both frames (the previous one's texture at 3, its right half at 4), with the
    // streamer's choice per block (mode 2) or the field as a whole (mode 1); otherwise the newer
    // frame moved back along the choice (mode 3) or the field.
    const bool split = splitEyes != 0;
    if (fx.alpha < 0.0f && fx.blendPrevious == 4) {
        // Made ahead by fxPrecomputeInBetween.
        const half3 c = precomputed.sample(colorSampler, sampleCoord).rgb;
        return videoFrameFragmentShader_common(c * c);
    }
    if (fx.alpha < 0.0f && fx.blendPrevious == 2) {
        return videoFrameFragmentShader_common(frameInterpolationSelectedSecret(
            fxMotion, fxSelected, fx, sampleCoord, in_tex_y, right_tex_y, prev_tex, prev_right_tex, split, colorSampler));
    }
    if (fx.alpha < 0.0f && fx.blendPrevious == 1) {
        return videoFrameFragmentShader_common(frameInterpolationBlendSecret(
            fxMotion, fx, sampleCoord, in_tex_y, right_tex_y, prev_tex, prev_right_tex, split, colorSampler));
    }
    if (fx.alpha > 0.0f && fx.blendPrevious == 5) {
        sampleCoord = frameInterpolationPredictCoord(fxMotion, fx, sampleCoord);
    } else if (fx.alpha != 0.0f && fx.blendPrevious == 3) {
        sampleCoord = frameInterpolationSelectedCoord(fxMotion, fxSelected, fx, sampleCoord);
    } else if (fx.alpha != 0.0f) {
        sampleCoord = frameExtrapolationCoord(fxMotion, fx, sampleCoord);
    }

    // Apple's private formats sample as RGB, so sharpening would need a luma of its own; only the
    // bicubic filter applies here, and the FSR setting falls back to it.
    return videoFrameFragmentShader_common(sampleSecret(in_tex_y, right_tex_y, split, colorSampler, sampleCoord));
}

// PyroWave decodes Y, Cb and Cr into separate planes, while the video pipeline samples biplanar
// pixel buffers like VideoToolbox produces. Runs once per decoded stripe, on the chroma rows
// starting at firstRow. See PyroWaveDecoder.swift.
kernel void pyrowaveInterleaveChroma(texture2d<float, access::read> cb [[texture(0)]],
                                     texture2d<float, access::read> cr [[texture(1)]],
                                     texture2d<float, access::write> cbcr [[texture(2)]],
                                     constant uint &firstRow [[buffer(0)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    uint2 coord = uint2(gid.x, gid.y + firstRow);
    if (coord.x >= cbcr.get_width() || coord.y >= cbcr.get_height()) {
        return;
    }
    cbcr.write(float4(cb.read(coord).r, cr.read(coord).r, 0.0, 0.0), coord);
}

fragment float4 videoFrameDepthFragmentShader(ColorInOut in [[stage_in]], texture2d<float> in_tex_y, texture2d<float> in_tex_uv) {
    return float4(0.0, 0.0, 0.0, 0.0);
}

struct CopyVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex CopyVertexOut copyVertexShader(constant VertexPosOnly* inArr [[buffer(BufferIndexMeshPositions)]], uint vertexID [[vertex_id]], constant rasterization_rate_map_data &vrr [[buffer(BufferIndexVRR)]]) {
    CopyVertexOut out;
    rasterization_rate_map_decoder map(vrr);
    
    // generate vertices in-shader, not sure if fast or slow tbh
    /*const uint gridSize = 128;

    ushort x = (vertexID >> 1) % gridSize;
    ushort y = (vertexID & 1) + (((vertexID >> 1) / gridSize) % (gridSize-1));
    ushort idx = (vertexID >= (gridSize-1)*gridSize*2) ? 1 : 0;
    float otherEye = idx ? 1.0 : 0.0;

    // Normalize coordinates to the range [0, 1]
    float2 uv = float2(
        (float(x) / float(gridSize - 1)),
        (float(y) / float(gridSize - 1))
    );

    // Normalize coordinates to the range [-1, 1]ish
    out.position = float4((uv * float2(2.0, -1.0)) + float2(-1.0, otherEye), otherEye, 1.0);
    out.uv = map.map_screen_to_physical_coordinates(uv * VRR_SCREEN_SIZE, idx);*/
    
    float2 uv = inArr[vertexID].position.xy;
    float otherEye = inArr[vertexID].position.z;
    ushort idx = (otherEye != 0.0) ? 1.0 : 0.0;
    out.position = float4((uv * float2(2.0, -1.0)) + float2(-1.0, otherEye), otherEye, 1.0);
    out.uv = map.map_screen_to_physical_coordinates(uv * VRR_SCREEN_SIZE, idx);
    
    return out;
}

//constant rasterization_rate_map_data &vrr [[buffer(BufferIndexVRR)]]
fragment half4 copyFragmentShader(CopyVertexOut in [[stage_in]], texture2d_array<half> in_tex) {
    constexpr sampler colorSampler(coord::pixel,
                    address::clamp_to_edge,
                    filter::linear);
    ushort idx = in.position.z != 0.0 ? 1 : 0;
    
    half4 color = in_tex.sample(colorSampler, in.uv, idx);
    //half4 color = half4(in.uv.x, in.uv.y, 0.0, 1.0);

    return color;
}
