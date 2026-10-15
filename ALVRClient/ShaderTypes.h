//
//  ShaderTypes.h
//

//
//  Header containing types and enum constants shared between Metal shaders and Swift/ObjC source
//
#ifndef ShaderTypes_h
#define ShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
typedef metal::int32_t EnumBackingType;
#else
#import <Foundation/Foundation.h>
typedef NSInteger EnumBackingType;
#endif

#include <simd/simd.h>

typedef NS_ENUM(EnumBackingType, BufferIndex)
{
    BufferIndexMeshPositions = 0,
    BufferIndexMeshGenerics  = 1,
    BufferIndexUniforms      = 2,
    BufferIndexPlaneUniforms = 3,
    BufferIndexEncodingUniforms = 4,
    BufferIndexVRR = 5,
    BufferIndexFrameExtrapolation = 6,
    BufferIndexSplitEyes = 7,
};

typedef NS_ENUM(EnumBackingType, VertexAttribute)
{
    VertexAttributePosition  = 0,
    VertexAttributeTexcoord  = 1,
};

typedef NS_ENUM(EnumBackingType, TextureIndex)
{
    TextureIndexColor    = 0,
};

typedef struct
{
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 modelViewMatrixFrame;
    matrix_float4x4 modelViewMatrix;
    simd_float4 tangents;
} Uniforms;

typedef struct
{
    Uniforms uniforms[2];
} UniformsArray;

typedef struct
{
    matrix_float4x4 planeTransform;
    simd_float4 planeColor;
    float planeDoProximity;
} PlaneUniform;

typedef struct
{
    matrix_float4x4 yuvTransform;
} EncodingUniform;

// The video fragment shader's view of FrameExtrapolation: the motion field of the frame being
// shown (scene motion per frame interval of the source, head rotation taken out, picture pixels,
// one vector per cell) and how far ahead to move it. alpha 0 shows the frame as decoded.
typedef struct
{
    simd_float2 frameSize;   // picture pixels, both eyes side by side
    float alpha;             // source frame intervals to extrapolate, 0 = off
    float cellPixels;        // picture pixels per cell of the field
    int cellsPerEye;
    int gridRows;
    int iterations;          // fixed point iterations inverting the field
    int blendPrevious;       // 1: the previous frame is bound (textures 3, 4) and the field's pair;
                             // 2: as 1, with the motion chosen per block at texture 5;
                             // 3: the newer frame moved along the choice only; 4 (Apple's private
                             // formats): the in-between made ahead (fxPrecomputeInBetween) at texture 7;
                             // 5: the frame moved forward by alpha along the NEXT frame's field
                             // ("Predict with Streamer Vectors")
} FrameExtrapolationUniform;

// "Choose motion per pixel": picture pixels per block of the choice (fxSelectMotion).
#define FX_SELECT_BLOCK 8

#endif /* ShaderTypes_h */

typedef NS_ENUM(EnumBackingType, ALVRFunctionConstant)
{
    ALVRFunctionConstantFfrEnabled = 100,
    ALVRFunctionConstantFfrCommonShaderTargetResolution = 101,
    ALVRFunctionConstantFfrCommonShaderOptimizedResolution = 102,
    ALVRFunctionConstantFfrCommonShaderEyeSizeRatio = 103,
    ALVRFunctionConstantFfrCommonShaderCenterSize = 104,
    ALVRFunctionConstantFfrCommonShaderCenterShift = 105,
    ALVRFunctionConstantFfrCommonShaderEdgeRatio = 106,
    ALVRFunctionConstantChromaKeyEnabled = 200,
    ALVRFunctionConstantChromaKeyColor = 201,
    ALVRFunctionConstantChromaKeyLerpDistRange = 202,
    ALVRFunctionConstantRealityKitEnabled = 300,
    ALVRFunctionConstantVRRScreenSize = 301,
    ALVRFunctionConstantVRRPhysSize = 302,
    ALVRFunctionConstantEncodingGamma = 303,
    ALVRFunctionConstantEncodingYUVTransform0 = 304,
    ALVRFunctionConstantEncodingYUVTransform1 = 305,
    ALVRFunctionConstantEncodingYUVTransform2 = 306,
    ALVRFunctionConstantEncodingYUVTransform3 = 307,
    ALVRFunctionConstantVideoFilter = 400,
    ALVRFunctionConstantVideoSharpen = 401,
};
