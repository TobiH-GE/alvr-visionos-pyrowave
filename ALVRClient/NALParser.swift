//
//  NALParser.swift
//  ALVRClient
//
//  Created by Max Thomas on 11/4/25.
//

import Foundation

let H264_NAL_TYPE_SPS = 7
let HEVC_NAL_TYPE_VPS: UInt8 = 32
let HEVC_NAL_TYPE_SPS: UInt8 = 33
let HEVC_NAL_TYPE_PPS: UInt8 = 34
let HEVC_NAL_TYPE_SEI: UInt8 = 39

public struct NaluIndex {
    var startOffset: Int
    var payloadStartOffset: Int
    var payloadSize: Int
    var threeByteHeader: Bool
}

// Function to parse NAL units and extract VPS, SPS, PPS, and SEI data
public func extractParameterSets(from nalData: UnsafeMutableBufferPointer<UInt8>) -> (vps: [UInt8]?, sps: [UInt8]?, pps: [UInt8]?, sei: [UInt8]?) {
    var vps: [UInt8]?
    var sps: [UInt8]?
    var pps: [UInt8]?
    var sei: [UInt8]? // = [0x4E, 0x01, 0x88, 0x06, 0,0,0,0,0, 0x10, 0x80]
    
    let nalDataNoBounds = nalData.baseAddress!
    
    var index = 0
    while index < nalData.count - 4 {
        // Find the start code (0x00000001 or 0x000001)
        if nalDataNoBounds[index] == 0 && nalDataNoBounds[index + 1] == 0 && nalDataNoBounds[index + 2] == 0 && nalDataNoBounds[index + 3] == 1 {
            // NAL unit starts after the start code
            let nalUnitStartIndex = index + 4
            var nalUnitEndIndex = index + 4
            
            // Find the next start code to determine the end of this NAL unit
            for nextIndex in nalUnitStartIndex..<nalData.count - 4 {
                if nalDataNoBounds[nextIndex] == 0 && nalDataNoBounds[nextIndex + 1] == 0 && nalDataNoBounds[nextIndex + 2] == 0 && nalDataNoBounds[nextIndex + 3] == 1 {
                    nalUnitEndIndex = nextIndex
                    break
                }
                nalUnitEndIndex = nalData.count // If no more start codes, this NAL unit goes to the end of the data
            }
            
            let nalUnitType = (nalDataNoBounds[nalUnitStartIndex] & 0x7E) >> 1 // Get NAL unit type (HEVC)
            let nalUnitData = nalData.extracting(nalUnitStartIndex..<nalUnitEndIndex)
            // MV-HEVC: the second view's SPS and PPS (nuh_layer_id 1) are not the base layer's,
            // see extractLayer1ParameterSets.
            let layerId = nalUnitStartIndex + 1 < nalData.count
                ? Int((nalDataNoBounds[nalUnitStartIndex] & 1) << 5 | (nalDataNoBounds[nalUnitStartIndex + 1] >> 3)) : 0

            print("Switch nalUnitType of: \(nalUnitType)")
            switch layerId == 0 ? nalUnitType : 0xff {
            case HEVC_NAL_TYPE_VPS:
                vps = [UInt8](nalUnitData)
            case HEVC_NAL_TYPE_SPS:
                sps = [UInt8](nalUnitData)
            case HEVC_NAL_TYPE_PPS:
                pps = [UInt8](nalUnitData)
            case HEVC_NAL_TYPE_SEI:
                sei = [UInt8](nalUnitData)
            default:
                break
            }
            
            index = nalUnitEndIndex
        } else {
            index += 1 // Move to the next byte if start code not found
        }
    }
    
    return (vps, sps, pps, sei)
}

// MV-HEVC (streamer setting "MV-HEVC"): the parameter sets of the second view (nuh_layer_id 1) in
// the decoder configuration, SPS before PPS; empty for an ordinary stream.
public func extractLayer1ParameterSets(from nalData: UnsafeMutableBufferPointer<UInt8>) -> [[UInt8]] {
    var sets: [[UInt8]] = []
    let b = nalData.baseAddress!
    var index = 0
    var starts: [Int] = []
    while index + 4 <= nalData.count {
        if b[index] == 0 && b[index + 1] == 0 && b[index + 2] == 0 && b[index + 3] == 1 {
            starts.append(index + 4)
            index += 4
        } else {
            index += 1
        }
    }
    for (k, start) in starts.enumerated() where start + 1 < nalData.count {
        let end = k + 1 < starts.count ? starts[k + 1] - 4 : nalData.count
        let type = (b[start] & 0x7E) >> 1
        let layer = Int((b[start] & 1) << 5 | (b[start + 1] >> 3))
        if layer == 1 && (type == HEVC_NAL_TYPE_SPS || type == HEVC_NAL_TYPE_PPS) && end > start {
            sets.append([UInt8](nalData.extracting(start..<end)))
        }
    }
    return sets.sorted { ($0[0] >> 1) < ($1[0] >> 1) }
}

// LHEVCDecoderConfigurationRecord (ISO/IEC 14496-15, 9.6.3) for the "lhvC" sample description
// atom: VideoToolbox decodes the second view of an MV-HEVC stream only with it (measured on a Mac
// 2026-10-16: without it the session returns view 0 only).
public func lhvCAtom(layer1Sets: [[UInt8]]) -> Data {
    // Version 1; min_spatial_segmentation_idc 0; parallelismType 0; one temporal layer, nested,
    // 4 byte NAL lengths.
    var d: [UInt8] = [1, 0xF0, 0x00, 0xFC, 0xC0 | (1 << 3) | (1 << 2) | 3]
    let types = [HEVC_NAL_TYPE_SPS, HEVC_NAL_TYPE_PPS].filter { t in layer1Sets.contains { ($0[0] & 0x7E) >> 1 == t } }
    d.append(UInt8(types.count))
    for t in types {
        let of = layer1Sets.filter { ($0[0] & 0x7E) >> 1 == t }
        d.append(0x80 | t)
        d += [UInt8(of.count >> 8), UInt8(of.count & 0xff)]
        for s in of {
            d += [UInt8(s.count >> 8), UInt8(s.count & 0xff)]
            d += s
        }
    }
    return Data(d)
}



// Based on https://webrtc.googlesource.com/src/+/refs/heads/main/common_video/h264/h264_common.cc
public func findNaluIndices(bufferBounded: UnsafeMutableBufferPointer<UInt8>) -> ([NaluIndex], Bool) {
    var elgibleForModifyInPlace = true
    guard bufferBounded.count >= /* kNaluShortStartSequenceSize */ 3 else {
        return ([], false)
    }
    
    var sequences = [NaluIndex]()
    
    let end = bufferBounded.count - /* kNaluShortStartSequenceSize */ 3
    var i = 0
    let buffer = Data(bytesNoCopy: bufferBounded.baseAddress!, count: bufferBounded.count, deallocator: .none) // ?? why is this faster
    while i < end {
        if buffer[i + 2] > 1 {
            i += 3
        } else if buffer[i + 2] == 1 {
            if buffer[i + 1] == 0 && buffer[i] == 0 {
                var index = NaluIndex(startOffset: i, payloadStartOffset: i + 3, payloadSize: 0, threeByteHeader: true)
                if index.startOffset > 0 && buffer[index.startOffset - 1] == 0 {
                    index.startOffset -= 1
                    index.threeByteHeader = false
                }
                else {
                    elgibleForModifyInPlace = false
                }
                
                if !sequences.isEmpty {
                    sequences[sequences.count - 1].payloadSize = index.startOffset - sequences.last!.payloadStartOffset
                }
                
                sequences.append(index)
            }
            
            i += 3
        } else {
            i += 1
        }
    }
    
    if !sequences.isEmpty {
        sequences[sequences.count - 1].payloadSize = bufferBounded.count - sequences.last!.payloadStartOffset
    }
    
    return (sequences, elgibleForModifyInPlace)
}
