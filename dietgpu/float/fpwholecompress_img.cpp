#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"

using namespace dietgpu;

template <FloatType FT>
void runBatchPointerTest(
    StackDeviceMemory& res,
    const std::string& inputFilePath, 
    const std::string& outputFilePath, 
    int probBits) {
    using FTI = FloatTypeInfo<FT>;
  
    std::ifstream inputFile(inputFilePath, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cerr << "Unable to open file: " << inputFilePath << std::endl;
        return;
    }
    inputFile.seekg(0, std::ios::end);
    size_t fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);
    auto numElements = fileSize / sizeof(typename FTI::WordT);
    std::vector<typename FTI::WordT> orig(numElements);
    inputFile.read(reinterpret_cast<char*>(orig.data()), fileSize);
    if (!inputFile) {
        std::cerr << "Error reading file." << std::endl;
        return;
    }
    inputFile.close();
    orig.resize(numElements);
  // run on a different stream to test stream assignment
    hipStream_t stream;
    hipStreamCreate(&stream);
    std::vector<uint32_t> batchSizes = {static_cast<uint32_t>(numElements)};

  int numInBatch = batchSizes.size();
  uint32_t totalSize = 0;
  uint32_t maxSize = 0;
  for (auto v : batchSizes) {
    totalSize += v;
    maxSize = std::max(maxSize, v);
  }

  auto maxCompressedSize = getMaxFloatCompressedSize(FT, maxSize);

//   auto orig = generateFloats<FT>(totalSize);
  auto orig_dev = res.copyAlloc(stream, orig);

  auto inPtrs = std::vector<const void*>(batchSizes.size());
  {
    uint32_t curOffset = 0;
    for (int i = 0; i < inPtrs.size(); ++i) {
      inPtrs[i] = (const typename FTI::WordT*)orig_dev.data() + curOffset;
      curOffset += batchSizes[i];
    }
  }

  auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * maxCompressedSize);

  auto encPtrs = std::vector<void*>(batchSizes.size());
  {
    for (int i = 0; i < inPtrs.size(); ++i) {
      encPtrs[i] = (uint8_t*)enc_dev.data() + i * maxCompressedSize;
    }
  }

  auto outBatchSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  auto compConfig =
      FloatCompressConfig(FT, ANSCodecConfig(probBits), false, true);

  floatCompress(
      res,
      compConfig,
      numInBatch,
      inPtrs.data(),
      batchSizes.data(),
      encPtrs.data(),
      outBatchSize_dev.data(),
      stream);

  // Decode data
  auto dec_dev = res.alloc<typename FTI::WordT>(stream, totalSize);

  auto decPtrs = std::vector<void*>(batchSizes.size());
  {
    uint32_t curOffset = 0;
    for (int i = 0; i < inPtrs.size(); ++i) {
      decPtrs[i] = (typename FTI::WordT*)dec_dev.data() + curOffset;
      curOffset += batchSizes[i];
    }
  }

  auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
  auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  auto decompConfig =
      FloatDecompressConfig(FT, ANSCodecConfig(probBits), false, true);

  floatDecompress(
      res,
      decompConfig,
      numInBatch,
      (const void**)encPtrs.data(),
      decPtrs.data(),
      batchSizes.data(),
      outSuccess_dev.data(),
      outSize_dev.data(),
      stream);

  auto outSuccess = outSuccess_dev.copyToHost(stream);
  auto outSize = outSize_dev.copyToHost(stream);

  for (int i = 0; i < outSuccess.size(); ++i) {
    EXPECT_TRUE(outSuccess[i]);
    EXPECT_EQ(outSize[i], batchSizes[i]);
  }

  auto dec = dec_dev.copyToHost(stream);
    std::ofstream outputFile(outputFilePath, std::ios::binary);
    outputFile.write(reinterpret_cast<const char*>(dec.data()), dec.size()*sizeof(typename FTI::WordT));
    outputFile.close();
}

// void runBatchPointerTest(
//     StackDeviceMemory& res,
//     FloatType ft,
//     int probBits,
//     const std::vector<uint32_t>& batchSizes) {
//   switch (ft) {
//     case FloatType::kFloat16:
//       runBatchPointerTest<FloatType::kFloat16>(res, probBits, batchSizes);
//       break;
//     case FloatType::kBFloat16:
//       runBatchPointerTest<FloatType::kBFloat16>(res, probBits, batchSizes);
//       break;
//     case FloatType::kFloat32:
//       runBatchPointerTest<FloatType::kFloat32>(res, probBits, batchSizes);
//       break;
//     default:
// //      CHECK(false);
//       assert(false && "This should never be true");
//       break;
//   }
// }

int main(int argc, char* argv[]) {
    auto res = makeStackMemory();
    int probBits = 10; 
    runBatchPointerTest<FloatType::kBFloat16>(res, argv[1], argv[2], probBits);
    return 0;
}
