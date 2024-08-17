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

void compressFloat32File(StackDeviceMemory& res, const std::string& inputFilePath, const std::string& tempFilePath, std::vector<uint32_t>& batchSizes, size_t& numElements, int probBits, uint32_t& maxCompressedSize, int& totalSize, std::vector<uint32_t>& compressedSize, int& numInBatch,hipStream_t stream) {
    auto compConfig = FloatCompressConfig(FloatType::kFloat32, ANSCodecConfig(probBits), false, true);
    std::ifstream inputFile(inputFilePath, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cerr << "Unable to open file: " << inputFilePath << std::endl;
        return;
    }
    inputFile.seekg(0, std::ios::end);
    size_t fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);
    numElements = fileSize / sizeof(float);
    std::vector<float> orig(numElements);
    inputFile.read(reinterpret_cast<char*>(orig.data()), fileSize);
    std::cout<<"fileSize:"<<fileSize<<std::endl;
    if (!inputFile) {
        std::cerr << "Error reading file." << std::endl;
        return;
    }
    inputFile.close();

    // 输出读取的 float 值
    // std::cout << "Contents of the float file:" << std::endl;
    // for (size_t i = 0; i < numElements; ++i) {
    //     std::cout << "orig[" << i << "] = " << orig[i] << std::endl;
    // }

    auto orig_dev = res.copyAlloc(stream, orig);
    batchSizes = {static_cast<uint32_t>(numElements)};
    std::cout<<"batchSizes:"<<batchSizes[0]<<std::endl;
    numInBatch = batchSizes.size();

    totalSize = 0;
    uint32_t maxSize = 0;
    for (auto v : batchSizes) {
      totalSize += v;
      maxSize = std::max(maxSize, v);
    }
    
    auto inPtrs = std::vector<const void*>(batchSizes.size());
    {
      uint32_t curOffset = 0;
      for (int i = 0; i < inPtrs.size(); ++i) {
        inPtrs[i] = (float*)orig_dev.data() + curOffset;
        curOffset += batchSizes[i]*sizeof(float);
      }
    }

    maxCompressedSize = getMaxFloatCompressedSize(FloatType::kFloat32, maxSize);
    std::cout<<"maxCompressedSize:"<<maxCompressedSize<<std::endl;
    auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * maxCompressedSize);

    auto encPtrs = std::vector<void*>(batchSizes.size());
    {
      for (int i = 0; i < inPtrs.size(); ++i) {
        encPtrs[i] = (uint8_t*)enc_dev.data() + i * maxCompressedSize;
      }
    }

    auto outBatchSize_dev = res.alloc<uint32_t>(stream, numInBatch);
    
    floatCompress(
        res,
        compConfig,
        numInBatch,
        inPtrs.data(),
        batchSizes.data(),
        encPtrs.data(),
        outBatchSize_dev.data(),
        stream);
    hipStreamSynchronize(stream);
    compressedSize = outBatchSize_dev.copyToHost(stream);
    std::vector<uint8_t> compressedHost(compressedSize[0]);
    hipMemcpy(compressedHost.data(), enc_dev.data(), compressedSize[0], hipMemcpyDeviceToHost);
    std::ofstream outputFile(tempFilePath, std::ios::binary);
    ASSERT_TRUE(outputFile.is_open()) << "Cannot open output file";
    outputFile.write(reinterpret_cast<const char*>(compressedHost.data()), compressedSize[0]);
    outputFile.close();
}

void decompressFloat32File(StackDeviceMemory& res, const std::string& tempFilePath, const std::string& outputFilePath, std::vector<uint32_t>& batchSizes, size_t& numElements, int probBits, uint32_t& maxCompressedSize, int& totalSize, std::vector<uint32_t>& compressedSize, int& numInBatch, hipStream_t stream) {
    std::ifstream inputFile(tempFilePath, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cerr << "Unable to open file: " << tempFilePath << std::endl;
        return;
    }
    std::vector<uint8_t> compressedData(compressedSize[0]);
    inputFile.read(reinterpret_cast<char*>(compressedData.data()), compressedSize[0]);
    inputFile.close();
    
    // // 输出读取的 uint8_t 值
    // std::cout << "Contents of the float file:"<<compressedSize[0] << std::endl;
    // for (size_t i = 0; i < compressedData.size(); ++i) {
    //     std::cout << "compressedData[" << i << "] = " << compressedData[i] << std::endl;
    // }

    auto compressedData_dev = res.alloc<uint8_t>(stream, compressedData.size());
    hipMemcpy(compressedData_dev.data(), compressedData.data(), compressedData.size(), hipMemcpyHostToDevice);
    auto encPtrs = std::vector<const void*>(batchSizes.size());
    for (int i = 0; i < encPtrs.size(); ++i) {
        encPtrs[i] = (uint8_t*)compressedData_dev.data() + i * maxCompressedSize;
    }

    // Decode data
    auto dec_dev = res.alloc<float>(stream, totalSize);
    auto decPtrs = std::vector<void*>(batchSizes.size());
    {
      uint32_t curOffset = 0;
      for (int i = 0; i < encPtrs.size(); ++i) {
        decPtrs[i] = (float*)dec_dev.data() + curOffset;
        curOffset += batchSizes[i]*sizeof(float);
      }
    }

    auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
    auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

    auto decompConfig =
        FloatDecompressConfig(FloatType::kFloat32, ANSCodecConfig(probBits), false, true);

    floatDecompress(
        res,
        decompConfig,
        1,
        (const void**)encPtrs.data(),
        decPtrs.data(),
        batchSizes.data(),
        outSuccess_dev.data(),
        outSize_dev.data(),
        stream);
    std::vector<float> decompressedHost(batchSizes[0]);
    hipMemcpy(decompressedHost.data(), dec_dev.data(), batchSizes[0]*sizeof(float), hipMemcpyDeviceToHost);
    std::ofstream outputFile(outputFilePath, std::ios::binary);
    ASSERT_TRUE(outputFile.is_open()) << "Cannot open output file";
    outputFile.write(reinterpret_cast<const char*>(decompressedHost.data()), decompressedHost.size()*sizeof(float));
    outputFile.close();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.img> <temp.ans> <output.img>" << std::endl;
        return 1;
    }
    auto res = makeStackMemory();
    size_t numElements;
    hipStream_t stream;
    hipStreamCreate(&stream);
    int probBits = 10; // 根据需要设置精度
    std::vector<uint32_t> batchSizes;
    uint32_t maxCompressedSize;
    std::vector<uint32_t> compressedSize;
    int numInBatch;
    int totalSize;
    try{
    compressFloat32File(res, argv[1], argv[2], batchSizes, numElements, probBits, maxCompressedSize, totalSize, compressedSize, numInBatch,stream);
    std::cout<<"OK!"<<std::endl;
    decompressFloat32File(res, argv[2], argv[3], batchSizes, numElements, probBits, maxCompressedSize, totalSize, compressedSize, numInBatch,stream);
    hipStreamDestroy(stream);
    std::cout << "Compression completed successfully." << std::endl;
    }catch (const std::exception& e) {
        std::cerr << "Error during compression: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
