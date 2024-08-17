/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <fstream>
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/utils/StackDeviceMemory.h"

using namespace dietgpu;
std::string inputFilePath;
std::vector<uint8_t> generateSymbols(int num, float lambda = 20.0f) {
  std::random_device rd;
  std::mt19937 gen(10);
  std::exponential_distribution<float> dist(lambda);

  auto out = std::vector<uint8_t>(num);
  for (auto& v : out) {
    auto sample = std::min(dist(gen), 1.0f);

    v = sample * 256.0;
    //std::cout<<static_cast<int>(v)<<" ";
  }

  return out;
}

std::vector<GpuMemoryReservation<uint8_t>> toDevice(
    StackDeviceMemory& res,
    const std::vector<std::vector<uint8_t>>& vs,
    hipStream_t stream) {
  auto out = std::vector<GpuMemoryReservation<uint8_t>>();

  for (auto& v : vs) {
    out.emplace_back(res.copyAlloc(stream, v, AllocType::Permanent));
  }

  return out;
}

std::vector<std::vector<uint8_t>> toHost(
    StackDeviceMemory& res,
    const std::vector<GpuMemoryReservation<uint8_t>>& vs,
    hipStream_t stream) {
  auto out = std::vector<std::vector<uint8_t>>();

  for (auto& v : vs) {
    out.emplace_back(v.copyToHost(stream));
  }

  return out;
}

std::vector<GpuMemoryReservation<uint8_t>> buffersToDevice(
    StackDeviceMemory& res,
    const std::vector<uint32_t>& sizes,
    hipStream_t stream) {
  auto out = std::vector<GpuMemoryReservation<uint8_t>>();

  for (auto& s : sizes) {
    out.emplace_back(res.alloc<uint8_t>(stream, s, AllocType::Permanent));
  }

  return out;
}

std::vector<std::vector<uint8_t>> genBatch(
    const std::vector<uint32_t>& sizes,
    double lambda) {
  auto out = std::vector<std::vector<uint8_t>>();

  for (auto s : sizes) {
    out.push_back(generateSymbols(s, lambda));
 //   std::cout<<std::endl;
  }

  return out;
}

void runBatchPointer(
    StackDeviceMemory& res,
    int prec,
    const 
    std::vector<uint32_t>& batchSizes,
    double lambda = 100.0) {
  // run on a different stream to test stream assignment
  //auto stream = CudaStream::makeNonBlocking();
  hipStream_t stream;
  hipStreamCreate(&stream);
  int numInBatch = batchSizes.size();
  uint32_t maxSize = 0;
  for (auto v : batchSizes) {
    maxSize = std::max(maxSize, v);
  }
  /*
  std::ifstream inputFile(inputFilePath, std::ios::binary | std::ios::ate);
  if (inputFile.fail()) {
      throw std::runtime_error("Cannot open input file");
  }
  std::streamsize fileSize = inputFile.tellg();
  std::vector<std::vector<uint8_t>> fileData(1);
  std::vector<uint8_t> file_host(fileSize);
  inputFile.read(reinterpret_cast<char*>(file_host.data()), fileSize);
  inputFile.seekg(0, std::ios::beg);
  fileData[0] = file_host;
  //auto devData = res.alloc<uint8_t>(stream, static_cast<uint32_t>(fileSize));
   
  inputFile.close();*/
  auto outBatchStride = getMaxCompressedSize(maxSize);

  auto batch_host = genBatch(batchSizes, lambda);
  //auto batch_host = fileData;
  auto batch_dev = toDevice(res, batch_host, stream);

  auto inPtrs = std::vector<const void*>(batchSizes.size());
  {
    for (int i = 0; i < inPtrs.size(); ++i) {
      inPtrs[i] = batch_dev[i].data();
    }
  }

  auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * outBatchStride);

  auto encPtrs = std::vector<void*>(batchSizes.size());
  for (int i = 0; i < inPtrs.size(); ++i) {
    encPtrs[i] = (uint8_t*)enc_dev.data() + i * outBatchStride;
  }

  auto outCompressedSize_dev = res.alloc<uint32_t>(stream, numInBatch);
  // auto encSize_0 = outCompressedSize_dev.copyToHost(stream);
  // 假设我们想输出第一个批次的压缩数据
/*std::vector<uint8_t> compressedData_0(encSize_0[0]);
hipMemcpy(compressedData_0.data(), encPtrs[0], encSize_0[0], hipMemcpyDeviceToHost);
for (size_t i = 0; i < compressedData_0.size(); ++i) {
  std::cout << std::hex << static_cast<int>(compressedData_0[i]) << " ";
}
  std::cout<<"encSize_0[0]:"<<encSize_0[0];
std::cout << std::endl;
*/  ansEncodeBatchPointer(
      res,
      ANSCodecConfig(prec, true),
      numInBatch,
      inPtrs.data(),
      batchSizes.data(),
      nullptr,
      encPtrs.data(),
      outCompressedSize_dev.data(),
      stream);
  auto encSize = outCompressedSize_dev.copyToHost(stream);
  std::cout<<"encSize[0]:"<<encSize[0]<<std::endl;
  for (auto v : encSize) {
    // Reported compressed sizes in bytes should be a multiple of 16 for aligned
    // packing
        //  std::cout<<"encSize:"<<v<<std::endl;
    EXPECT_EQ(v % 16, 0);
  }
  
/*// 2. 将压缩数据写入文件
std::vector<uint8_t> compressedData(encSize[0]);
hipMemcpy(compressedData.data(), encPtrs[0], encSize[0], hipMemcpyDeviceToHost);

std::ofstream outFile("compressed_data.bin", std::ios::binary);
if (!outFile) {
    // 处理错误
}
outFile.write(reinterpret_cast<const char*>(compressedData.data()), compressedData.size());
outFile.close();*/
 // std::cout<<encPtrs[0]<<" "<<encPtrs[1]<<" "<<encPtrs[2]<<std::endl;
  
  // 假设我们想输出第一个批次的压缩数据
//std::vector<uint8_t> compressedData(encSize[0]);
//hipMemcpy(compressedData.data(), encPtrs[0], encSize[0], hipMemcpyDeviceToHost);
/*for (size_t i = 0; i < compressedData.size(); ++i) {
  std::cout << std::hex << static_cast<int>(compressedData[i]) << " ";
}
std::cout << std::endl;*/
/*std::cout << "encSize[1]:"<<encSize[1]<<std::endl;
std::vector<uint8_t> compressedData_0(encSize[1]);
hipMemcpy(compressedData_0.data(), encPtrs[1], encSize[1], hipMemcpyDeviceToHost);
for (size_t i = 0; i < compressedData_0.size(); ++i) {
  std::cout << std::hex << static_cast<int>(compressedData_0[i]) << " ";
}
std::cout << std::endl;

  for(auto v : encPtrs) {
    std::cout<<static_cast<int>(*(uint8_t*)v)<<std::endl;
  }
  auto enc_host = toHost(res, enc_dev, stream);
  for(int i=0;i<enc_host.size();i++)
  for(int j=0;j<enc_host[0].size();j++)
  std::cout<<"enc_host:"<<static_cast<int>(enc_host[i][j])<<std::endl;
*/
  // Decode data
  //
//  hipStream_t stream;
//  hipStreamCreate(&stream);
  auto dec_dev = buffersToDevice(res, batchSizes, stream);
  //hipMemset(dec_dev.data(), 0, batchSizes);
// 假设 GpuMemoryReservation 类有 data() 和 size() 方法
/*for (auto& bufferReservation : dec_dev) {
    void* dataPtr = bufferReservation.data(); // 获取数据指针
    size_t dataSize = batchSizes.size(); // 获取数据大小

    // 创建一个与设备内存大小相同的主机端缓冲区
    std::vector<uint8_t> hostBuffer(dataSize);

    // 从设备内存复制数据到主机内存
    hipMemcpy(hostBuffer.data(), dataPtr, dataSize, hipMemcpyDeviceToHost);

    // 打印数据内容
    std::cout << "Device memory content: ";
    for (size_t i = 0; i < dataSize; ++i) {
        std::cout << std::hex << static_cast<int>(hostBuffer[i]) << " ";
    }
    std::cout << std::endl;
}*/
  auto decPtrs = std::vector<void*>(batchSizes.size());
  for (int i = 0; i < inPtrs.size(); ++i) {
    decPtrs[i] = dec_dev[i].data();
  }

  auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
  auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);
/*  auto outSuccess_0 = outSuccess_dev.copyToHost(stream);
  auto outSize_0 = outSize_dev.copyToHost(stream);

  for (int i = 0; i < outSuccess_0.size(); ++i) {
    EXPECT_TRUE(outSuccess_0[i]);
    EXPECT_EQ(outSize_0[i], batchSizes[i]);
  }
*/
// std::cout<<"1"<<std::endl;
/*  auto dec_host = toHost(res, dec_dev, stream);
  for(int i=0;i<batch_host.size();i++)
  for(int j=0;j<batch_host[0].size();j++)
  std::cout<<"batch_host:"<<static_cast<int>(batch_host[i][j])<<std::endl;
  for(int i=0;i<dec_host.size();i++)
  for(int j=0;j<dec_host[0].size();j++)
  std::cout<<"dec_host:"<<static_cast<int>(dec_host[i][j])<<std::endl;
*/
//  std::cout<<"OK!"<<std::endl; 
/*  // 3. 从文件读取数据
std::ifstream inFile("compressed_data.bin", std::ios::binary);
if (!inFile) {
    // 处理错误
}
std::vector<uint8_t> fileCompressedData(encSize[0]);
inFile.read(reinterpret_cast<char*>(fileCompressedData.data()), encSize[0]);
inFile.close();
std::cout<<std::endl<<"fileCompress:"<<std::endl;
  for(int i=0;i<fileCompressedData.size();i++)
  std::cout<<static_cast<int>(fileCompressedData[i])<<" ";
  std::cout<<std::endl;
// 4. 将读取的数据传输到设备内存
auto fileCompressedDev = res.alloc<uint8_t>(stream, encSize[0]);
hipMemcpy(fileCompressedDev.data(), fileCompressedData.data(), encSize[0], hipMemcpyHostToDevice);
//  auto file_dev = res.alloc<uint8_t>(stream, numInBatch * outBatchStride);

  auto filePtrs = std::vector<void*>(batchSizes.size());
  for (int i = 0; i < inPtrs.size(); ++i) {
    filePtrs[i] = (uint8_t*)fileCompressedDev.data() + i * outBatchStride;
  }

// 5. 解压缩数据
// 假设您已经有了解压缩所需的其他参数和设备内存空间
auto dec_dev = buffersToDevice(res, batchSizes, stream);
auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

ansDecodeBatchPointer(
    res,
    ANSCodecConfig(prec, true),
    numInBatch,
    (const void**)filePtrs.data(), // 使用从文件读取的数据
    decPtrs.data(),
    batchSizes.data(),
    outSuccess_dev.data(),
    outSize_dev.data(),
    stream);
*/
  ansDecodeBatchPointer(
      res,
      ANSCodecConfig(prec, true),
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
  auto dec_host = toHost(res, dec_dev, stream);
/*  std::cout<<"batch_host:"<<std::endl;
  for(int i=0;i<batch_host.size();i++)
  for(int j=0;j<batch_host[0].size();j++)
  //std::cout<<"batch_host:"<<static_cast<int>(batch_host[i][j])<<" ";
  std::cout<<static_cast<int>(batch_host[i][j])<<" ";
  std::cout<<std::endl<<"dec_host:"<<std::endl;
  for(int i=0;i<dec_host.size();i++)
  for(int j=0;j<dec_host[0].size();j++)
  std::cout<<static_cast<int>(dec_host[i][j])<<" ";
  //std::cout<<"dec_host:"<<static_cast<int>(dec_host[i][j])<<" ";
  std::cout<<std::endl;*/
/*
  for(int i=0;i<batch_host.size();i++)
  for(int j=0;j<batch_host[0].size();j++)
  if(static_cast<int>(batch_host[i][j])!=static_cast<int>(dec_host[i][j]))
 //std::cout<<"batch_host:"<<static_cast<int>(batch_host[i][j])<<" ";
  std::cout<<"i:"<<i<<"j:"<<j<<" "<<static_cast<int>(batch_host[i][j])<<" "<<static_cast<int>(dec_host[i][j])<<std::endl;
*/
  //dec_host = toHost(res, dec_dev, stream);
  EXPECT_EQ(batch_host, dec_host);
}

void runBatchStride(
    StackDeviceMemory& res,
    int prec,
    int numInBatch,
    int inBatchSize,
    double lambda = 100.0) {
  // run on a different stream to test stream assignment
  //auto stream = CudaStream::makeNonBlocking();
  hipStream_t stream;
  //hipStreamCreate(&stream);
  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  auto orig = generateSymbols(numInBatch * inBatchSize, lambda);
  auto orig_dev = res.copyAlloc(stream, orig);

  int outBatchStride = getMaxCompressedSize(inBatchSize);

  auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * outBatchStride);

  auto outCompressedSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  ansEncodeBatchStride(
      res,
      ANSCodecConfig(prec, true),
      numInBatch,
      orig_dev.data(),
      inBatchSize,
      inBatchSize,
      nullptr,
      enc_dev.data(),
      outBatchStride,
      outCompressedSize_dev.data(),
      stream);

  auto encSize = outCompressedSize_dev.copyToHost(stream);
  for (auto v : encSize) {
    // Reported compressed sizes in bytes should be a multiple of 16 for aligned
    // packing
    EXPECT_EQ(v % 16, 0);
  }

  auto dec_dev = res.alloc<uint8_t>(stream, numInBatch * inBatchSize);
  auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
  auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

  // FIXME: Copy the compressed data to the host and truncate it to make
  // sure the compressed size is accurate
  ansDecodeBatchStride(
      res,
      ANSCodecConfig(prec, true),
      numInBatch,
      enc_dev.data(),
      outBatchStride,
      dec_dev.data(),
      inBatchSize,
      inBatchSize,
      outSuccess_dev.data(),
      outSize_dev.data(),
      stream);
  //std::cout<<"2"<<std::endl;
  auto outSuccess = outSuccess_dev.copyToHost(stream);
  auto outSize = outSize_dev.copyToHost(stream);

  for (auto s : outSuccess) {
    EXPECT_TRUE(s);
  }
  //std::cout<<"2"<<std::endl;
  for (auto s : outSize) {
    EXPECT_EQ(s, inBatchSize);
  }

  auto dec = dec_dev.copyToHost(stream);
  EXPECT_EQ(orig, dec);
}

TEST(ANSTest, ZeroSized) {
  auto res = makeStackMemory();
  runBatchPointer(res, 10, {1000000}, 10.0);
  //std::cout<<"4"<<std::endl;
  //runBatchPointer(res, 10, {30}, 10.0);
}
/*
TEST(ANSTest, BatchPointer) {
  auto res = makeStackMemory();

  for (auto prec : {9, 10, 11}) {
    for (auto lambda : {1.0, 10.0, 100.0, 1000.0}) {
      runBatchPointer(res, prec, {1}, lambda);
      runBatchPointer(res, prec, {1, 1}, lambda);
      runBatchPointer(res, prec, {4096, 4095, 4096}, lambda);
      runBatchPointer(res, prec, {1234, 2345, 3456}, lambda);
      runBatchPointer(res, prec, {10000, 10013, 10000}, lambda);
    }
  }
}

TEST(ANSTest, BatchPointerLarge) {
  auto res = makeStackMemory();

  std::random_device rd;
  std::mt19937 gen(10);
  std::uniform_int_distribution<uint32_t> dist(100, 10000);

  std::vector<uint32_t> sizes;
  for (int i = 0; i < 100; ++i) {
    sizes.push_back(dist(gen));
  }

  runBatchPointer(res, 10, sizes);
}

TEST(ANSTest, BatchStride) {
  auto res = makeStackMemory();

  // FIXME: 16 byte alignment required
  runBatchStride(res, 10, 13, 8192 + 16);
}
*/
