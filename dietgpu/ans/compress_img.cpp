#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
//#include <ATen/ATen.h>
//#include <glog/logging.h>
//#include <torch/types.h>
//#include <torch/torch.h>
#include <chrono>
#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"

using namespace dietgpu;
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
//std::vector<uint32_t> batchSizes;
//std::vector<void*> encPtrs;
//std::vector<const void*> inPtrs;
//std::vector<uint32_t> compressedSize;
//int precision = 11; // 根据需要设置精度
//uint32_t outBatchStride;
void compressFileWithANS(StackDeviceMemory& res,
		const std::string& inputFilePath,
		const std::string& tempFilePath,
		std::vector<uint32_t>& batchSizes,
		std::vector<void*>& encPtrs,
		std::vector<uint32_t>& compressedSize,
		int precision,
		uint32_t& outBatchStride,
		hipStream_t stream
		//, const std::string& outputFilePath
		) {
        
    std::ifstream inputFile(inputFilePath, std::ios::binary | std::ios::ate);
    if (inputFile.fail()) {
        throw std::runtime_error("Cannot open input file");
    }
    std::streamsize fileSize = inputFile.tellg();
    //fileSize = 1073741824*2;    //printf("fileSize: %u\n", fileSize);
    //std::vector<uint8_t> myVector;
    // 获取并打印出vector的最大大小
    //std::cout << "The maximum size of the vector is: " << myVector.max_size() << std::endl;
    std::vector<uint8_t> fileData(fileSize);
    std::cout << "The maximum size of the vector is: " << fileData.size() << std::endl;
    inputFile.seekg(0, std::ios::beg);
    inputFile.read(reinterpret_cast<char*>(fileData.data()), fileSize);
    //inputFile.read(reinterpret_cast<char*>(fileData.data()), 3221225472);
    inputFile.close();
    std::cout<<"fileSize:"<<fileSize<<" "<<static_cast<uint32_t>(fileSize)<<std::endl;
    /*std::cout<<"input data:\n";
    for(int i=0;i<static_cast<uint32_t>(fileSize);i++)
    {
	    std::cout<<fileData[i]<<" ";
    }
    std::cout<<std::endl;*/
    //dietgpu::StackDeviceMemory res;

    auto devData = res.alloc<uint8_t>(stream, static_cast<uint64_t>(fileSize));
    hipMemcpy(devData.data(), fileData.data(), static_cast<size_t>(fileSize), hipMemcpyHostToDevice);
    //printf("fileData[fileSize-1]:%u\n",fileData[fileSize-1]);
    printf("devData[fileSize-1]:%u\n",devData.data()[fileSize-1]);
    //int precision = 11; // 根据需要设置精度
    auto num = fileSize;
/*    batchSizes.resize(int(fileSize/1073741824) + 1);
    for(int i=0;i<batchSizes.size()-1;i++){
        batchSizes[i] = 1073741824;
        num -= 1073741824;
    }
    batchSizes[fileSize/1073741824] = num;
*/  
    int bsize;
    if(fileSize % 536870912 == 0){
       bsize = fileSize / 536870912;
    batchSizes.resize(bsize);
    for(int i=0;i<batchSizes.size();i++){
        batchSizes[i] = 536870912;
    }
    }
    else {
    bsize = fileSize / 536870912 + 1;
    batchSizes.resize(bsize);
    for(int i=0;i<batchSizes.size()-1;i++){
        batchSizes[i] = 536870912;
        num -= 536870912;
    }
    batchSizes[bsize - 1] = num;
    }
 
    //std::cout<<"totalSizes:"<<fileSize<<std::endl;
    std::cout<<"num:"<<num<<std::endl;    
    auto numInBatch = batchSizes.size();
    uint32_t maxSize = 0;
    for (auto v : batchSizes) {
      maxSize = std::max(maxSize, v);
    }
    outBatchStride = getMaxCompressedSize(maxSize);
    std::cout<<"outBatchStride:"<<numInBatch<<" "<<numInBatch*outBatchStride<<std::endl; 
    auto outCompressedSizeDev = res.alloc<uint32_t>(stream, numInBatch);

    auto inPtrs = std::vector<const void*>(batchSizes.size());
    {
      uint32_t curOffset = 0;
      for (int i = 0; i < inPtrs.size(); ++i) {
        inPtrs[i] = (uint8_t*)devData.data() + curOffset;
        curOffset += batchSizes[i];
      }
    }
    printf("inPtrs[bsize-1][batchSizes[bsize-1]-1]:%u\n",*((uint8_t* )inPtrs[bsize-1]+batchSizes[bsize-1]-1));
    
    auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * outBatchStride);
 
    encPtrs = std::vector<void*>(batchSizes.size());
    for (int i = 0; i < inPtrs.size(); ++i) {
       encPtrs[i] = (uint8_t*)enc_dev.data() + i * outBatchStride;
    }

    std::cout<<"encode start!"<<std::endl;
    double comp_time = 0.0;
    auto start = std::chrono::high_resolution_clock::now();   
    ansEncodeBatchPointer(
        res,
        dietgpu::ANSCodecConfig(precision, true),
        numInBatch, 
        inPtrs.data(),
        batchSizes.data(),
        nullptr,
        encPtrs.data(),
        outCompressedSizeDev.data(),
        stream);
    auto end = std::chrono::high_resolution_clock::now();
    //if (i > 0) {  
        comp_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
    //} 
    uint64_t total_size = fileSize;
    double c_bw = ( 1.0 * total_size / 1e9 ) / ( comp_time * 1e-3 );  

    std::cout << "comp   time " << std::fixed << std::setprecision(3) << comp_time << " ms B/W "   
                  << std::fixed << std::setprecision(1) << c_bw << " GB/s " << std::endl;

    hipStreamSynchronize(stream);

    compressedSize = outCompressedSizeDev.copyToHost(stream);
    //printf("Hey encPtrs.size:%d\n",encPtrs.size());
    std::ofstream outputFile(tempFilePath, std::ios::binary);
    if (!outputFile) {
        throw std::runtime_error("Cannot open output file");
    }
    for(int i=0;i<numInBatch;i++){
        std::vector<uint8_t> compressedData(compressedSize[i]);
        hipMemcpy(compressedData.data(), encPtrs.data()[i], compressedSize[i]*sizeof(uint8_t), hipMemcpyDeviceToHost);
        outputFile.write(reinterpret_cast<const char*>(compressedData.data()), compressedSize[i]*sizeof(uint8_t));
    }
    outputFile.close();
    //std::cout<<batchSizes[0]<<" "<<encPtrs[0]<<" "<<inPtrs[0]<<" "<<compressedSize[0]<<" "<<precision<<" "<<outBatchStride<<std::endl;
    /*
    std::ifstream inFile(tempFilePath, std::ios::binary | std::ios::ate);
    if (!inFile) {
        throw std::runtime_error("Cannot open input file");
    }
    std::vector<uint8_t> fileCompressedData(compressedSize[0]);
    inFile.read(reinterpret_cast<char*>(fileCompressedData.data()), compressedSize[0]);
    inFile.close();
    
    auto fileCompressedDev = res.alloc<uint8_t>(stream, compressedSize[0]);
    hipMemcpy(fileCompressedDev.data(), fileCompressedData.data(), compressedSize[0], hipMemcpyHostToDevice);
    auto filePtrs = std::vector<void*>(batchSizes.size());
    for (int i = 0; i < inPtrs.size(); ++i) {
        filePtrs[i] = (uint8_t*)fileCompressedDev.data() + i * outBatchStride;
    }

    auto dec_dev = buffersToDevice(res, batchSizes, stream);
    auto decPtrs = std::vector<void*>(batchSizes.size());
    for (int i = 0; i < inPtrs.size(); ++i) {
        decPtrs[i] = dec_dev[i].data();
    }

    auto outSuccess_dev = res.alloc<uint8_t>(stream, 1);
    auto outSize_dev = res.alloc<uint32_t>(stream, 1);

    ansDecodeBatchPointer(
        res,
        ANSCodecConfig(precision, true),
        1,
        (const void**)filePtrs.data(), // 使用从文件读取的数据
        decPtrs.data(),
        batchSizes.data(),
        outSuccess_dev.data(),
        outSize_dev.data(),
        stream);

    std::ofstream outFile(outputFilePath, std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Cannot open output file");
    }
    outFile.write(reinterpret_cast<const char*>(decPtrs.data()), batchSizes[0]);
    outFile.close();*/
}

void decompressFileWithANS(StackDeviceMemory& res,
		const std::string& tempFilePath, 
		const std::string& outputFilePath,                 
		std::vector<uint32_t>& batchSizes,
        std::vector<void*>& encPtrs,
        std::vector<uint32_t>& compressedSize,                
		int precision,
		uint32_t& outBatchStride,
		hipStream_t stream) {
    auto numInBatch = batchSizes.size();
    //std::cout<<batchSizes[0]<<" "<<encPtrs[0]<<" "<<inPtrs[0]<<" "<<compressedSize[0]<<" "<<precision<<" "<<outBatchStride<<std::endl;
    auto enc_dev = res.alloc<uint8_t>(stream, numInBatch * outBatchStride);

    auto filePtrs = std::vector<void*>(batchSizes.size());
    for (int i = 0; i < batchSizes.size(); ++i) {
      filePtrs[i] = (uint8_t*)enc_dev.data() + i * outBatchStride;
    }
    std::ifstream inFile(tempFilePath, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Cannot open input file");
    }
    for(int i=0;i<batchSizes.size();i++){
        std::vector<uint8_t> fileCompressedData(compressedSize[i]);
        inFile.read(reinterpret_cast<char*>(fileCompressedData.data()), compressedSize[i]);
        hipMemcpy(filePtrs[i],fileCompressedData.data(),compressedSize[i]*sizeof(uint8_t),hipMemcpyHostToDevice);
        // std::cout<<"temp[i] data:\n";
        // for(int j=0;j<compressedSize[i];j++)
	    // std::cout<<fileCompressedData[j]<<" ";
        // std::cout<<std::endl;
    }
    inFile.close();
    //std::cout<<"temp data:\n";
    // for(int i=0;i<compressedSize[0];i++)
	// std::cout<<fileCompressedData[i]<<" ";
    // std::cout<<std::endl;
    // ANSCoalescedHeader* headerOut = (ANSCoalescedHeader*) fileCompressedData.data();
    // auto header = *headerOut;
    // printf("Hey Hey Hey !  totalCompressedWords:%u\n", header.totalCompressedWords);
    // auto fileCompressedDev = res.alloc<uint8_t>(stream, compressedSize[0]);
    // hipMemcpy(fileCompressedDev.data(), fileCompressedData.data(), compressedSize[0], hipMemcpyHostToDevice);

    auto dec_dev = buffersToDevice(res, batchSizes, stream);
    auto decPtrs = std::vector<void*>(batchSizes.size());
    for (int i = 0; i < batchSizes.size(); ++i) {
        decPtrs[i] = dec_dev[i].data();
    }

    auto outSuccess_dev = res.alloc<uint8_t>(stream, numInBatch);
    auto outSize_dev = res.alloc<uint32_t>(stream, numInBatch);

    double decomp_time = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    ansDecodeBatchPointer(
        res,
        ANSCodecConfig(precision, true),
        numInBatch,
        (const void**)filePtrs.data(), // 使用从文件读取的数据
        decPtrs.data(),
        batchSizes.data(),
        outSuccess_dev.data(),
        outSize_dev.data(),
        stream);
    auto end = std::chrono::high_resolution_clock::now();  
    //if (i > 0) {  
        decomp_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;  
    //}
    uint64_t comp_size = 0;
    for(auto v: compressedSize)
        comp_size += v;
    double dc_bw = ( 1.0 * comp_size / 1e9 ) / ( decomp_time * 1e-3 );

    std::cout << "decomp time " << std::fixed << std::setprecision(3) << decomp_time << " ms B/W "   
                  << std::fixed << std::setprecision(1) << dc_bw << " GB/s" << std::endl;
    //hipMemcpy(unCompressData.data(),dec_dev[0].data(),batchSizes[0],hipMemcpyDeviceToHost);
    /*std::cout<<"output data:\n";
    for(int i=0;i<batchSizes[0];i++)
    {
	    std::cout<<unCompressData[i]<<" ";
    }
    std::cout<<std::endl;*/
    std::ofstream outFile(outputFilePath, std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Cannot open output file");
    }
    for(int i=0;i<batchSizes.size();i++){
        std::vector<uint8_t> unCompressData(batchSizes[i]);
        hipMemcpy(unCompressData.data(),decPtrs[i],batchSizes[i]*sizeof(uint8_t),hipMemcpyDeviceToHost);
        outFile.write(reinterpret_cast<const char*>(unCompressData.data()), batchSizes[i]*sizeof(uint8_t));
    }
    outFile.close();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.img> <temp.ans> <output.img>" << std::endl;
        return 1;
    }
    auto res = makeStackMemory();
    hipStream_t stream;   
    hipStreamCreate(&stream);
    std::vector<uint32_t> batchSizes;
    std::vector<void*> encPtrs;
    std::vector<uint32_t> compressedSize;
    int precision = 10; // 根据需要设置精度
    uint32_t outBatchStride;
    try {
        compressFileWithANS(res,argv[1], argv[2],batchSizes,encPtrs,compressedSize,precision,outBatchStride, stream);
	    std::cout<<"OK!"<<std::endl;
	    decompressFileWithANS(res,argv[2],argv[3],batchSizes,encPtrs,compressedSize,precision,outBatchStride, stream);
        std::cout << "Compression completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during compression: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
