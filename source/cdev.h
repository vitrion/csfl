#ifndef CDEV_H
#define CDEV_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;

class cdev
{
public:
	// CUDA Device constructor
	cdev();
	unsigned int numOfDevices;
	vector<cudaDeviceProp> ID;
private:
};

cdev::cdev()
{
	cudaError_t cudaStatus;
	int devCount;
	cudaStatus = cudaGetDeviceCount(&devCount);
	numOfDevices = unsigned(devCount);
	if (cudaStatus != cudaSuccess)
		cerr << "cudaGetDeviceCount failed! Do you have a CUDA-capable GPU installed?";

	for (int i = 0; i < devCount; i++)
	{
		cudaDeviceProp deviceProperties;
		cudaStatus = cudaGetDeviceProperties(&deviceProperties, i);
		if (cudaStatus != cudaSuccess)
			cerr << "cudaGetDeviceProperties failed! Impossible to get the device properties.";
		ID.push_back(deviceProperties);
	}
}

#endif