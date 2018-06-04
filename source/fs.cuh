#ifndef FS_CUH
#define FS_CUH

// ********************************************************************************************
// ********************************* LOADING HEADER FILES *************************************
// ********************************************************************************************

#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include <sstream>
using std::stringstream;

#include "cuda_runtime.h"

#include "cdev.h"
#include "flsFriends.cuh"
#include "fsExcept.cuh"

// ********************************************************************************************
// *********************************** CUDA DEFINITIONS ***************************************
// ********************************************************************************************

// Global CUDA Resources
cdev devices;
// Global CUDA Utility Functions
long long capacity;
unsigned int xB, yB, zB, K;
unsigned int xG, yG, zG, maxShared;
dim3 blocks, grids;
cudaError_t cudaStatus;
float *dev_result1, *dev_result2, *dev_op1, *dev_op2, *dev_op3, *dev_op4;
bool *dev_boolResult;
__device__ float disc[samples];

// Global CUDA Utility Functions
void CUDAinit() {
	cudaDeviceReset();
	xB = 1; yB = 1; zB = 1;
	xG = 1; yG = 1; zG = 1;
	if (samples <= devices.ID[0].maxThreadsPerBlock)
		xB = samples;
	else {
		xB = devices.ID[0].maxThreadsPerBlock;
		if (samples <= xB * devices.ID[0].maxGridSize[1])
			xG = unsigned(ceil(float(samples) / float(xB)));
		else
			if (samples <= xB * devices.ID[0].maxGridSize[1] * devices.ID[0].maxGridSize[2]) {
				xG = devices.ID[0].maxGridSize[1];
				yG = unsigned(ceil(float(samples) / float(xB * xG)));
			}
			else {
				xG = devices.ID[0].maxGridSize[1];
				yG = devices.ID[0].maxGridSize[2];
				zG = unsigned(ceil(float(samples) / float(xB * xG * yG)));
			}
	}
	if (samples < devices.ID[0].sharedMemPerMultiprocessor)
		maxShared = samples;
	else
		maxShared = devices.ID[0].sharedMemPerMultiprocessor;
		
	K = unsigned(ceil(float(samples) / devices.ID[0].sharedMemPerMultiprocessor));
	capacity = xB * yB * zB * xG * yG * zG;
	grids = dim3(xG, yG, zG);
	blocks = dim3(xB, yB, zB);

	cudaStatus = cudaMalloc((void **)&dev_result1, samples * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cerr << "Memory allocation failed in CUDA initialization of dev_result1: " << cudaGetErrorString(cudaStatus) << endl;
		cin.get();
		exit(1);
	}
	cudaStatus = cudaMalloc((void **)&dev_result2, samples * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cerr << "Memory allocation failed in CUDA initialization of dev_result1: " << cudaGetErrorString(cudaStatus) << endl;
		cin.get();
		exit(1);
	}
	cudaStatus = cudaMalloc((void **)&dev_op1, samples * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cerr << "Memory allocation failed in CUDA initialization of dev_op1: " << cudaGetErrorString(cudaStatus) << endl;
		cin.get();
		exit(1);
	}
	cudaStatus = cudaMalloc((void **)&dev_op2, samples * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cerr << "Memory allocation failed in CUDA initialization of dev_op2: " << cudaGetErrorString(cudaStatus) << endl;
		cin.get();
		exit(1);
	}
	cudaStatus = cudaMalloc((void **)&dev_op3, samples * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cerr << "Memory allocation failed in CUDA initialization of dev_op1: " << cudaGetErrorString(cudaStatus) << endl;
		cin.get();
		exit(1);
	}
	cudaStatus = cudaMalloc((void **)&dev_op4, samples * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		cerr << "Memory allocation failed in CUDA initialization of dev_op2: " << cudaGetErrorString(cudaStatus) << endl;
		cin.get();
		exit(1);
	}
	cudaStatus = cudaMalloc((void **)&dev_boolResult, samples * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		cerr << "Array allocation failed in fs::getSupport (result) in line 1130: "
			<< cudaGetErrorString(cudaStatus) << endl;
		cin.get();
		exit(1);
	}
}

void CUDAend() {
	cudaFree(dev_result1);
	cudaFree(dev_result2);
	cudaFree(dev_op1);
	cudaFree(dev_op2);
	cudaFree(dev_op3);
	cudaFree(dev_op4);
	cudaFree(disc);
	cudaFree(dev_boolResult);
	streams.clear();
	cudaDeviceReset();
}

// Global CUDA kernel definitions
__global__ void _linearSpace(float *output, float base, float delta) {	
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = base + float(i) * delta;
}
__global__ void _linearSpace(float base, float delta) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		disc[i] = base + float(i) * delta;
}
__global__ void _singletonMF(float *output, float *input, float center, float normalization) {
	// extern __shared__ float d[];
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	disc[i] = input[i];
	__syncthreads();
	if (i < samples)
		if (disc[i] >= float(1.01) * center || disc[i] <= float(0.99) * center)
			output[i] = float(0.0);
		else
			output[i] = normalization;
}

__global__ void _intervalMF(float *output, float *input, float left, float right, float normalization) {
	// extern __shared__ float d[];
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	disc[i] = input[i];
	__syncthreads();
	if (i < samples)
		if (disc[i] > right || disc[i] < left)
			output[i] = float(0.0);
		else
			output[i] = normalization;
}

__global__ void _triangularMF(float *output, float *input, float left, float center, float right, float normalization) {
	// extern __shared__ float d[];
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	disc[i] = input[i];
	__syncthreads();
	if (i < samples)
		if (disc[i] >= center && disc[i] <= right)
			output[i] = (right - disc[i]) / (right - center);
		else if (disc[i] >= left && disc[i] <= center)
			output[i] = (disc[i] - left) / (center - left);
		else
			output[i] = float(0.0);
}

__global__ void _trapezoidalMF(float *output, float *input, float left, float centerLeft, float centerRight, float right, float normalization) {
	// extern __shared__ float d[];
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	disc[i] = input[i];
	__syncthreads();
	if (i < samples)
		if (disc[i] >= centerLeft && disc[i] <= centerRight)
			output[i] = 1;
		else if (disc[i] >= centerRight && disc[i] <= right)
			output[i] = (right - disc[i]) / (right - centerRight);
		else if (disc[i] >= left && disc[i] <= centerLeft)
			output[i] = (disc[i] - left) / (centerLeft - left);
		else
			output[i] = float(0.0);
}

__global__ void _sMF(float *output, float *input, float left, float right, float normalization) {
	// extern __shared__ float d[];
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	disc[i] = input[i];
	__syncthreads();
	if (i < samples)
		if (disc[i] >= left && disc[i] <= (left + right) / 2)
			output[i] = float(2.0) * pow((disc[i] - left) / (right - left), float(2.0));
		else if (disc[i] >= (left + right) / 2 && disc[i] <= right)
			output[i] = float(1.0) - float(2.0) * pow((disc[i] - right) / (right - left), float(2.0));
		else if (disc[i] >= right)
			output[i] = 1;
		else
			output[i] = float(0.0);
}

__global__ void _zMF(float *output, float *input, float left, float right, float normalization) {
	// extern __shared__ float d[];
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	disc[i] = input[i];
	__syncthreads();
	if (i < samples)
		if (disc[i] >= left && disc[i] <= (left + right) / 2)
			output[i] = float(1.0) - float(2.0) * pow((disc[i] - left) / (right - left), float(2.0));
		else if (disc[i] >= (left + right) / 2 && disc[i] <= right)
			output[i] = float(2.0) * pow((disc[i] - right) / (right - left), float(2.0));
		else if (disc[i] >= right)
			output[i] = 0;
		else
			output[i] = 1;
}

__global__ void _gaussianMF(float *output, float *input, float center, float sigma, float normalization) {
	// extern __shared__ float d[];
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	disc[i] = input[i];
	__syncthreads();
	if (i < samples)
		output[i] = normalization * exp(-pow(disc[i] - center, float(2.0)) / (float(2.0) * pow(sigma, float(2.0))));
}

__global__ void _minimumIntersection(float *output, float *input1, float *input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] < input2[i] ? input1[i] : input2[i];
}
__global__ void _minimumIntersection(float *output, float *input1, float input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] < input2 ? input1[i] : input2;
}
__global__ void _minimumIntersection(float *output, float input1, float *input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1 < input2[i] ? input1 : input2[i];
}
__global__ void _productIntersection(float *output, float *input1, float *input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] * input2[i];
}
__global__ void _boundedIntersection(float *output, float *input1, float *input2, float normalization) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = (input1[i] + input2[i] - normalization) > float(0.0) ? (input1[i] + input2[i] - normalization) : float(0.0);
}
__global__ void _drasticIntersection(float *output, float *input1, float *input2, float normalization) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		if (input1[i] == normalization)
			output[i] = input2[i];
		else if (input2[i] == normalization)
			output[i] = input1[i];
		else
			output[i] = float(0.0);
}
__global__ void _maximumUnion(float *output, float *input1, float *input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] > input2[i] ? input1[i] : input2[i];
}
__global__ void _maximumUnion(float *output, float *input1, float input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] > input2 ? input1[i] : input2;
}
__global__ void _maximumUnion(float *output, float input1, float *input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1 > input2[i] ? input1 : input2[i];
}
__global__ void _algebraicUnion(float *output, float *input1, float *input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] * input2[i];
}
__global__ void _boundedUnion(float *output, float *input1, float *input2, float normalization) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] + input2[i] - input1[i] * input2[i];
}
__global__ void _drasticUnion(float *output, float *input1, float *input2, float normalization) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		if (input1[i] == float(0.0))
			output[i] = input2[i];
		else if (input2[i] == float(0.0))
			output[i] = input1[i];
		else
			output[i] = normalization;
}

__global__ void _complement(float *output, float *input1) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = 1 - input1[i];
}
__global__ void _fsAdd(float *output1, float *output2, float *input1, float *input2, float *input3, float *input4) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples) {
		output1[i] = input1[i] + input2[i];
		__syncthreads();
		output2[i] = (input3[i] + input4[i]) / 2;
		__syncthreads();
	}
}
__global__ void _fsSub(float *output1, float *output2, float *input1, float *input2, float *input3, float *input4) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples){
		output1[i] = input1[i] - input2[samples - i];
		__syncthreads();
		output2[i] = (input3[i] + input4[i]) / 2;
		__syncthreads();
	}
}
__global__ void _fsMul(float *output1, float *output2, float *input1, float *input2, float *input3, float *input4) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples) {
		output1[i] = input1[i] * input2[i];
		__syncthreads();
		output2[i] = (input3[i] + input4[i]) / 2;
		__syncthreads();
	}
}
__global__ void _fsMul(float *output1, float *input1, float *input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output1[i] = input1[i] * input2[i];
}
__global__ void _fsMul(float *output1, float *input1, float input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output1[i] = input1[i] * input2;
}
__global__ void _fsDiv(float *output1, float *output2, float *input1, float *input2, float *input3, float *input4) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples) {
		output1[i] = input1[i] / input2[i];
		__syncthreads();
		output2[i] = (input3[i] + input4[i]) / 2;
		__syncthreads();
	}
}
__global__ void _fsMod(float *output1, float *output2, float *input1, float *input2, float *input3, float *input4) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples) {
		output1[i] = static_cast<int>(round(input1[i])) % static_cast<int>(round(input2[i]));
		__syncthreads();
		output2[i] = (input3[i] + input4[i]) / 2;
		__syncthreads();
	}
}
__global__ void _fsCmpEq(bool *output, float *input1, float *input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] == input2[i];		
}
__global__ void _fsCmpEq(bool *output, float *input1, float input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] == input2;
}
__global__ void _fsCmpUneq(bool *output, float *input1, float *input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] != input2[i];
}
__global__ void _fsCmpUneq(bool *output, float *input1, float input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] != input2;
}
__global__ void _fsCmpGreat(bool *output, float *input1, float *input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] > input2[i];
}
__global__ void _fsCmpGreat(bool *output, float *input1, float input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] > input2;
}
__global__ void _fsCmpLow(bool *output, float *input1, float *input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] < input2[i];
}
__global__ void _fsCmpLow(bool *output, float *input1, float input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] < input2;
}
__global__ void _fsCmpGreatEq(bool *output, float *input1, float *input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] >= input2[i];
}
__global__ void _fsCmpGreatEq(bool *output, float *input1, float input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] >= input2;
}
__global__ void _fsCmpLowEq(bool *output, float *input1, float *input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] <= input2[i];
}
__global__ void _fsCmpLowEq(bool *output, float *input1, float input2) {
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < samples)
		output[i] = input1[i] <= input2;
}
// ********************************************************************************************
// ******************************** FUZZY SET CLASS DEFINITION ********************************
// ********************************************************************************************

class fs
{
public:
	// Fuzzy Set: Constructors

	// Constructor 0: Default Fuzzy Set Constructor
	fs(); // OK

	// Constructor 1: Fuzzy Set Constructor defining the Range
	fs(const float &, const float &); // OK

	// Constructor 2: Fuzzy Set Constructor defining the Range and Normalization Value
	fs(const float &, const float &, const float &); // OK

	// Constructor 3: Fuzzy Set Constructor defining the Range, Normalization Value and Linguistic Tag
	fs(const float &, const float &, const float &, const string &); // OK

	// Constructor 4: Fuzzy Set Constructor defining the Normalization Value
	fs(const float &);

	// Constructor 5: Fuzzy Set Constructor defining the Set Linguistic Tag
	fs(const string &); // OK

	// Constructor 6: Fuzzy Set Constructor defining the Set Linguistic Tag, Shape and Parameters
	fs(const string &, const string &, const vector<float> &); // OK

	// Constructor 7: Fuzzy Set Constructor defining the Set Shape and Parameters
	fs(const string &, const vector<float> &); // OK

	// Constructor 8: Fuzzy Set Constructor defining the CUDA processing
	fs(const bool &); // OK

	// Constructor 9: Fuzzy Set Constructor defining the CUDA and Stream processing
	fs(const bool &, const bool &);

	// Constructor 10: Fuzzy Set Constructor defining the Range, Normalization Value, Linguistic Tag and Shape
	fs(const float &, const float &, const float &, const string &, const string &); // OK

	// Constructor 11: Fuzzy Set Constructor defining the Range, Normalization Value, Linguistic Tag, Shape and Parameters
	fs(const float &, const float &, const float &, const string &, const string &, const vector<float> &); // OK

	// Constructor 12: Fuzzy Set Constructor defining the Range, Normalization Value, Linguistic Tag, Shape, Parameters and CUDA processing
	fs(const float &, const float &, const float &, const string &, const string &, const vector<float> &, const bool &); // OK

	// Constructor 13: Fuzzy Set Constructor defining the Normalization Value, Linguistic Tag, Shape and Parameters
	fs(const float &, const string &, const string &, const vector<float> &); // OK

	// Constructor 14: Fuzzy Set Constructor defining the Range, Normalization Value, Linguistic Tag, Shape, Parameters, CUDA and Stream processing
	fs(const float &, const float &, const float &, const string &, const string &, const vector<float> &, const bool &, const bool &); // OK

	// Constructor 15; Fuzzy Set Constructor defining the range, CUDA and Stream processing
	fs(const float &, const float &, const bool &, const bool &);

	// Constructor 16; Fuzzy Set Constructor defining the range, CUDA and Stream processing
	fs(const float &, const float &, const string &, const bool &, const bool &);

	// Fuzzy Set: Set Functions
	void setRange(const float &, const float &);
	void setRange(const float &, const float &, const bool &); // OK *** CUDA Accelerated ***
	void setName(const string &); // OK
	void setShape(const string &); // OK
	void setNorm(const float &); // OK
	void setNorm(); // OK
	void setParams(const vector<float> &); // OK
	void setConjOp(const string &); // OK
	void setDisjOp(const string &); // OK
	void setAggregMethod(const string &); // OK
	void setDefuzzMethod(const string &); // OK
	void setMembership(const vector<float> &); // OK
	void setOffset(const float &); // OK
	void setSupport(); // OK

	// Fuzzy Set: Get Functions
	string getName() const; // OK
	string getShape() const; // OK
	float getNormalization() const; // OK
	vector<float> getRange() const; // OK
	vector<float> getParameters() const; // OK
	vector<unsigned int> getSupport(); // OK *** CUDA
	vector<unsigned int> getSupport(const float &); // OK *** CUDA
	vector<unsigned int> getSupport(vector<float> *, const float &); // OK *** CUDA
	string getConjOp() const; // OK
	string getDisjOp() const; // OK
	string getAggregMethod() const; // OK
	string getDefuzzMethod() const; // OK
	vector<float> getDiscourse() const; // OK
	vector<float> getMembership() const; // OK
	
	// Fuzzy Set: Fuzzification
	void fuzzify(); // OK
	float fuzzify(const float &, const unsigned int &); // OK

	// Fuzzy Set: Defuzzification
	float defuzzify() const; // OK
	void defuzzification(); // OK

	// OVERLOADED OPERATORS
	// Fuzzy Set: Arithmetic Operators
	fs operator+(const fs &); // Fuzzy Add // OK *** CUDA Accelerated ***
	fs operator-(const fs &); // Fuzzy Subtract // OK *** CUDA Accelerated ***
	fs operator*(const fs &); // Fuzzy Multiplication // OK *** CUDA Accelerated ***
	fs operator/(const fs &); // Fuzzy Division // OK *** CUDA Accelerated ***
	fs operator%(const fs &); // Fuzzy Modulus // OK *** CUDA Accelerated ***

	// Fuzzy Set: Comparison Operators
	bool operator==(const fs &) const; // Fuzzy Equality // OK *** CUDA
	bool operator!=(const fs &) const; // Fuzzy Unequality // OK *** CUDA
	bool operator<(const fs &) const; // Fuzzy Lower Than // OK *** CUDA
	bool operator>(const fs &) const; // Fuzzy Greater Than // OK *** CUDA
	bool operator<=(const fs &) const; // Fuzzy Lower or Equal Than // OK *** CUDA
	bool operator>=(const fs &) const; // Fuzzy Greater or Equal Than // OK *** CUDA

	// Fuzzy Set: Logic Operators
	fs operator!() const; // Fuzzy Complement // OK *** CUDA
	fs operator&(const fs &) const; // Fuzzy Conjunction // OK *** CUDA
	fs operator&&(const fs &) const; // Minimum // OK *** CUDA
	fs operator||(const fs &) const; // Maximum // OK *** CUDA
	fs operator!(); // Fuzzy complement // OK *** NOT CUDA
	fs operator|(const fs &) const; // Fuzzy Disjunction // OK *** CUDA
	fs operator+=(const fs &) const; // Fuzzy Aggregation // OK *** CUDA
	const fs &operator=(const fs &); // Fuzzy Assignment // OK *** CUDA

	fs operator&(const float &) const; // Fuzzy Conjunction // OK *** CUDA
	fs operator|(const float &) const; // Fuzzy Disjunction // OK *** CUDA

	// Fuzzy Set Display Functions
	void display();
	void display(const fs &);

	// Friend Functions
	friend vector<float> linearSpace(const float &, const float &, const unsigned int);

	// About
	string about() const;
	bool isCUDA; // OK
	bool isStream;
	vector<float> discourse; // OK
	vector<float> membership; // OK
	unsigned int idxOfMax;
	unsigned int idxOfCtr;
private:
	// Fuzzy Set: Configuration Data
	string name; // OK
	string shape; // OK
	float normalization; // OK
	float crisp; // OK
	vector<float> parameters; // OK	
	vector<float> range; // OK
	vector<unsigned int> support; // OK
	string conjOperator; // OK
	string disjOperator; // OK
	string aggregMethod; // OK
	string defuzzMethod; // OK
	vector<string> availableMF; // OK
	vector<string> availableTNorms; // OK
	vector<string> availableSNorms; // OK
	vector<string> availableDefuzz; // OK

	// Fuzzy Set: Flags

	// Fuzzy Set: Membership Functions
	void singletonMF(); // OK *** CUDA Accelerated ***
	void intervalMF(); // OK *** CUDA Accelerated ***
	void triangularMF(); // OK *** CUDA Accelerated ***
	void trapezoidalMF(); // OK *** CUDA Accelerated ***
	void sMF(); // OK *** CUDA Accelerated ***
	void zMF(); // OK *** CUDA Accelerated ***
	void gaussianMF(); // OK *** CUDA Accelerated ***
	float singletonMF(const float &) const; // OK
	float intervalMF(const float &) const; // OK
	float triangularMF(const float &) const; // OK
	float trapezoidalMF(const float &) const; // OK
	float sMF(const float &) const; // OK
	float zMF(const float &) const; // OK
	float gaussianMF(const float &) const; // OK

	// Utility Functions
	void setAvailableFunctions(); // OK
	void setDefaultFunctions(); // OK

	// Fuzzy Set: Error Checking Functions
	void check4MaxNorm(); // OK
	void check4ordinality(); // OK
	void check4paramNumb(); // OK
	void check4NaN(); // OK
	void check4MFAvail(); // OK
	void check4ConjOpAvail(); // OK
	void check4DisjOpAvail(); // OK
	void check4AggregMethodAvail(); // OK
	void check4DefuzzAvail(); // OK
	void check4SizeMismatch(const vector<float> &, const vector<float> &) const; // OK
	void check4ProcessAvail();
	void check4ValidMFShape4Shift();

	// Fuzzy Set: Norm Functions
	float tnorm(const string &, const float &, const float &) const; // OK
	vector<float> tnorm(const string &, const vector<float> &, const vector<float> &) const; // OK *** CUDA Accelerated ***
	float snorm(const string &, const float &, const float &) const; // OK
	vector<float> snorm(const string &, const vector<float> &, const vector<float> &) const; // OK *** CUDA Accelerated ***
	vector<float> fnot(vector<float> &) const; // OK
	float minimumIntersection(const float &, const float &) const; // OK
	float productIntersection(const float &, const float &) const; // OK
	float boundedIntersection(const float &, const float &) const; // OK
	float drasticIntersection(const float &, const float &) const; // OK
	float maximumUnion(const float &, const float &) const; // OK
	float algebraicUnion(const float &, const float &) const; // OK
	float boundedUnion(const float &, const float &) const; // OK
	float drasticUnion(const float &, const float &) const; // OK
	float min(const float &, const float &) const; // OK
	float max(const float &, const float &) const; // OK
	float min(const vector<float> &) const; // OK
	float max(const vector<float> &) const; // OK
	vector<unsigned int> minIdx(vector<float> *); // OK
	vector<unsigned int> maxIdx(vector<float> *); // OK
	vector<float> min(const vector<float> &, const vector<float> &) const; // OK *** CUDA Accelerated ***
	vector<float> max(const vector<float> &, const vector<float> &) const; // OK *** CUDA Accelerated ***
	vector<float> min(const vector<float> &, const float &) const; // OK *** CUDA Accelerated ***
	vector<float> max(const vector<float> &, const float &) const; // OK *** CUDA Accelerated ***
	vector<float> min(const float &, const vector<float> &) const; // OK *** CUDA Accelerated ***
	vector<float> max(const float &, const vector<float> &) const; // OK *** CUDA Accelerated ***
	vector<fs> normalize(vector<fs> &) const; // OK *** CUDA Accelerated ***

	// Fuzzy Set: Defuzzification Methods
	float adaptiveIntegration() const;
	float basicDefuzzDistributions() const;
	float bisectorOfArea() const; // OK
	float constraintDecision() const;
	float centerOfArea() const;
	float centroid(); // OK
	float extendedCenterOfArea() const;
	float extendedQuality() const;
	float fuzzyClustering() const;
	float fuzzyMean() const;
	float generalLevelSet() const;
	float indexedCenterOfGravity() const;
	float influenceValue() const;
	float smallestOfMaximum(); // OK
	float largestOfMaximum(); // OK
	float meanOfMaxima(); // OK
	float quality() const;
	float randomChoiceOfMaximum() const;
	float semiLinear() const;
	float weightedFuzzyMean() const;
	float geometric() const;
};

// ********************************************************************************************
// ********************************  FUZZY SET CLASS METHODS **********************************
// ********************************************************************************************

// FUZZY SET: CONSTRUCTORS

// Constructor 0
fs::fs() {
	isCUDA = false;
	isStream = false;
	setAvailableFunctions();
	setDefaultFunctions();
	//setRange(float(0.0), float(1.0));
	setNorm(float(1.0));
	setName("Untitled");
	parameters = {};
	/*fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();*/
}

// Constructor 1
fs::fs(const float &lowerRange, const float &upperRange) {
	isCUDA = false;
	isStream = false;
	setAvailableFunctions();
	setDefaultFunctions();
	setRange(lowerRange, upperRange);
	setNorm(float(1.0));
	setName("Untitled");
	parameters = {};
	/*fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();*/
}

// Constructor 2
fs::fs(const float &lowerRange, const float &upperRange, const float &proposedNormalization) {
	isCUDA = false;
	isStream = false;
	setAvailableFunctions();
	setDefaultFunctions();
	setRange(lowerRange, upperRange);
	setNorm(proposedNormalization);
	setName("Untitled");
	parameters = {};
	/*fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();*/
}

// Constructor 3
fs::fs(const float &lowerRange, const float &upperRange, const float &proposedNormalization,
	const string &name) {
	isCUDA = false;
	isStream = false;
	setAvailableFunctions();
	setDefaultFunctions();
	setRange(lowerRange, upperRange);
	setNorm(proposedNormalization);
	setName(name);
	parameters = {};
	/*fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();*/
}

// Constructor 4
fs::fs(const float &proposedNormalization) {
	isCUDA = false;
	isStream = false;
	setAvailableFunctions();
	setDefaultFunctions();
	//setRange(float(0.0), float(1.0));
	setNorm(proposedNormalization);
	setName("Untitled");
	parameters = {};
	/*fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();*/
}

// Constructor 5
fs::fs(const string &name) {
	isCUDA = false;
	isStream = false;
	setAvailableFunctions();
	setDefaultFunctions();
	//setRange(float(0.0), float(1.0));
	setNorm(float(1.0));
	setName(name);
	parameters = {};
	/*fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();*/
}

// Constructor 6
fs::fs(const string &name, const string &shape, const vector<float> &parameters) {
	isCUDA = false;
	isStream = false;
	setAvailableFunctions();
	setDefaultFunctions();
	setRange(float(0.0), 1);
	setNorm(float(1.0));
	setName(name);
	setShape(shape);
	setParams(parameters);
	fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();
}

// Constructor 7
fs::fs(const string &shape, const vector<float> &parameters) {
	isCUDA = false;
	isStream = false;
	setAvailableFunctions();
	setDefaultFunctions();
	setRange(float(0.0), 1);
	setNorm(float(1.0));
	setName("Untitled");
	setShape(shape);
	setParams(parameters);
	fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();
}

// Constructor 8
fs::fs(const bool &allowCUDA) {
	isCUDA = allowCUDA;
	isStream = false;
	setAvailableFunctions();
	setDefaultFunctions();
	//setRange(float(0.0), 1);
	setNorm(float(1.0));
	setName("Untitled");
	parameters = {};
	/*fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();*/
}

// Constructor 9
fs::fs(const bool &allowCUDA, const bool &s) {
	isCUDA = allowCUDA;
	isStream = s;
	setAvailableFunctions();
	setDefaultFunctions();
	//setRange(float(0.0), 1);
	setNorm(float(1.0));
	setName("Untitled");
	parameters = {};
	/*fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();*/
}

// Constructor 10
fs::fs(const float &lowerRange, const float &upperRange, const float &proposedNormalization,
const string &name, const string &shape) {
	isCUDA = false;
	isStream = false;
	setAvailableFunctions();
	setDefaultFunctions();
	setRange(lowerRange, upperRange);
	setNorm(proposedNormalization);
	setName(name);
	setShape(shape);
	if (shape.compare("Singleton") == 0)
		setParams({ (lowerRange + upperRange) / 2 });
	else if (shape.compare("Interval") == 0)
		setParams({ lowerRange, upperRange });
	else if (shape.compare("Triangular") == 0)
		setParams({ lowerRange, (lowerRange + upperRange) / 2, upperRange });
	else if (shape.compare("Trapezoidal") == 0)
		setParams({ lowerRange, lowerRange + (upperRange - lowerRange) / 3, lowerRange + float(2.0) * (upperRange - lowerRange) / 3, upperRange });
	else if (shape.compare("S") == 0)
		setParams({ lowerRange, upperRange });
	else if (shape.compare("Z") == 0)
		setParams({ lowerRange, upperRange });
	else if (shape.compare("Gaussian") == 0)
		setParams({ 0, (lowerRange + upperRange) / 2 });
	fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();
}

// Constructor 11
fs::fs(const float &lowerRange, const float &upperRange, const float &proposedNormalization,
const string &name, const string &shape, const vector<float> &parameters) {
	isCUDA = false;
	isStream = false;
	setAvailableFunctions();
	setDefaultFunctions();
	setRange(lowerRange, upperRange);
	setNorm(proposedNormalization);
	setName(name);
	setShape(shape);
	setParams(parameters);
	fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();
}

// Constructor 12
fs::fs(const float &lowerRange, const float &upperRange, const float &proposedNormalization,
const string &name, const string &shape, const vector<float> &parameters, const bool &allowCUDA) {
	isCUDA = allowCUDA;
	isStream = false;
	setAvailableFunctions();
	setDefaultFunctions();
	setRange(lowerRange, upperRange);
	setNorm(proposedNormalization);
	setName(name);
	setShape(shape);
	setParams(parameters);
	fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();
}

// Constructor 13
fs::fs(const float &proposedNormalization, const string &name, const string &shape, const vector<float> &parameters) {
	isCUDA = false;
	isStream = false;
	setAvailableFunctions();
	setDefaultFunctions();
	setRange(float(0.0), 1);
	setNorm(proposedNormalization);
	setName(name);
	setShape(shape);
	setParams(parameters);
	fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();
}

// Constructor 14
fs::fs(const float &lowerRange, const float &upperRange, const float &proposedNormalization,
	const string &name, const string &shape, const vector<float> &parameters, const bool &allowCUDA, const bool &s) {
	isCUDA = allowCUDA;
	isStream = s;
	setAvailableFunctions();
	setDefaultFunctions();
	setRange(lowerRange, upperRange);
	setNorm(proposedNormalization);
	setName(name);
	setShape(shape);
	setParams(parameters);
	fuzzify();
	normalization = max(membership);
	setSupport();
	defuzzification();
}

// Constructor 15:
fs::fs(const float &lowerRange, const float &upperRange, const bool &allowCUDA, const bool &s) {
	isCUDA = allowCUDA;
	isStream = s;
	setAvailableFunctions();
	setDefaultFunctions();
	setRange(lowerRange, upperRange);
	setNorm(float(1.0));
	setName("Untitled");
	parameters = {};
}

fs::fs(const float &lowerRange, const float &upperRange, const string &name, const bool &allowCUDA, const bool &s) {
	isCUDA = allowCUDA;
	isStream = s;
	setAvailableFunctions();
	setDefaultFunctions();
	setRange(lowerRange, upperRange);
	setNorm(float(1.0));
	setName(name);
	parameters = {};
}

// FUZZY SET: ERROR CHECKING FUNTIONS
void fs::check4MaxNorm() {
	if (normalization < float(0.0) || normalization > float(1.0))
		throw maxNormFault();
}

void fs::check4ordinality() {
	for (unsigned int i = 1; i < parameters.size(); i++)
		if (parameters[i] < parameters[i - 1])
			throw ordinalityFault("\"" + name + "\"", "\"" + shape + "\"");
}

void fs::check4paramNumb() {
	stringstream proposed, required;

	if (shape.compare("Singleton") == 0 && parameters.size() != 1) {
		proposed << parameters.size();
		required << 1;
		throw paramNumbFault("\"" + name + "\"", "\"" + shape + "\"", proposed.str(), required.str());
	}
	else if (shape.compare("Interval") == 0 && parameters.size() != 2) {
		proposed << parameters.size();
		required << 2;
		throw paramNumbFault("\"" + name + "\"", "\"" + shape + "\"", proposed.str(), required.str());
	}
	else if (shape.compare("Triangular") == 0 && parameters.size() != 3) {
		proposed << parameters.size();
		required << 3;
		throw paramNumbFault("\"" + name + "\"", "\"" + shape + "\"", proposed.str(), required.str());
	}
	else if (shape.compare("Trapezoidal") == 0 && parameters.size() != 4) {
		proposed << parameters.size();
		required << 4;
		throw paramNumbFault("\"" + name + "\"", "\"" + shape + "\"", proposed.str(), required.str());
	}
	else if (shape.compare("S") == 0 && parameters.size() != 2) {
		proposed << parameters.size();
		required << 2;
		throw paramNumbFault("\"" + name + "\"", "\"" + shape + "\"", proposed.str(), required.str());
	}
	else if (shape.compare("Z") == 0 && parameters.size() != 2) {
		proposed << parameters.size();
		required << 2;
		throw paramNumbFault("\"" + name + "\"", "\"" + shape + "\"", proposed.str(), required.str());
	}
	else if (shape.compare("Gaussian") == 0 && parameters.size() != 2) {
		proposed << parameters.size();
		required << 2;
		throw paramNumbFault("\"" + name + "\"", "\"" + shape + "\"", proposed.str(), required.str());
	}
}

void fs::check4NaN() {
	string expr;

	if (shape.compare("Triangular") == 0) {
		if ((parameters[0] == parameters[1]) || (parameters[1] == parameters[2])) {
			expr = "Parameter A must be different than parameter B or parameter B must be different than parameter C.";
			throw paramNaNFault("\"" + name + "\"", "\"" + shape + "\"", expr);
		}
	}
	else if (shape.compare("Trapezoidal") == 0) {
		if ((parameters[0] == parameters[1]) || (parameters[2] == parameters[3])) {
			expr = "Parameter A must be different than parameter B or parameter C must be different than parameter D.";
			throw paramNaNFault("\"" + name + "\"", "\"" + shape + "\"", expr);
		}
	}
	else if (shape.compare("S") == 0) {
		if (parameters[0] == parameters[1]) {
			expr = "Parameter A must be different than parameter B.";
			throw paramNaNFault("\"" + name + "\"", "\"" + shape + "\"", expr);
		}
	}
	else if (shape.compare("Z") == 0) {
		if (parameters[0] == parameters[1]) {
			expr = "Parameter A must be different than parameter B.";
			throw paramNaNFault("\"" + name + "\"", "\"" + shape + "\"", expr);
		}
	}
}

void fs::check4MFAvail() {
	unsigned int j = 0;
	for (unsigned int i = 0; i < availableMF.size(); i++)
		if (shape == availableMF[i]) {
			j++;
			break;
		}
	if (j == 0)
		throw mfShapeFault("\"" + shape + "\"");
}

void fs::check4ConjOpAvail() {
	unsigned int j = 0;
	for (unsigned int i = 0; i < availableTNorms.size(); i++)
		if (conjOperator == availableTNorms[i]) {
			j++;
			break;
		}
	if (j == 0)
		throw conjOpFault("\"" + conjOperator + "\"");
}

void fs::check4DisjOpAvail() {
	unsigned int j = 0;
	for (unsigned int i = 0; i < availableSNorms.size(); i++)
		if (disjOperator == availableSNorms[i]) {
			j++;
			break;
		}
	if (j == 0)
		throw disjOpFault("\"" + disjOperator + "\"");
}

void fs::check4AggregMethodAvail() {
	unsigned int j = 0;
	for (unsigned int i = 0; i < availableSNorms.size(); i++)
		if (aggregMethod == availableSNorms[i]) {
			j++;
			break;
		}
	if (j == 0)
		throw aggregMethodFault("\"" + disjOperator + "\"");
}

void fs::check4DefuzzAvail() {
	unsigned int j = 0;
	for (unsigned int i = 0; i < availableDefuzz.size(); i++)
		if (defuzzMethod == availableDefuzz[i]) {
			j++;
			break;
		}
	if (j == 0)
		throw defuzzMethodFault("\"" + defuzzMethod + "\"");
}

void fs::check4SizeMismatch(const vector<float> &left, const vector <float> &right) const {
	if (left.size() != right.size())
		throw sizeMismatchFault();
}

void fs::check4ValidMFShape4Shift() {
	if (this->shape == "Unknown")
		throw invalidShape4ShiftFault();
}

// FUZZY SET: SET FUNCTIONS
void fs::setRange(const float &lower, const float &upper) {
	setRange(lower, upper, false);
}
void fs::setRange(const float &lower, const float &upper, const bool &interpolate) {
	vector<float> result(samples);
	vector<float> last_range = range;
	float dx = (upper - lower) / (samples - 1);
	if (isCUDA) {
		float *r = &result[0];
		// Launch the kernel
		_linearSpace <<<grids, blocks>>>(dev_result1, lower, dx);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::setRange in line 849: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer new discourse from the device to the host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data copy failed in fs::setRange in line 856: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else {
		float summation = lower;
		for (unsigned int i = 0; i < samples; i++) {
			result[i] = summation;
			summation += dx;
		}
	}
	range = { lower, upper };
	discourse = result;
	setSupport();
	
	/*if (parameters.size() != 0 ) {
		if (shape.compare("Interval") == 0)
			parameters = { discourse[support[0]], discourse[support[1]] };
		else if (shape.compare("Triangular") == 0) {
			float m = float(0.99) * max(membership);
			vector<unsigned int> n = getSupport(m);
			parameters = { discourse[support[0]], discourse[n[0] + 1], discourse[support[1]] };
		}
		else if (shape.compare("Trapezoidal") == 0) {
			float m = float(0.99) * max(membership);
			vector<unsigned int> n = getSupport(m);
			parameters = { discourse[support[0]], discourse[n[0] + 1], discourse[n[1] - 1], discourse[support[1]] };
		}
		else if (shape.compare("S") == 0) {
			float m = float(0.99) * max(membership);
			vector<unsigned int> n = getSupport(m);
			parameters = { discourse[support[0]], discourse[n[0] + 1] };
		}
		else if (shape.compare("Z") == 0) {
			float m = float(0.99) * max(membership);
			vector<unsigned int> n = getSupport(m);
			parameters = { discourse[n[0]], discourse[support[1]] };
		}
		else if (shape.compare("Gaussian") == 0) {
			float m = float(0.99) * max(membership);
			vector<unsigned int> n = getSupport(m);
			if (last_range.size() != 0) {
				float q = (last_range[1] - last_range[0]) / (range[1] - range[0]);
				parameters = { discourse[n[0] + 1], q * parameters[1] };
			}
		}
	}*/

		/*if (interpolate) {
			unsigned int supportIntervals = support[1] - support[0];
			unsigned int step = unsigned(ceil(float(samples) / float(supportIntervals)));
			vector<float> y(supportIntervals + 1), o(step * supportIntervals);
			vector<float> prime_y;
			float *p = &membership[support[0]];
			float *q = &y[0];
			cudaStatus = cudaMemcpy(q, p, (supportIntervals + 1) * sizeof(float), cudaMemcpyHostToHost);
			if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::setRange in line 880: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
			}
			float *t = 0;
			float *u = 0;
			for (unsigned int i = 0; i < supportIntervals; i++) {
			t = &o[i * step];
			if (isCUDA) {
			float dx = (y[i + 1] - y[i]) / (step);
			_linearSpace <<<grids, blocks>>>(dev_result1, y[i], dx);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::setRange in line 849: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
			}
			cudaStatus = cudaMemcpy(t, dev_result1, step * sizeof(float), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data copy failed in fs::setRange in line 856: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
			}
			}
			else {
			prime_y = linearSpace(y[i], y[i + 1], step + 1);
			u = &prime_y[0];
			cudaStatus = cudaMemcpy(t, u, step * sizeof(float), cudaMemcpyHostToHost);
			if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::setRange in line 892: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
			}
			}
			}
			u = &o[0];
			t = &membership[0];
			cudaStatus = cudaMemcpy(t, u, samples * sizeof(float), cudaMemcpyHostToHost);
			if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::setRange in line 901: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
			}
			setSupport();
			defuzzification();
			}
			range = { lower, upper };*/
}

void fs::setName(const string &proposed_name) {
	name = proposed_name;
}

void fs::setShape(const string &proposed_shape) {
	shape = proposed_shape;
	try {
		check4MFAvail();
	}
	catch (mfShapeFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}

void fs::setNorm(const float &proposed_normalization) {
	normalization = proposed_normalization;
	try {
		check4MaxNorm();
	}
	catch (maxNormFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	if (parameters.size() != 0)
		setNorm();
}

void fs::setNorm() {
	vector<unsigned int> index = maxIdx(&membership);
	unsigned int k = 0;
	if (index[0] == 0 && index[1] == 1)
		k = index[0];
	else if (index[0] == (samples - 2) && index[1] == (samples - 1))
		k = index[1];
	else {
		for (unsigned int h = 0; h < index.size(); h++)
			k += index[h];
		k = floor(k / float(index.size()));
	}
	if (index[0] == 0 && index[1] == (samples - 1))
		normalization = max(membership);
	else
		normalization = membership[k];
	idxOfMax = k;
}

void fs::setParams(const vector<float> &proposed_parameters) {
	parameters = proposed_parameters;
	// Verify if parameters are sorted in ascending order
	if (shape.compare("Gaussian") != 0 && shape.compare("Singleton") != 0) {
		try {
			check4ordinality();
		}
		catch (ordinalityFault &e) {
			cerr << e.what() << endl;
			cin.get();
			exit(1);
		}
	}
	// Verify if parameter number corresponds to the membership function parameter requirements
	try {
		check4paramNumb();
	}
	catch (paramNumbFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	// Verify if some parameters generate a NaN expression
	try {
		check4NaN();
	}
	catch (paramNaNFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}
void fs::setConjOp(const string &op) {
	conjOperator = op;
	try {
		check4ConjOpAvail();
	}
	catch (conjOpFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}
void fs::setDisjOp(const string &op) {
	disjOperator = op;
	try {
		check4DisjOpAvail();
	}
	catch (disjOpFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}
void fs::setAggregMethod(const string &agg) {
	aggregMethod = agg;
	try {
		check4AggregMethodAvail();
	}
	catch (aggregMethodFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}
void fs::setDefuzzMethod(const string &def) {
	defuzzMethod = def;
	try {
		check4DefuzzAvail();
	}
	catch (defuzzMethodFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}

void fs::setAvailableFunctions() {
	vector<string> mf = { "Singleton",
		"Interval",
		"Triangular",
		"Trapezoidal",
		"S",
		"Z",
		"Gaussian", 
		"Unknown" };
	vector<string> tn = { "Minimum",
		"Product",
		"Bounded",
		"Drastic" };
	vector<string> sn = { "Maximum",
		"Algebraic Sum",
		"Bounded",
		"Drastic" };
	vector<string> def = { "Adaptive Integration",
		"Basic Distributions",
		"Bisector Of Area",
		"Constraint Decisions",
		"Center Of Area",
		"Centroid",
		"Extended Center Of Area",
		"Extended Quality",
		"Fuzzy Clustering",
		"Fuzzy Mean",
		"General Level Set",
		"Indexed Centroid",
		"Influence Value",
		"Smallest Of Maximum",
		"Largest Of Maximum",
		"Mean Of Maxima",
		"Quality",
		"Random Choice Maximum",
		"Semi Linear",
		"Weighted Fuzzy Mean",
		"Geometric" };
	availableMF = mf;
	availableTNorms = tn;
	availableSNorms = sn;
	availableDefuzz = def;
	return;
}

void fs::setDefaultFunctions() {
	shape = availableMF[7];
	conjOperator = availableTNorms[0];
	disjOperator = availableSNorms[0];
	aggregMethod = availableSNorms[0];
	defuzzMethod = availableDefuzz[5];
	discourse.resize(samples);
	membership.resize(samples);
	support = {};
}
void fs::setSupport() {
	support = getSupport(float(0));
}

void fs::setMembership(const vector<float> &memb) {
	try {
		check4SizeMismatch(memb, membership);
	}
	catch (sizeMismatchFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	membership = memb;
}

void fs::setOffset(const float &offset) {
	if (shape.compare("Unknown") != 0) {
		vector<float> a(samples);
		membership = a;
		vector<float> p = parameters;

		if (shape.compare("Singleton") == 0)
			p[0] += offset;
		else if (shape.compare("Interval") == 0) {
			p[0] += offset;
			p[1] += offset;
		}
		else if (shape.compare("Triangular") == 0) {
			p[0] += offset;
			p[1] += offset;
			p[2] += offset;
		}
		else if (shape.compare("Trapezoidal") == 0) {
			p[0] += offset;
			p[1] += offset;
			p[2] += offset;
			p[3] += offset;
		}
		else if (shape.compare("S") == 0 || shape.compare("Z") == 0) {
			p[0] += offset;
			p[1] += offset;
		}
		else
			p[0] += offset;
		setParams(p);
		fuzzify();
		normalization = max(membership);
		setSupport();
		defuzzification();
	}
	else {
		vector<float> newMembership(samples);
		unsigned int indexOfShifter;
		int m;
		for (unsigned int i = 0; i < samples; i++) {
			if (offset > discourse[samples - 1])
				indexOfShifter = samples - 1;
			else {
				if (discourse[i] >= offset) {
					indexOfShifter = i;
					break;
				}
			}
		}

		if (offset < 0) {
			m = idxOfCtr - indexOfShifter;
			if (m > 0)
				for (unsigned int i = 0; i < samples; i++)
					if (m + i <= samples - 1)
						newMembership[i] = membership[m + i];
					else
						newMembership[i] = membership[samples - 1];
		}
		else {
			m = indexOfShifter - idxOfCtr;
			if (m > 0)
				for (int i = 0; i < samples; i++) {
					int n = (int(samples) - 1 - i) - m;
					if (n >= 0)
						newMembership[samples - 1 - i] = membership[(samples - 1 - i) - m];
					else
						newMembership[samples - 1 - i] = membership[0];
				}
		}
		if (m > 0) {
			membership = newMembership;
			normalization = max(membership);
			setSupport();
			defuzzification();
		}
	}
}

// FUZY SET: GET FUNCTIONS

string fs::getName() const {
	return name;
}

string fs::getShape() const {
	return shape;
}

float fs::getNormalization() const {
	return normalization;
}

vector<float> fs::getRange() const {
	return {discourse[0], discourse[samples - 1]};
}

vector<float> fs::getParameters() const {
	return parameters;
}

vector<unsigned int> fs::getSupport() {	
	return support;
}

vector<unsigned int> fs::getSupport(const float &alpha) {
	vector<unsigned int> result;
	//static bool r[samples];
	//if (isCUDA) {
	//	float *p = &membership[0];
	//	// Transfer data from host to device
	//	cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float),
	//		cudaMemcpyHostToDevice);
	//	if (cudaStatus != cudaSuccess) {
	//		cerr << "Host to device data transfer failed in fs::getSupport (operand 1) in line 1140: "
	//			<< cudaGetErrorString(cudaStatus) << endl;
	//		cin.get();
	//		exit(1);
	//	}
	//	// Launch the kernel
	//	_fsCmpGreat <<<grids, blocks>>>(dev_boolResult, dev_op1, alpha);
	//	cudaStatus = cudaGetLastError();
	//	if (cudaStatus != cudaSuccess) {
	//		cerr << "Kernel launch failed in fs::getSupport in line 1149: "
	//			<< cudaGetErrorString(cudaStatus) << endl;
	//		cin.get();
	//		exit(1);
	//	}
	//	// Transfer data back from device to host
	//	cudaStatus = cudaMemcpy(r, dev_boolResult, samples * sizeof(bool),
	//		cudaMemcpyDeviceToHost);
	//	if (cudaStatus != cudaSuccess) {
	//		cerr << "Device to host data transfer failed in fs::getSupport (result) in line 1158: "
	//			<< cudaGetErrorString(cudaStatus) << endl;
	//		cin.get();
	//		exit(1);
	//	}
	//	for (unsigned int i = 0; i < samples - 1; i++) {
	//		if (r[0]) {
	//			result.push_back(0);
	//			break;
	//		}
	//		else {
	//			if (r[i] == false && r[i + 1] == true) {
	//				result.push_back(i);
	//				break;
	//			}
	//		}
	//	}
	//	if (result.size() != 0) {
	//		unsigned int j;
	//		for (unsigned int i = 0; i < samples - 1; i++) {
	//			j = samples - i - 1;
	//			if (r[samples - 1]) {
	//				result.push_back(samples - i - 1);
	//				break;
	//			}
	//			else {
	//				if ((!r[j] && r[j - 1])) {
	//					result.push_back(j);
	//					break;
	//				}
	//			}
	//		}
	//	}
	//	else {
	//		result = { 0, samples - 1 };
	//	}
	//}
	//else {
		for (unsigned int i = 0; i < samples; i++) {
			if (membership[i] > alpha) {
				if (i == 0)
					result.push_back(0);
				else
					result.push_back(i - 1);
				break;
			}
		}
		if (result.size() != 0) {
			unsigned int j;
			for (unsigned int i = 0; i < samples; i++) {
				j = samples - i - 1;
				if (membership[j] > alpha) {
					if (j == samples - 1)
						result.push_back(samples - 1);
					else
						result.push_back(j + 1);
					break;
				}
			}
		}
		else
			result = { 0, samples - 1 };
	//}
	return result;
}

vector<unsigned int> fs::getSupport(vector<float> *operand, const float &alpha) {
	vector<unsigned int> result;
	//static bool r[samples];
	//if (isCUDA) {
	//	float *p = &(*operand)[0];
	//	// Transfer data from host to device
	//	cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float),
	//		cudaMemcpyHostToDevice);
	//	if (cudaStatus != cudaSuccess) {
	//		cerr << "Host to device data transfer failed in fs::getSupport (operand 1) in line 1140: "
	//			<< cudaGetErrorString(cudaStatus) << endl;
	//		cin.get();
	//		exit(1);
	//	}
	//	// Launch the kernel
	//	_fsCmpGreatEq <<<grids, blocks>>>(dev_boolResult, dev_op1, alpha);
	//	cudaStatus = cudaGetLastError();
	//	if (cudaStatus != cudaSuccess) {
	//		cerr << "Kernel launch failed in fs::getSupport in line 1149: "
	//			<< cudaGetErrorString(cudaStatus) << endl;
	//		cin.get();
	//		exit(1);
	//	}
	//	// Transfer data back from device to host
	//	cudaStatus = cudaMemcpy(r, dev_boolResult, samples * sizeof(bool),
	//		cudaMemcpyDeviceToHost);
	//	if (cudaStatus != cudaSuccess) {
	//		cerr << "Device to host data transfer failed in fs::getSupport (result) in line 1158: "
	//			<< cudaGetErrorString(cudaStatus) << endl;
	//		cin.get();
	//		exit(1);
	//	}
	//	for (unsigned int i = 0; i < samples - 1; i++) {
	//		if (r[0]) {
	//			result.push_back(0);
	//			break;
	//		}
	//		else {
	//			if (r[i] == false && r[i + 1] == true) {
	//				result.push_back(i);
	//				break;
	//			}
	//		}
	//	}
	//	if (result.size() != 0) {
	//		unsigned int j;
	//		for (unsigned int i = 0; i < samples - 1; i++) {
	//			j = samples - i - 1;
	//			if (r[samples - 1]) {
	//				result.push_back(samples - i - 1);
	//				break;
	//			}
	//			else {
	//				if ((!r[j] && r[j - 1])) {
	//					result.push_back(j);
	//					break;
	//				}
	//			}
	//		}
	//	}
	//	else {
	//		result = { 0, samples - 1 };
	//	}
	//}
	//else {
		for (unsigned int i = 0; i < samples; i++) {
			if ((*operand)[i] >= alpha) {
				if (i == 0)
					result.push_back(0);
				else
					result.push_back(i - 1);
				break;
			}
		}
		if (result.size() != 0) {
			unsigned int j;
			for (unsigned int i = 0; i < samples; i++) {
				j = samples - i - 1;
				if ((*operand)[j] >= alpha) {
					if (j == samples - 1)
						result.push_back(samples - 1);
					else
						result.push_back(j + 1);
					break;
				}
			}
		}
		else
			result = { 0, samples - 1 };
	//}
	return result;
}

string fs::getConjOp() const {
	return conjOperator;
}

string fs::getDisjOp() const {
	return disjOperator;
}

string fs::getAggregMethod() const {
	return aggregMethod;
}

string fs::getDefuzzMethod() const {
	return defuzzMethod;
}

vector<float> fs::getDiscourse() const {
	return discourse;
}

vector<float> fs::getMembership() const {
	return membership;
}


// FUZZY SET: FUZZIFICATION
void fs::fuzzify() {
	if (shape.compare("Singleton") == 0)
		singletonMF();
	else if (shape.compare("Interval") == 0)
		intervalMF();
	else if (shape.compare("Triangular") == 0)
		triangularMF();
	else if (shape.compare("Trapezoidal") == 0)
		trapezoidalMF();
	else if (shape.compare("S") == 0)
		sMF();
	else if (shape.compare("Z") == 0)
		zMF();
	else if (shape.compare("Gaussian") == 0)
		gaussianMF();
	setNorm();
}

float fs::fuzzify(const float &crisp_input, const unsigned int &index) {
	float result;
	unsigned int left = index, right = samples - 1;
	float x, y;
	if (crisp_input <= discourse[0])
		return membership[0];
	else if (crisp_input >= discourse[samples - 1])
		return membership[samples - 1];
	else {
		if (left < samples - 1)
			if (discourse[left] == crisp_input)
				right = left;
			else
				right = left + 1;
		else
			right = left;

		if (left == right) {
			return membership[right];
		}
		else {
			x = discourse[right] - discourse[left];
			y = membership[right] - membership[left];
			if ((x == 0) && (y == 0))
				return float(0.0);
			else
				if (y >= 0)
					result = abs((crisp_input - discourse[left]) * y / x) + membership[left];
				else
					result = abs((crisp_input - discourse[left]) * y / x) + membership[right];
		}
	}
	return result;
}

// FUZZY SET: DEFUZZIFICATION
float fs::defuzzify() const {
	return crisp;
}
void fs::defuzzification() {
	if (defuzzMethod.compare("Adaptive Integration") == 0)
		crisp = adaptiveIntegration();
	else if (defuzzMethod.compare("Basic Distributions") == 0)
		crisp = basicDefuzzDistributions();
	else if (defuzzMethod.compare("Bisector Of Area") == 0)
		crisp = bisectorOfArea();
	else if (defuzzMethod.compare("Constraint Decisions") == 0)
		crisp = constraintDecision();
	else if (defuzzMethod.compare("Center Of Area") == 0)
		crisp = centerOfArea();
	else if (defuzzMethod.compare("Centroid") == 0)
		crisp = centroid();
	else if (defuzzMethod.compare("Extended Center Of Area") == 0)
		crisp = extendedCenterOfArea();
	else if (defuzzMethod.compare("Extended Quality") == 0)
		crisp = extendedQuality();
	else if (defuzzMethod.compare("Fuzzy Clustering") == 0)
		crisp = fuzzyClustering();
	else if (defuzzMethod.compare("Fuzzy Mean") == 0)
		crisp = fuzzyMean();
	else if (defuzzMethod.compare("General Level Set") == 0)
		crisp = generalLevelSet();
	else if (defuzzMethod.compare("Indexed Centroid") == 0)
		crisp = indexedCenterOfGravity();
	else if (defuzzMethod.compare("Influence Value") == 0)
		crisp = influenceValue();
	else if (defuzzMethod.compare("Smallest Of Maximum") == 0)
		crisp = smallestOfMaximum();
	else if (defuzzMethod.compare("Largest Of Maximum") == 0)
		crisp = largestOfMaximum();
	else if (defuzzMethod.compare("Mean Of Maxima") == 0)
		crisp = meanOfMaxima();
	else if (defuzzMethod.compare("Quality") == 0)
		crisp = quality();
	else if (defuzzMethod.compare("Random Choice Maximum") == 0)
		crisp = randomChoiceOfMaximum();
	else if (defuzzMethod.compare("Semi Linear") == 0)
		crisp = semiLinear();
	else if (defuzzMethod.compare("Weighted Fuzzy Mean") == 0)
		crisp = weightedFuzzyMean();
	else if (defuzzMethod.compare("Geometric") == 0)
		crisp = geometric();

	for (unsigned int i = 0; i < samples; i++)
		if (discourse[i] >= crisp) {
			idxOfCtr = i;
			break;
		}
}

// OVERLOADED OPERATORS
// FUZZY SET: ARITHMETIC OPERATORS
fs fs::operator+(const fs &right) { // FUZZY SUM
	vector<unsigned int> intervalA(2), intervalB(2);
	vector<fs> denormalizedSets(2, fs(isCUDA, isStream));
	vector<fs> normalizedSets;
	// Fuzzy set normalization.
	fs setA(isCUDA, isStream), setB(isCUDA, isStream);
	if (this->normalization != right.normalization) {
		// The highest membership set normalizes the lowest membership set
		denormalizedSets = { *this, right };
		normalizedSets = normalize(denormalizedSets);
		// Both normalized sets are ready for computing the operation
		setA = normalizedSets[0];
		setB = normalizedSets[1];
	}
	else {
		setA = *this;
		setB = right;
	}
	this->name = "(" + this->name + ")" + " + " + "(" + right.name + ")";
	this->shape = "Unknown";
	this->parameters = {};
	// Get both set supports
	intervalA = setA.getSupport();
	intervalB = setB.getSupport();
	// Result discourse definition using both set supports
	setA.setRange(setA.discourse[intervalA[0]], setA.discourse[intervalA[1]], true);
	setB.setRange(setB.discourse[intervalB[0]], setB.discourse[intervalB[1]], true);
	// Operand computation
	if (isCUDA) {
		if (setA.discourse.size() != 0) {
			float *r = &this->discourse[0];
			float *s = &this->membership[0];
			float *p = &setA.discourse[0];
			float *q = &setB.discourse[0];
			float *m = &setA.membership[0];
			float *n = &setB.membership[0];
			// Transfer data from host to device
			cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), 
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator+ (operand 1) in line 1491: " 
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op2, q, samples * sizeof(float), 
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator+ (operand 2) in line 1499: " 
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op3, m, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator+ (operand 1) in line 1491: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op4, n, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator+ (operand 2) in line 1499: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			// Launch the kernel
			_fsAdd <<<grids, blocks>>>(dev_result1, dev_result2, dev_op1, dev_op2, dev_op3, dev_op4);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				cerr << "Kernel launch failed in fs::operator+ in line 1508:" 
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			// Transfer data back from device to host
			cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), 
				cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cerr << "Device to host data transfer failed in fs::operator+ (result) in line 1517: " 
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(s, dev_result2, samples * sizeof(float),
				cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cerr << "Device to host data transfer failed in fs::operator+ (result) in line 1517: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
		}
	}
	else {
		for (unsigned int i = 0; i < samples; i++) {
			// New discourse universe
			this-> discourse[i] = setA.discourse[i] + setB.discourse[i];
			// New membership degrees
			this-> membership[i] = (setA.membership[i] + setB.membership[i]) / 2;
		}
	}
	this->isCUDA = setA.isCUDA | setB.isCUDA;
	this->setSupport();
	this->defuzzification();
	this->setNorm();
	return *this;
}

fs fs::operator-(const fs &right) {// FUZZY SUBTRACT
	// Data declaration and initialization
	vector<unsigned int> intervalA(2), intervalB(2);
	vector<fs> denormalizedSets(2, fs(isCUDA, isStream));
	vector<fs> normalizedSets;
	// Fuzzy set normalization.
	fs setA(isCUDA, isStream), setB(isCUDA, isStream);
	if (this->normalization != right.normalization) {
		// The highest membership set normalizes the lowest membership set
		denormalizedSets = { *this, right };
		normalizedSets = normalize(denormalizedSets);
		// Both normalized sets are ready for computing the operation
		setA = normalizedSets[0];
		setB = normalizedSets[1];
	}
	else {
		setA = *this;
		setB = right;
	}
	this->name = "(" + this->name + ")" + " - " + "(" + right.name + ")";
	this->shape = "Unknown";
	this->parameters = {};
	// Get both set supports
	intervalA = setA.getSupport();
	intervalB = setB.getSupport();
	// Result discourse definition using both set supports
	setA.setRange(setA.discourse[intervalA[0]], setA.discourse[intervalA[1]], true);
	setB.setRange(setB.discourse[intervalB[0]], setB.discourse[intervalB[1]], true);
	// Operand computation
	if (isCUDA) {
		if (setA.discourse.size() != 0) {
			float *r = &this-> discourse[0];
			float *s = &this-> membership[0];
			float *p = &setA.discourse[0];
			float *q = &setB.discourse[0];
			float *m = &setA.membership[0];
			float *n = &setB.membership[0];
			// Transfer data from host to device
			cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator- (operand 1) in line 1638: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op2, q, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator- (operand 2) in line 1646: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op3, m, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator+ (operand 1) in line 1491: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op4, n, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator+ (operand 2) in line 1499: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			// Launch the kernel
			_fsSub <<<grids, blocks>>>(dev_result1, dev_result2, dev_op1, dev_op2, dev_op3, dev_op4);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				cerr << "Kernel launch failed in fs::operator- in line 1655: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			// Transfer data back from device to host
			cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float),
				cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cerr << "Device to host data transfer failed in fs::operator- (result) in line 1664: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(s, dev_result2, samples * sizeof(float),
				cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cerr << "Device to host data transfer failed in fs::operator+ (result) in line 1517: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
		}
	}
	else {
		for (unsigned int i = 0; i < samples; i++) {
			// New discourse universe
			this-> discourse[i] = setA.discourse[i] - setB.discourse[samples - 1 - i];
			// New membership degrees
			this-> membership[i] = (setA.membership[i] + setB.membership[i]) / 2;
		}
	}
	this-> isCUDA = setA.isCUDA | setB.isCUDA;
	this-> setSupport();
	this-> defuzzification();
	this-> setNorm();
	return *this;
}

fs fs::operator*(const fs &right) { // FUZZY MULTIPLICATION
	// Data declaration and initialization
	vector<unsigned int> intervalA(2), intervalB(2);
	vector<fs> denormalizedSets(2, fs(isCUDA, isStream));
	vector<fs> normalizedSets;
	// Fuzzy set normalization.
	fs setA(isCUDA, isStream), setB(isCUDA, isStream);
	if (this->normalization != right.normalization) {
		// The highest membership set normalizes the lowest membership set
		denormalizedSets = { *this, right };
		normalizedSets = normalize(denormalizedSets);
		// Both normalized sets are ready for computing the operation
		setA = normalizedSets[0];
		setB = normalizedSets[1];
	}
	else {
		setA = *this;
		setB = right;
	}
	this->name = "(" + this->name + ")" + " * " + "(" + right.name + ")";
	this->shape = "Unknown";
	this->parameters = {};
	// Get both set supports
	intervalA = setA.getSupport();
	intervalB = setB.getSupport();
	// Result discourse definition using both set supports
	setA.setRange(setA.discourse[intervalA[0]], setA.discourse[intervalA[1]], true);
	setB.setRange(setB.discourse[intervalB[0]], setB.discourse[intervalB[1]], true);
	// Operand computation
	if (isCUDA) {
		if (setA.discourse.size() != 0) {
			float *r = &this->discourse[0];
			float *s = &this->membership[0];
			float *p = &setA.discourse[0];
			float *q = &setB.discourse[0];
			float *m = &setA.membership[0];
			float *n = &setB.membership[0];
			// Transfer data from host to device
			cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator* (operand 1) in line 1786: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op2, q, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator* (operand 2) in line 1794: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op3, m, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator+ (operand 1) in line 1491: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op4, n, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator+ (operand 2) in line 1499: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			// Launch the kernel
			_fsMul <<<grids, blocks>>>(dev_result1, dev_result2, dev_op1, dev_op2, dev_op3, dev_op4);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				cerr << "Kernel launch failed in fs::operator* in line 1803: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			// Transfer data back from device to host
			cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float),
				cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cerr << "Device to host data transfer failed in fs::operator* (result) in line 1812: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(s, dev_result2, samples * sizeof(float),
				cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cerr << "Device to host data transfer failed in fs::operator+ (result) in line 1517: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
		}
	}
	else {
		for (unsigned int i = 0; i < samples; i++) {
			// New discourse universe
			this-> discourse[i] = setA.discourse[i] * setB.discourse[i];
			// New membership degrees
			this-> membership[i] = (setA.membership[i] + setB.membership[i]) / 2;
		}
	}
	this-> isCUDA = setA.isCUDA | setB.isCUDA;
	this-> setSupport();
	this-> defuzzification();
	this->setNorm();
	return *this;
}

fs fs::operator/(const fs &right) { // FUZZY DIVISION
	// Data declaration and initialization
	vector<unsigned int> intervalA(2), intervalB(2);
	vector<fs> denormalizedSets(2, fs(isCUDA, isStream));
	vector<fs> normalizedSets;
	// Fuzzy set normalization.
	fs setA(isCUDA, isStream), setB(isCUDA, isStream);
	if (this->normalization != right.normalization) {
		// The highest membership set normalizes the lowest membership set
		denormalizedSets = { *this, right };
		normalizedSets = normalize(denormalizedSets);
		// Both normalized sets are ready for computing the operation
		setA = normalizedSets[0];
		setB = normalizedSets[1];
	}
	else {
		setA = *this;
		setB = right;
	}
	this->name = "(" + this->name + ")" + " / " + "(" + right.name + ")";
	this->shape = "Unknown";
	this->parameters = {};
	// Get both set supports
	intervalA = setA.getSupport();
	intervalB = setB.getSupport();
	// Result discourse definition using both set supports
	setA.setRange(setA.discourse[intervalA[0]], setA.discourse[intervalA[1]], true);
	setB.setRange(setB.discourse[intervalB[0]], setB.discourse[intervalB[1]], true);
	// Operand computation
	if (isCUDA) {
		if (setA.discourse.size() != 0) {
			float *r = &this-> discourse[0];
			float *s = &this-> membership[0];
			float *p = &setA.discourse[0];
			float *q = &setB.discourse[0];
			float *m = &setA.membership[0];
			float *n = &setB.membership[0];
			// Transfer data from host to device
			cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator/ (operand 1) in line 1933: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op2, q, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator/ (operand 2) in line 1941: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op3, m, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator+ (operand 1) in line 1491: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op4, n, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator+ (operand 2) in line 1499: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			// Launch the kernel
			_fsDiv <<<grids, blocks>>>(dev_result1, dev_result2, dev_op1, dev_op2, dev_op3, dev_op4);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				cerr << "Kernel launch failed in fs::operator/ in line 1950: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			// Transfer data back from device to host
			cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float),
				cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cerr << "Device to host data transfer failed in fs::operator/ (result) in line 1959: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(s, dev_result2, samples * sizeof(float),
				cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cerr << "Device to host data transfer failed in fs::operator+ (result) in line 1517: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
		}
	}
	else {
		for (unsigned int i = 0; i < samples; i++) {
			// New discourse universe
			if (setB.discourse[i] == 0){
				cerr << "Division by zero is not valid for fs::operator/  in line 2011. "
					<< "Discourse universe must have non-zero values" << endl;
				cin.get();
				exit(1);
			}
			this-> discourse[i] = setA.discourse[i] / setB.discourse[i];
			// New membership degrees
			this-> membership[i] = (setA.membership[i] + setB.membership[i]) / 2;
		}
	}
	this-> isCUDA = setA.isCUDA | setB.isCUDA;
	this-> setSupport();
	this-> defuzzification();
	this->setNorm();
	return *this;
}

fs fs::operator%(const fs &right) { // FUZZY MODULUS
	// Data declaration and initialization
	vector<unsigned int> intervalA(2), intervalB(2);
	vector<fs> denormalizedSets(2, fs(isCUDA, isStream));
	vector<fs> normalizedSets;
	// Fuzzy set normalization.
	fs setA(isCUDA, isStream), setB(isCUDA, isStream);
	if (this->normalization != right.normalization) {
		// The highest membership set normalizes the lowest membership set
		denormalizedSets = { *this, right };
		normalizedSets = normalize(denormalizedSets);
		// Both normalized sets are ready for computing the operation
		setA = normalizedSets[0];
		setB = normalizedSets[1];
	}
	else {
		setA = *this;
		setB = right;
	}
	this->name = "(" + this->name + ")" + " % " + "(" + right.name + ")";
	this->shape = "Unknown";
	this->parameters = {};
	// Get both set supports
	intervalA = setA.getSupport();
	intervalB = setB.getSupport();
	// Result discourse definition using both set supports
	setA.setRange(setA.discourse[intervalA[0]], setA.discourse[intervalA[1]], true);
	setB.setRange(setB.discourse[intervalB[0]], setB.discourse[intervalB[1]], true);
	// Operand computation
	if (isCUDA) {
		if (setA.discourse.size() != 0) {
			float *r = &this-> discourse[0];
			float *s = &this-> membership[0];
			float *p = &setA.discourse[0];
			float *q = &setB.discourse[0];
			float *m = &setA.membership[0];
			float *n = &setB.membership[0];
			// Transfer data from host to device
			cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator% (operand 1) in line 2085: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op2, q, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator% (operand 2) in line 2093: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op3, m, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator+ (operand 1) in line 1491: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(dev_op4, n, samples * sizeof(float),
				cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::operator+ (operand 2) in line 1499: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			// Launch the kernel
			_fsMod <<<grids, blocks>>>(dev_result1, dev_result2, dev_op1, dev_op2, dev_op3, dev_op4);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				cerr << "Kernel launch failed in fs::operator% in line 2102: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			// Transfer data back from device to host
			cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float),
				cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cerr << "Device to host data transfer failed in fs::operator% (result) in line 2111: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			cudaStatus = cudaMemcpy(s, dev_result2, samples * sizeof(float),
				cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cerr << "Device to host data transfer failed in fs::operator+ (result) in line 1517: "
					<< cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
		}
	}
	else {
		for (unsigned int i = 0; i < samples; i++){
			// New discourse universe
			if (setB.discourse[i] == 0) {
				cerr << "Division by zero is not valid for fs::operator%  in line 2164. "
					<< "Discourse universe must have non-zero values" << endl;
				cin.get();
				exit(1);
			}
			this-> discourse[i] = static_cast<int>(round(setA.discourse[i])) %
				static_cast<int>(ceil(setB.discourse[i]));
			// New membership degrees
			this-> membership[i] = (setA.membership[i] + setB.membership[i]) / 2;
		}
	}
	this-> isCUDA = setA.isCUDA | setB.isCUDA;
	this-> setSupport();
	this-> defuzzification();
	this->setNorm();
	return *this;
}

// FUZZY SET: COMPARISON OPERATORS
bool fs::operator==(const fs &right) const {// Fuzzy Equality
	fs left(isCUDA, isStream);
	left = *this;
	float y1 = left.crisp;
	float y2 = right.crisp;
	if (y1 == y2)
		return true;
	else
		return false;
}

bool fs::operator!=(const fs &right) const {// Fuzzy Unequality
	float y1 = this->crisp;
	float y2 = right.crisp;
	if (y1 != y2)
		return true;
	else
		return false;
}

bool fs::operator<(const fs &right) const {// Fuzzy Lower Than
	float y1 = this->crisp;
	float y2 = right.crisp;
	if (y1 < y2)
		return true;
	else
		return false;
}

bool fs::operator>(const fs &right) const {// Fuzzy Greater Than
	float y1 = this->crisp;
	float y2 = right.crisp;
	if (y1 > y2)
		return true;
	else
		return false;
}

bool fs::operator<=(const fs &right) const {// Fuzzy Lower or Equal Than
	fs left(isCUDA, isStream);
	left = *this;
	float y1 = left.crisp;
	float y2 = right.crisp;
	if (y1 <= y2)
		return true;
	else
		return false;
}

bool fs::operator>=(const fs &right) const {// Fuzzy Greater or Equal Than
	float y1 = this->crisp;
	float y2 = right.crisp;
	if (y1 >= y2)
		return true;
	else
		return false;
}

// FUZZY SET: LOGIC OPERATORS
fs fs::operator&(const fs &right) const {// AND
	fs result(isCUDA, isStream);
	result.name = "(" + this->name + ")" + " AND " + "(" + right.name + ")";
	result.shape = "Unknown";
	result.parameters = {};
	result.membership = tnorm(this->conjOperator, this->membership, right.membership);
	result.isCUDA = this->isCUDA | right.isCUDA;
	result.setRange(this->range[0], this->range[1]);
	result.setSupport();
	result.defuzzification();
	result.setNorm();
	return result;
}

fs fs::operator&&(const fs &right) const {// AND
	fs result(isCUDA, isStream);
	result.name = "(" + this->name + ")" + " AND " + "(" + right.name + ")";
	result.shape = "Unknown";
	result.parameters = {};
	result.membership = tnorm("Minimum", this->membership, right.membership);
	result.isCUDA = this->isCUDA | right.isCUDA;
	result.setRange(this->range[0], this->range[1]);
	result.setSupport();
	result.defuzzification();
	result.setNorm();
	return result;
}

fs fs::operator|(const fs &right) const {// OR
	fs result(isCUDA, isStream);
	result.name = "(" + this->name + ")" + " OR " + "(" + right.name + ")";
	result.shape = "Unknown";
	result.parameters = {};
	result.membership = snorm(this->disjOperator, this->membership, right.membership);
	result.isCUDA = this->isCUDA | right.isCUDA;
	result.setRange(this->range[0], this->range[1]);
	result.setSupport();
	result.defuzzification();
	result.setNorm();
	return result;
}

fs fs::operator||(const fs &right) const {// AND
	fs result(isCUDA, isStream);
	result.name = "(" + this->name + ")" + " AND " + "(" + right.name + ")";
	result.shape = "Unknown";
	result.parameters = {};
	result.membership = snorm("Maximum", this->membership, right.membership);
	result.isCUDA = this->isCUDA | right.isCUDA;
	result.isStream = this->isStream | right.isStream;
	result.setRange(this->range[0], this->range[1]);
	result.setSupport();
	result.defuzzification();
	result.setNorm();
	return result;
}

fs fs::operator!() { // NOT
	fs result(isCUDA, isStream);
	result.name = "NOT(" + this->name + ")";
	result.shape = "Unknown";
	result.parameters = {};
	result.membership = fnot(this->membership);
	result.isCUDA = this->isCUDA;
	result.setRange(this->range[0], this->range[1]);
	result.setSupport();
	result.defuzzification();
	result.setNorm();
	return result;
}

fs fs::operator+=(const fs &right) const {// AGGREGATION
	fs result(isCUDA, isStream);
	result.name = "(" + this->name + ")" + " OR " + "(" + right.name + ")";
	result.shape = "Unknown";
	result.parameters = {};
	result.membership = snorm("Maximum", this->membership, right.membership);
	result.isCUDA = this->isCUDA | right.isCUDA;
	result.setSupport();
	result.defuzzification();
	result.setNorm();
	return result;
}

const fs &fs::operator=(const fs &right) {// ASSIGNMENT
	if (&right != this) {
		this->name = right.name;
		this->shape = right.shape;
		this->range = right.range;
		this->normalization = right.normalization;
		this->parameters = right.parameters;
		this->discourse = right.discourse;
		this->membership = right.membership;
		this->conjOperator = right.conjOperator;
		this->disjOperator = right.disjOperator;
		this->aggregMethod = right.aggregMethod;
		this->defuzzMethod = right.defuzzMethod;
		this->isCUDA = right.isCUDA;
		this->isStream = right.isStream;
		this->support = right.support;
		this->crisp = right.crisp;
		this->idxOfMax = right.idxOfMax;
		this->idxOfCtr = right.idxOfCtr;
	}
	return *this;
}

fs fs::operator&(const float &right) const {
	fs result(isCUDA, isStream);
	ostringstream convert;
	convert << right;
	string str = convert.str();
	fs op(this->discourse[0], this->discourse[samples - 1], right, str, "Interval", 
	{ this->discourse[0], this->discourse[samples - 1] }, this->isCUDA);
	result.name = "(" + this->name + ")" + " AND " + "( " + op.name + " )";
	result.shape = "Unknown";
	result.parameters = {};
	result.membership = tnorm(this->conjOperator, this->membership, op.membership);
	result.isCUDA = this->isCUDA;
	result.setSupport();
	result.defuzzification();
	result.setNorm();
	return result;
}

fs fs::operator|(const float &right) const {
	fs result(isCUDA, isStream);
	ostringstream convert;
	convert << right;
	string str = convert.str();
	fs op(this->discourse[0], this->discourse[samples - 1], right, str, "Interval", 
	{ this->discourse[0], this->discourse[samples - 1] }, this->isCUDA);
	result.name = "(" + this->name + ")" + " OR " + "( " + op.name + " )";
	result.shape = "Unknown";
	result.parameters = {};
	result.membership = snorm(this->conjOperator, this->membership, op.membership);
	result.isCUDA = this->isCUDA;
	result.setSupport();
	result.defuzzification();
	result.setNorm();
	return result;
}

// FUZZY SET: MEMBERSHIP FUNCTIONS
float fs::singletonMF(const float &crisp) const {
	float result;
	if (crisp >= 1.01 * parameters[0] || crisp <= 0.99 * parameters[0])
		result = 0;
	else
		result = normalization;
	return result;
}

void fs::singletonMF() {	
	unsigned int left = 0, right = samples - 1;
	float mid;
	for (unsigned int i = 0; i < samples; i++) {
		if (discourse[i] >= parameters[0]) {
			if (i == 0)
				right = 0;
			else if (i == samples - 1)
				right = samples - 1;
			else
				right = i;
			break;
		}
	}
	unsigned int j;
	for (unsigned int i = 0; i < samples; i++) {
		j = samples - i - 1;
		if (discourse[j] <= parameters[0]) {
			if (j == samples - 1)
				left = samples - 1;
			else if (j == 0)
				left = 0;
			else
				left = j;
			break;
		}
	}
	mid = (discourse[left] + discourse[right]) / 2;
	if (parameters[0] < mid)
		membership[left] = 1;
	else
		membership[right] = 1;
}

float fs::intervalMF(const float &crisp) const {
	float result;
	if (crisp > parameters[1] || crisp < parameters[0])
		result = 0;
	else
		result = normalization;
	return result;
}

void fs::intervalMF() {
	if (isCUDA) {
		float *r = &membership[0];
		float *p = &discourse[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::intervalMF (discourse) in line 2449: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		_intervalMF <<<grids, blocks>>>(dev_result1, dev_op1, parameters[0], parameters[1], normalization);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::intervalMF in line 2457:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::intervalMF (membership) in line 2464: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			membership[i] = intervalMF(discourse[i]);
}

float fs::triangularMF(const float &crisp) const {
	float result;
	if (crisp >= parameters[1] && crisp <= parameters[2])
		result = (parameters[2] - crisp) / (parameters[2] - parameters[1]);
	else if (crisp >= parameters[0] && crisp <= parameters[1])
		result = (crisp - parameters[0]) / (parameters[1] - parameters[0]);
	else
		result = 0;
	return result;
}

void fs::triangularMF() {
	if (isCUDA) {
		float *r = &membership[0];
		float *p = &discourse[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::triangularMF (discourse) in line 2508: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		_triangularMF <<<grids, blocks>>>(dev_result1, dev_op1, parameters[0], parameters[1], parameters[2], normalization);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::triangularMF in line 2516:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::triangularMF (membership) in line 2523: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else{
		//chrono::system_clock::time_point begin = chrono::high_resolution_clock::now();
		/*nvtxEventAttributes_t eventAttrib = { 0 };

		eventAttrib.version = NVTX_VERSION;
		eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

		eventAttrib.colorType = NVTX_COLOR_ARGB;
		eventAttrib.color = 0xFFFF0000;
		eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
		eventAttrib.message.ascii = ": TriangularMF";

		nvtxRangeId_t t = nvtxRangeStartEx(&eventAttrib);*/
		for (unsigned int i = 0; i < samples; i++)
			membership[i] = triangularMF(discourse[i]);
		//nvtxRangeEnd(t);
		/*chrono::system_clock::time_point end = chrono::high_resolution_clock::now();
		float a = chrono::duration<float>(end - begin).count();
		cout << "Elapsed time is: " << a << endl;
		cin.get();*/
	}
}

float fs::trapezoidalMF(const float &crisp) const {
	float result;
	if (crisp >= parameters[1] && crisp <= parameters[2])
		result = 1;
	else if (crisp >= parameters[2] && crisp <= parameters[3])
		result = (parameters[3] - crisp) / (parameters[3] - parameters[2]);
	else if (crisp >= parameters[0] && crisp <= parameters[1])
		result = (crisp - parameters[0]) / (parameters[1] - parameters[0]);
	else
		result = 0;
	return result;
}

void fs::trapezoidalMF() {
	if (isCUDA) {
		float *r = &membership[0];
		float *p = &discourse[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::trapezoidalMF (discourse) in line 2569: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		_trapezoidalMF <<<grids, blocks>>>(dev_result1, dev_op1, parameters[0], parameters[1], parameters[2], parameters[3], normalization);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::trapezoidalMF in line 2577:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::trapezoidalMF (membership) in line 2584: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			membership[i] = trapezoidalMF(discourse[i]);
}

float fs::sMF(const float &crisp) const {
	float result;
	if (crisp >= parameters[0] && crisp <= (parameters[0] + parameters[1]) / 2)
		result = float(2.0) * pow((crisp - parameters[0]) / (parameters[1] - parameters[0]), float(2.0));
	else if (crisp >= (parameters[0] + parameters[1]) / 2 && crisp <= parameters[1])
		result = float(1.0) - float(2.0) * pow((crisp - parameters[1]) / (parameters[1] - parameters[0]), float(2.0));
	else if (crisp >= parameters[1])
		result = 1;
	else
		result = float(0.0);
	return result;
}

void fs::sMF() {
	if (isCUDA) {
		float *r = &membership[0];
		float *p = &discourse[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::sMF (discourse) in line 2630: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		_sMF <<<grids, blocks>>>(dev_result1, dev_op1, parameters[0], parameters[1], normalization);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::sMF in line 2638:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::sMF (membership) in line 2645: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			membership[i] = sMF(discourse[i]);
}

float fs::zMF(const float &crisp) const {
	float result;
	if (crisp >= parameters[0] && crisp <= (parameters[0] + parameters[1]) / 2)
		result = float(1.0) - float(2.0) * pow((crisp - parameters[0]) / (parameters[1] - parameters[0]), float(2.0));
	else if (crisp >= (parameters[0] + parameters[1]) / 2 && crisp <= parameters[1])
		result = float(2.0) * pow((crisp - parameters[1]) / (parameters[1] - parameters[0]), float(2.0));
	else if (crisp >= parameters[1])
		result = float(0.0);
	else
		result = 1;
	return result;
}

void fs::zMF() {
	if (isCUDA) {
		float *r = &membership[0];
		float *p = &discourse[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::zMF (discourse) in line 2691: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		_zMF <<<grids, blocks>>>(dev_result1, dev_op1, parameters[0], parameters[1], normalization);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::zMF in line 2699:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::zMF (membership) in line 2706: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			membership[i] = zMF(discourse[i]);
}

float fs::gaussianMF(const float &crisp) const {
	return normalization * exp(-pow(crisp - parameters[0], float(2.0)) / (float(2.0) * pow(parameters[1], float(2.0))));
}

void fs::gaussianMF() {
	if (isCUDA) {
		float *r = &membership[0];
		float *p = &discourse[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::gaussianMF (discourse) in line 2743: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		_gaussianMF <<<grids, blocks>>>(dev_result1, dev_op1, parameters[0], parameters[1], normalization);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::gaussianMF in line 2751:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::gaussianMF (membership) in line 2758: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			membership[i] = gaussianMF(discourse[i]);
}

// FUZZY SET: NORM FUNCTIONS
float fs::tnorm(const string &type, const float &operand1, const float &operand2) const {
	float result;
	if (type.compare("Minimum") == 0)
		result = minimumIntersection(operand1, operand2);
	else if (type.compare("Product") == 0)
		result = productIntersection(operand1, operand2);
	else if (type.compare("Bounded") == 0)
		result = boundedIntersection(operand1, operand2);
	else if (type.compare("Drastic") == 0)
		result = drasticIntersection(operand1, operand2);
	return result;
}

vector<float> fs::tnorm(const string &type, const vector<float> &operand1, const vector<float> &operand2) const {
	vector<float> result(samples);
	if (isCUDA) {
		float *r = &result[0];
		const float *p = &operand1[0];
		const float *q = &operand2[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::tnorm (operand 1) in line 2813: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		cudaStatus = cudaMemcpy(dev_op2, q, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::tnorm (operand 2) in line 2819: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		if (type.compare("Minimum") == 0)
			_minimumIntersection <<<grids, blocks>>>(dev_result1, dev_op1, dev_op2);
		else if (type.compare("Product") == 0)
			_productIntersection <<<grids, blocks>>>(dev_result1, dev_op1, dev_op2);
		else if (type.compare("Bounded") == 0)
			_boundedIntersection <<<grids, blocks>>>(dev_result1, dev_op1, dev_op2, normalization);
		else if (type.compare("Drastic") == 0)
			_drasticIntersection <<<grids, blocks>>>(dev_result1, dev_op1, dev_op2, normalization);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::tnorm in line 2834:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::tnorm (result) in line 2841: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			if (type.compare("Minimum") == 0)
				result[i] = minimumIntersection(operand1[i], operand2[i]);
			else if (type.compare("Product") == 0)
				result[i] = productIntersection(operand1[i], operand2[i]);
			else if (type.compare("Bounded") == 0)
				result[i] = boundedIntersection(operand1[i], operand2[i]);
			else if (type.compare("Drastic") == 0)
				result[i] = drasticIntersection(operand1[i], operand2[i]);
	return result;
}

float fs::snorm(const string &type, const float &operand1, const float &operand2) const {
	float result;
	if (type.compare("Maximum") == 0)
		result = maximumUnion(operand1, operand2);
	else if (type.compare("Algebraic Sum") == 0)
		result = algebraicUnion(operand1, operand2);
	else if (type.compare("Bounded") == 0)
		result = boundedUnion(operand1, operand2);
	else if (type.compare("Drastic") == 0)
		result = drasticUnion(operand1, operand2);
	return result;
}

vector<float> fs::snorm(const string &type, const vector<float> &operand1, const vector<float> &operand2) const {
	vector<float> result(samples);
	if (isCUDA) {
		float *r = &result[0];
		const float *p = &operand1[0];
		const float *q = &operand2[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::snorm (operand 1) in line 2904: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		cudaStatus = cudaMemcpy(dev_op2, q, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::snorm (operand 2) in line 2910: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		if (type.compare("Maximum") == 0)
			_maximumUnion <<<grids, blocks>>>(dev_result1, dev_op1, dev_op2);
		else if (type.compare("Algebraic Sum") == 0)
			_algebraicUnion <<<grids, blocks>>>(dev_result1, dev_op1, dev_op2);
		else if (type.compare("Bounded") == 0)
			_boundedUnion <<<grids, blocks>>>(dev_result1, dev_op1, dev_op2, normalization);
		else if (type.compare("Drastic") == 0)
			_drasticUnion <<<grids, blocks>>>(dev_result1, dev_op1, dev_op2, normalization);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::snorm in line 2925:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::snorm (result) in line 2932: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			if (type.compare("Maximum") == 0)
				result[i] = maximumUnion(operand1[i], operand2[i]);
			else if (type.compare("Algebraic Sum") == 0)
				result[i] = algebraicUnion(operand1[i], operand2[i]);
			else if (type.compare("Bounded") == 0)
				result[i] = boundedUnion(operand1[i], operand2[i]);
			else if (type.compare("Drastic") == 0)
				result[i] = drasticUnion(operand1[i], operand2[i]);
	return result;
}

vector<float> fs::fnot(vector<float> &m) const{
	vector<float> result(samples);
	if (isCUDA) {
		float *r = &result[0];
		const float *p = &m[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::snorm (operand 1) in line 2904: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		_complement <<<grids, blocks>>>(dev_result1, dev_op1);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::snorm in line 2925:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::snorm (result) in line 2932: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			result[i] = 1 - m[i];
	return result;
}

float fs::minimumIntersection(const float &operand1, const float &operand2) const {
	return min(operand1, operand2);
}

float fs::productIntersection(const float &operand1, const float &operand2) const {
	return operand1 * operand2;
}

float fs::boundedIntersection(const float &operand1, const float &operand2) const {
	return maximumUnion(float(0.0), operand1 + operand2 - normalization);
}

float fs::drasticIntersection(const float &operand1, const float &operand2) const {
	float result;
	if (operand1 == normalization)
		result = operand2;
	else if (operand2 == normalization)
		result = operand1;
	else
		result = float(0.0);
	return result;
}

float fs::maximumUnion(const float &operand1, const float &operand2) const {
	return max(operand1, operand2);
}

float fs::algebraicUnion(const float &operand1, const float &operand2) const {
	return operand1 + operand2 - operand1 * operand2;
}

float fs::boundedUnion(const float &operand1, const float &operand2) const {
	return minimumIntersection(normalization, operand1 + operand2);
}

float fs::drasticUnion(const float &operand1, const float &operand2) const {
	float result;
	if (operand1 == float(0.0))
		result = operand2;
	else if (operand2 == float(0.0))
		result = operand1;
	else
		result = normalization;
	return result;
}
float fs::min(const float &operand1, const float &operand2) const {
	float result;
	if (operand1 < operand2)
		result = operand1;
	else
		result = operand2;
	return result;
}

float fs::max(const float &operand1, const float &operand2) const {
	float result;
	if (operand1 > operand2)
		result = operand1;
	else
		result = operand2;
	return result;
}

float fs::min(const vector<float> &operand) const {
	float result = 1;
	for (unsigned int i = 0; i < samples; i++)
		result = min(operand[i], result);
	return result;
}

float fs::max(const vector<float> &operand) const {
	float result = float(0.0);
	for (unsigned int i = 0; i < samples; i++)
		result = max(operand[i], result);
	return result;
}

vector<unsigned int> fs::minIdx(vector<float> *operand) {
	float m = min((*operand));
	vector<unsigned int> indices = getSupport(operand, m);
	if ((*operand)[0] >= 0.01 && (*operand)[samples - 1] >= 0.01 && parameters.size() == 0)
		return{ 0, samples - 1 };
	else
		return indices;
}

vector<unsigned int> fs::maxIdx(vector<float> *operand) {
	float m = max((*operand));
	vector<unsigned int> indices = getSupport(operand, m);
	if ((*operand)[0] >= 0.01 && (*operand)[samples - 1] >= 0.01 && parameters.size() == 0)
		return {0, samples - 1};
	else
		return indices;
}

vector<float> fs::min(const vector<float> &operand1, const vector<float> &operand2) const {
	vector<float> result(samples);
	if (isCUDA) {
		float *r = &result[0];
		const float *p = &operand1[0];
		const float *q = &operand2[0];
		/*cudaStream_t s;
		streams.push_back(s);*/
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::min (operand 1) in line 3065: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		cudaStatus = cudaMemcpy(dev_op2, q, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::min (operand 2) in line 3071: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		_minimumIntersection <<<grids, blocks>>>(dev_result1, dev_op1, dev_op2);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::min in line 3079:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::min (result) in line 3086: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			result[i] = min(operand1[i], operand2[i]);
	return result;
}

vector<float> fs::max(const vector<float> &operand1, const vector<float> &operand2) const {
	vector<float> result(samples);
	if (isCUDA) {
		float *r = &result[0];
		const float *p = &operand1[0];
		const float *q = &operand2[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::max (operand 1) in line 3129: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		cudaStatus = cudaMemcpy(dev_op2, q, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::max (operand 2) in line 3135: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		_maximumUnion <<<grids, blocks>>>(dev_result1, dev_op1, dev_op2);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::max in line 3143:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::max (result) in line 3150: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			result[i] = max(operand1[i], operand2[i]);
	return result;
}

vector<float> fs::min(const vector<float> &operand1, const float &operand2) const {
	vector<float> result(samples);
	if (isCUDA) {
		float *r = &result[0];
		const float *p = &operand1[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::min (operand 1) in line 3186: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		_minimumIntersection <<<grids, blocks>>>(dev_result1, dev_op1, operand2);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::min in line 3194:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::min (result) in line 3201: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			result[i] = min(operand1[i], operand2);
	return result;
}

vector<float> fs::max(const vector<float> &operand1, const float &operand2) const {
	vector<float> result(samples);
	if (isCUDA) {
		float *r = &result[0];
		const float *p = &operand1[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::max (operand 1) in line 3236: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		_maximumUnion <<<grids, blocks >>>(dev_result1, dev_op1, operand2);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::max in line 3244:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::max (result) in line 3251: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			result[i] = max(operand1[i], operand2);
	return result;
}

vector<float> fs::min(const float &operand1, const vector<float> &operand2) const {
	vector<float> result(samples);
	if (isCUDA) {
		float *r = &result[0];
		const float *q = &operand2[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op2, q, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::min (operand 2) in line 3286: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		_minimumIntersection <<<grids, blocks>>>(dev_result1, operand1, dev_op2);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::min in line 3294:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::min (result) in line 3301: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			result[i] = min(operand1, operand2[i]);
	return result;
}

vector<float> fs::max(const float &operand1, const vector<float> &operand2) const {
	vector<float> result(samples);
	if (isCUDA) {
		float *r = &result[0];
		const float *q = &operand2[0];
		// Transfer data from host to device
		cudaStatus = cudaMemcpy(dev_op2, q, samples * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fs::max (operand 2) in line 3336: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Launch the kernel
		_maximumUnion <<<grids, blocks>>>(dev_result1, operand1, dev_op2);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << "Kernel launch failed in fs::max in line 3344:" << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		// Transfer data back from device to host
		cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to host data transfer failed in fs::max (result) in line 3351: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	else
		for (unsigned int i = 0; i < samples; i++)
			result[i] = max(operand1, operand2[i]);
	return result;
}

vector<fs> fs::normalize(vector<fs> &denormalized) const {
	float m = float(0.0);
	vector<fs> result;
	for (unsigned int i = 0; i < denormalized.size(); i++)
		m = max(m, denormalized[i].normalization);
	for (unsigned int i = 0; i < denormalized.size(); i++) {
		fs normalized(isCUDA, isStream);
		normalized = denormalized[i];
		float d = m / normalized.normalization;
		if (isCUDA) {
			float *r = &normalized.membership[0];
			// Transfer data from host to device
			cudaStatus = cudaMemcpy(dev_op1, r, samples * sizeof(float), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fs::normalize (operand 1) in line : " << cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			// Launch the kernel
			_fsMul <<<grids, blocks>>>(dev_result1, dev_op1, d);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				cerr << "Kernel launch failed in fs::normalize in line 3400:" << cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
			// Transfer data back from device to host
			cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cerr << "Device to host data transfer failed in fs::normalize (result) in line 3407: " << cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
		}
		else
			for (unsigned int j = 0; j < samples; j++)
				normalized.membership[j] = normalized.membership[j] * d;
		result.push_back(normalized);
	}
	return result;
}

// FUZZY SET: DEFUZZIFICATION METHODS
float fs::adaptiveIntegration() const {
	float result = float(0.0);
	return result;
}

float fs::basicDefuzzDistributions() const {
	float result = float(0.0);
	return result;
}

float fs::bisectorOfArea() const {
	vector<float> A;
	for (unsigned int i = 0; i < samples - 1; i++) {
		float B = discourse[i + 1] - discourse[i];
		float h1 = membership[i];
		float h2 = abs(membership[i + 1] - membership[i]);
		A.push_back(B * h1 + (B * h2) / 2);
	}
	float acc = float(0.0);
	for (unsigned int i = 0; i < A.size(); i++)
		acc += A[i];
	float mid = acc / 2;
	acc = float(0.0);
	unsigned int index;
	for (unsigned int i = 0; i < A.size(); i++) {
		acc += A[i];
		if (acc >= mid) {
			index = i;
			break;
		}
	}
	return discourse[index + 1];
}

float fs::constraintDecision() const {
	float result = float(0.0);
	return result;
}

float fs::centerOfArea() const {
	float num = float(0.0), den = float(0.0);
	for (unsigned int i = 0; i < samples - 1; i++) {
		float B = discourse[i + 1] - discourse[i];
		float x = (discourse[i + 1] + discourse[i]) / 2;
		float h1 = membership[i];
		float h2 = abs(membership[i + 1] - membership[i]);
		float area = B * h1 + (B * h2) / 2;
		num += area * x;
		den += area;
	}
	return num / den;
}

float fs::centroid() {
	float result, num = float(0.0), den = float(0.0);
	//vector<float> multRes(samples);
	//if (isCUDA) {
	//	float *r = &multRes[0];
	//	float *p = &discourse[0];
	//	float *q = &membership[0];
	//	// Transfer data from host to device
	//	cudaStatus = cudaMemcpy(dev_op1, p, samples * sizeof(float),
	//		cudaMemcpyHostToDevice);
	//	if (cudaStatus != cudaSuccess) {
	//		cerr << "Host to device data transfer failed in fs::centroid in line 3510: "
	//			<< cudaGetErrorString(cudaStatus) << endl;
	//		cin.get();
	//		exit(1);
	//	}
	//	cudaStatus = cudaMemcpy(dev_op2, q, samples * sizeof(float),
	//		cudaMemcpyHostToDevice);
	//	if (cudaStatus != cudaSuccess) {
	//		cerr << "Host to device data transfer failed in fs::centroid in line 3518: "
	//			<< cudaGetErrorString(cudaStatus) << endl;
	//		cin.get();
	//		exit(1);
	//	}
	//	// Launch the kernel
	//	_fsMul <<<grids, blocks>>>(dev_result1, dev_op1, dev_op2);
	//	cudaStatus = cudaGetLastError();
	//	if (cudaStatus != cudaSuccess) {
	//		cerr << "Kernel launch failed in fs::centroid in line 3527: "
	//			<< cudaGetErrorString(cudaStatus) << endl;
	//		cin.get();
	//		exit(1);
	//	}
	//	// Transfer data back from device to host
	//	cudaStatus = cudaMemcpy(r, dev_result1, samples * sizeof(float),
	//		cudaMemcpyDeviceToHost);
	//	if (cudaStatus != cudaSuccess) {
	//		cerr << "Device to host data transfer failed in fs::centroid in line 3536: "
	//			<< cudaGetErrorString(cudaStatus) << endl;
	//		cin.get();
	//		exit(1);
	//	}
	//	for (unsigned int i = 0; i < samples; i++) {
	//		num += multRes[i];
	//		den += membership[i];
	//	}
	//}
	//else {
	for (unsigned int i = 0; i < samples; i++) {
		num += discourse[i] * membership[i];
		den += membership[i];
	}
	//}
	if (den == float(0.0))
		result = (discourse[0] + discourse[samples - 1]) / 2;
	else
		result = num / den;
	return result;
}

float fs::extendedCenterOfArea() const {
	float result = float(0.0);
	return result;
}

float fs::extendedQuality() const {
	float result = float(0.0);
	return result;
}

float fs::fuzzyClustering() const {
	float result = float(0.0);
	return result;
}

float fs::fuzzyMean() const {
	float result = float(0.0);
	return result;
}

float fs::generalLevelSet() const {
	float result = float(0.0);
	return result;
}

float fs::indexedCenterOfGravity() const {
	float result = float(0.0);
	return result;
}

float fs::influenceValue() const {
	float result = float(0.0);
	return result;
}

float fs::smallestOfMaximum() {
	vector<unsigned int> interval = getSupport(normalization);
	return discourse[interval[0]];
}

float fs::largestOfMaximum() {
	vector<unsigned int> interval = getSupport(normalization);
	return discourse[interval[1]];
}

float fs::meanOfMaxima() {
	vector<unsigned int> interval = getSupport(normalization);
	return (discourse[interval[0]] + discourse[interval[1]]) / 2;
}

float fs::quality() const {
	float result = float(0.0);
	return result;
}

float fs::randomChoiceOfMaximum() const {
	float result = float(0.0);
	return result;
}

float fs::semiLinear() const {
	float result = float(0.0);
	return result;
}

float fs::weightedFuzzyMean() const {
	float result = float(0.0);
	return result;
}

float fs::geometric() const {
	float result = float(0.0);
	return result;
}

#endif
