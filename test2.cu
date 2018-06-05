extern const unsigned int samples(31);
#include "C:\Users\Arturo\OneDrive\Documentos\Catedras\UTM\Research\FLS\Programs\v1_1\fls.cuh"
#include <fstream>

int main()
{
	bool isCUDA;
	bool isStream;
	vector<float> numOfInputs(512, 1.0);
	unsigned int numOfRules = 120;
	float sigma = 0.25;
	unsigned int numOfSetsPerVar = 4;
	nvtxRangeId_t t14, t17, t18, t19;

	// FLS initialization
	CUDAinit();

	cout << "T1FLS configuration started..." << endl;
	fls macro;
	//t14 = nvtxRangeStartEx(&events[14]);	
	macro.setName("macroTest");
	macro.setInferenceModel("Mamdani");	

	vector<float> range = { -1, 1 };
	vector<string> vnames;
	for (unsigned int i = 0; i < numOfInputs.size(); i++) {
		ostringstream convert;
		convert << i;
		string str = convert.str();
		vnames.push_back("x" + str);
		macro.addFuzzyVar("Input", vnames[i], range[0], range[1]);
	}
	macro.addFuzzyVar("Output", "y", -10, 10);

	// Generate automatically every set in every variable
	float dx = (range[1] - range[0]) / float(numOfSetsPerVar - 1);
	vector<float> r(numOfSetsPerVar + 2);
	for (unsigned int i = 0; i < numOfSetsPerVar + 2; i++)
		r[i] = float(i) * dx + range[0] - dx;
	for (unsigned int i = 0; i < macro.getInVarCount(); i++) {
		string varName = macro.getInVarName(i);
		ostringstream convert1;
		convert1 << i;
		string varID = convert1.str();
		for (unsigned int j = 0; j < numOfSetsPerVar; j++) {
			ostringstream convert2;
			convert2 << j;
			string setID = convert2.str();
			if (j == 0)
				macro.addFuzzySet(varName, "A" + varID + setID, "Z", { r[j + 1], r[j + 2] }, 1);
			else if (j == numOfSetsPerVar - 1)
				macro.addFuzzySet(varName, "A" + varID + setID, "S", { r[j], r[j + 1] }, 1);
			else
				macro.addFuzzySet(varName, "A" + varID + setID, "Triangular", { r[j], r[j + 1], r[j + 2] }, 1);
		}
	}
	string varName = macro.getOutVarName(0);
	for (unsigned int j = 0; j < numOfSetsPerVar; j++) {
		ostringstream convert2;
		convert2 << j;
		string setID = convert2.str();
		if (j == 0)
			macro.addFuzzySet(varName, "B" + setID, "Z", { r[j + 1], r[j + 2] }, 1);
		else if (j == numOfSetsPerVar - 1)
			macro.addFuzzySet(varName, "B" + setID, "S", { r[j], r[j + 1] }, 1);
		else
			macro.addFuzzySet(varName, "B" + setID, "Triangular", { r[j], r[j + 1], r[j + 2] }, 1);
	}
	// Generates all the possible rules according to the available premises
	macro.addFuzzyRule(numOfRules);
	//macro.addFuzzyRule();
	// Generate the inference schedule before execution
	macro.configure();

	//nvtxRangeEnd(t14);

	for (unsigned int h = 0; h < 1; h++) {			
		// Starts initialization
		if (h == 0){
			isCUDA = true;// false;
			isStream = true;// false;
		}
		else if (h == 1) {
			isCUDA = true;
			isStream = false;
		}
		else {
			isCUDA = true;
			isStream = true;
		}
		macro.setHetProc(isCUDA);
		macro.setStreams(isStream);

		vector<fs> x_primes(numOfInputs.size());
		for (unsigned int i = 0; i < numOfInputs.size(); i++) {
			ostringstream convert;
			convert << i;
			string str = convert.str();
			fs prime(range[0], range[1], 1.0, "x" + str + "_prime", "Gaussian", { numOfInputs[i], sigma }, isCUDA, isStream);
			x_primes[i] = prime;
		}

		// Ends initialization
		if (h == 0){
			cout << "Heterogeneous with Streams fuzzy processing started..." << endl;
			t19 = nvtxRangeStartEx(&events[19]);
			//cout << "Sequential fuzzy processing started..." << endl;
			//t17 = nvtxRangeStartEx(&events[17]);
		}
		else if (h == 1) {
			cout << "Heterogeneous fuzzy processing started..." << endl;
			t18 = nvtxRangeStartEx(&events[18]);
		}
		else {
			cout << "Heterogeneous with Streams fuzzy processing started..." << endl;
			t19 = nvtxRangeStartEx(&events[19]);
		}
		// Starts execution

		// Fuzzification process
		macro.fuzzify(x_primes);
		// Save the rules back to a file to verify nested operations
		macro.saveResultingRulesAsStr("test2.rs");
		// Execute inference process according to inference schedule
		macro.infer();		
		// Defuzzification process
		vector<float> result = macro.defuzzify();

		// Ends execution
		if (h == 0){
			nvtxRangeEnd(t19);
			cout << "Heterogeneous with Streams fuzzy processing finished." << endl;
			//nvtxRangeEnd(t17);
			//cout << "Sequential fuzzy processing finished." << endl;
		}
		else if (h == 1) {
			nvtxRangeEnd(t18);
			cout << "Heterogeneous fuzzy processing finished." << endl;
		}
		else {
			nvtxRangeEnd(t19);
			cout << "Heterogeneous with Streams fuzzy processing finished." << endl;
		}
		cout << result[0] << endl;
	}
	CUDAend();
	// cin.get();
    return 0;
}