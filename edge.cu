extern const unsigned int samples(8001);
#include "C:\Users\Arturo\OneDrive\Documentos\Catedras\UTM\Research\FLS\Programs\v1_1\fls.cuh"
#include <iomanip>
#include <fstream>

int main()
{
	bool isCUDA;
	bool isStream;
	float sigma = 0.2;
	vector<float> range = { 0, 1 };
	float offset1 = 0.65, offset2 = 0.75;
	float ap1 = 0.3, ap2 = 0.3;
	vector<float> firstDarkParams = { float(offset1 - ap1 / 2), float(offset1 + ap1 / 2) };
	vector<float> firstBrightParams = { float(offset1 - ap1 / 2), float(offset1 + ap1 / 2) };
	vector<float> secondDarkParams = { float(offset2 - ap1 / 2), float(offset2 + ap1 / 2) };
	vector<float> secondBrightParams = { float(offset2 - ap1 / 2), float(offset2 + ap1 / 2) };

	vector<float> odarkParams = { float(0.0), float(0.05) };
	vector<float> obrightParams = { float(9.0), float(9.1) };
	nvtxRangeId_t t14, t15, t16, t17, t18, t19;

	string fileName = "lenagrayfull";
	ifstream inputFile;
	inputFile.open(fileName + ".dat");
	//vector<unsigned int> imSize = { 254, 308 };
	//vector<unsigned int> imSize = { 112, 68 };
	//vector<unsigned int> imSize = { 248, 302 };
	vector<unsigned int> imSize = { 722, 470 };
	//vector<unsigned int> imSize = { 264, 264 };
	//vector<unsigned int> imSize = { 16, 16 };
	vector<vector<float>> first(imSize[0], vector<float>(imSize[1])), second(imSize[0], vector<float>(imSize[1])), fuzzyEdges(imSize[0], vector<float>(imSize[1]));
	for (unsigned int i = 0; i < 2 * imSize[0]; i++)
		for (unsigned int j = 0; j < imSize[1]; j++) {
			if (i <= imSize[0] - 1)
				inputFile >> first[i][j];
			else
				inputFile >> second[i - imSize[0]][j];
		}

	vector<float> pixelInputs(9, 0.0);

	CUDAinit();
	for (unsigned int h = 0; h < 1; h++) {
		fls edge;
		// Starts initialization
		if (h == 0){
			isCUDA = false;
			isStream = false;
			t14 = nvtxRangeStartEx(&events[14]);
		}
		else if (h == 1) {
			isCUDA = true;
			isStream = false;
			t15 = nvtxRangeStartEx(&events[15]);
		}
		else {
			isCUDA = true;
			isStream = true;
			t16 = nvtxRangeStartEx(&events[16]);
		}
		edge.setName("edgeDetector");
		edge.setInferenceModel("Mamdani");
		edge.setHetProc(isCUDA);
		edge.setStreams(isStream);

		for (unsigned int i = 0; i < pixelInputs.size(); i++) {
			ostringstream convert;
			convert << i + 1;
			string str = convert.str();
			edge.addFuzzyVar("Input", "p" + str, range[0], range[1]);
		}
		//edge.addFuzzyVar("Input", "f", range[0], range[1]);
		edge.addFuzzyVar("Input", "s", range[0], range[1]);
		edge.addFuzzyVar("Output", "e", -4.0, 10.0);

		/*edge.addFuzzySet("f", "D", "Z", { firstDarkParams[0], firstDarkParams[1] }, 1);
		edge.addFuzzySet("f", "B", "S", { firstBrightParams[0], firstBrightParams[1] }, 1);
		edge.addFuzzySet("s", "D", "Z", { secondDarkParams[0], secondDarkParams[1] }, 1);
		edge.addFuzzySet("s", "B", "S", { secondBrightParams[0], secondBrightParams[1] }, 1);*/

		// Generate automatically every set in every variable
		for (unsigned int i = 0; i < edge.getInVarCount(); i++) {
			string varName = edge.getInVarName(i);
			if (i != edge.getInVarCount() - 1) {
				edge.addFuzzySet(varName, "D", "Z", { firstDarkParams[0], firstDarkParams[1] }, 1);
				edge.addFuzzySet(varName, "B", "S", { firstBrightParams[0], firstBrightParams[1] }, 1);
			}
			else {
				edge.addFuzzySet(varName, "D", "Z", { secondDarkParams[0], secondDarkParams[1] }, 1);
				edge.addFuzzySet(varName, "B", "S", { secondBrightParams[0], secondBrightParams[1] }, 1);
			}
		}

		edge.addFuzzySet("e", "D", "Z", { odarkParams[0], odarkParams[1] }, 1);
		edge.addFuzzySet("e", "B", "S", { obrightParams[0], obrightParams[1] }, 1);

		/*edge.addFuzzyRule("(p1:B)AND(p5:B)AND(p9:B)THEN(e:B)");
		edge.addFuzzyRule("(p2:B)AND(p5:B)AND(p8:B)THEN(e:B)");
		edge.addFuzzyRule("(p3:B)AND(p5:B)AND(p7:B)THEN(e:B)");
		edge.addFuzzyRule("(p4:B)AND(p5:B)AND(p6:B)THEN(e:B)");
		edge.addFuzzyRule("ELSE(e:D)");*/
		
		// More than 6 the pixels must become the new pixel dark, when pixel 5 is bright
		///*edge.addFuzzyRule("(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p4:B)AND(p5:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p5:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p5:B)AND(p6:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p7:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p7:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:B)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");

		//// More than 6 the pixels must become the new pixel dark, when pixel 5 is dark
		//edge.addFuzzyRule("(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p4:B)AND(p5:D)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p5:D)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p5:D)AND(p6:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p7:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p7:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p5:D)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");

		//// Main rules that preserve the edges
		//edge.addFuzzyRule("(p5:B)AND(p9:B)THEN(e:B)");
		//edge.addFuzzyRule("(p5:B)AND(p8:B)THEN(e:B)");
		//edge.addFuzzyRule("(p5:B)AND(p8:B)AND(p9:B)THEN(e:B)");
		//edge.addFuzzyRule("(p5:B)AND(p7:B)THEN(e:B)");
		//edge.addFuzzyRule("(p5:B)AND(p7:B)AND(p9:B)THEN(e:B)");
		//edge.addFuzzyRule("(p5:B)AND(p7:B)AND(p8:B)THEN(e:B)");
		//edge.addFuzzyRule("(p5:B)AND(p6:B)THEN(e:B)");
		//edge.addFuzzyRule("(p5:B)AND(p6:B)AND(p9:B)THEN(e:B)");
		//edge.addFuzzyRule("(p5:B)AND(p6:B)AND(p8:B)THEN(e:B)");
		//edge.addFuzzyRule("(p5:B)AND(p6:B)AND(p7:B)THEN(e:B)");
		//edge.addFuzzyRule("(p4:B)AND(p5:B)THEN(e:B)");
		//edge.addFuzzyRule("(p4:B)AND(p5:B)AND(p9:B)THEN(e:B)");
		//edge.addFuzzyRule("(p4:B)AND(p5:B)AND(p8:B)THEN(e:B)");
		//edge.addFuzzyRule("(p4:B)AND(p5:B)AND(p7:B)THEN(e:B)");
		//edge.addFuzzyRule("(p4:B)AND(p5:B)AND(p6:B)THEN(e:B)");
		//edge.addFuzzyRule("(p3:B)AND(p5:B)THEN(e:B)");
		//edge.addFuzzyRule("(p3:B)AND(p5:B)AND(p9:B)THEN(e:B)");
		//edge.addFuzzyRule("(p3:B)AND(p5:B)AND(p8:B)THEN(e:B)");
		//edge.addFuzzyRule("(p3:B)AND(p5:B)AND(p7:B)THEN(e:B)");
		//edge.addFuzzyRule("(p3:B)AND(p5:B)AND(p6:B)THEN(e:B)");
		//edge.addFuzzyRule("(p3:B)AND(p4:B)AND(p5:B)THEN(e:B)");
		//edge.addFuzzyRule("(p2:B)AND(p5:B)THEN(e:B)");
		//edge.addFuzzyRule("(p2:B)AND(p5:B)AND(p9:B)THEN(e:B)");
		//edge.addFuzzyRule("(p2:B)AND(p5:B)AND(p8:B)THEN(e:B)");
		//edge.addFuzzyRule("(p2:B)AND(p5:B)AND(p7:B)THEN(e:B)");
		//edge.addFuzzyRule("(p2:B)AND(p5:B)AND(p6:B)THEN(e:B)");
		//edge.addFuzzyRule("(p2:B)AND(p4:B)AND(p5:B)THEN(e:B)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p5:B)THEN(e:B)");
		//edge.addFuzzyRule("(p1:B)AND(p5:B)THEN(e:B)");
		//edge.addFuzzyRule("(p1:B)AND(p5:B)AND(p9:B)THEN(e:B)");
		//edge.addFuzzyRule("(p1:B)AND(p5:B)AND(p8:B)THEN(e:B)");
		//edge.addFuzzyRule("(p1:B)AND(p5:B)AND(p7:B)THEN(e:B)");
		//edge.addFuzzyRule("(p1:B)AND(p5:B)AND(p6:B)THEN(e:B)");
		//edge.addFuzzyRule("(p1:B)AND(p4:B)AND(p5:B)THEN(e:B)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p5:B)THEN(e:B)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p5:B)THEN(e:B)");

		//// Main rules when pixel 5 is dark. In this case the new pixel must be dark also
		//edge.addFuzzyRule("(p5:D)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p5:D)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p5:D)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p5:D)AND(p7:B)THEN(e:D)");
		//edge.addFuzzyRule("(p5:D)AND(p7:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p5:D)AND(p7:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p5:D)AND(p6:B)THEN(e:D)");
		//edge.addFuzzyRule("(p5:D)AND(p6:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p5:D)AND(p6:B)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p5:D)AND(p6:B)AND(p7:B)THEN(e:D)");
		//edge.addFuzzyRule("(p4:B)AND(p5:D)THEN(e:D)");
		//edge.addFuzzyRule("(p4:B)AND(p5:D)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p4:B)AND(p5:D)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p4:B)AND(p5:D)AND(p7:B)THEN(e:D)");
		//edge.addFuzzyRule("(p4:B)AND(p5:D)AND(p6:B)THEN(e:D)");
		//edge.addFuzzyRule("(p3:B)AND(p5:D)THEN(e:D)");
		//edge.addFuzzyRule("(p3:B)AND(p5:D)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p3:B)AND(p5:D)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p3:B)AND(p5:D)AND(p7:B)THEN(e:D)");
		//edge.addFuzzyRule("(p3:B)AND(p5:D)AND(p6:B)THEN(e:D)");
		//edge.addFuzzyRule("(p3:B)AND(p4:B)AND(p5:D)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p5:D)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p5:D)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p5:D)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p5:D)AND(p7:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p5:D)AND(p6:B)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p4:B)AND(p5:D)THEN(e:D)");
		//edge.addFuzzyRule("(p2:B)AND(p3:B)AND(p5:D)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p5:D)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p5:D)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p5:D)AND(p8:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p5:D)AND(p7:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p5:D)AND(p6:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p4:B)AND(p5:D)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p3:B)AND(p5:D)THEN(e:D)");
		//edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p5:D)THEN(e:D)");

		edge.addFuzzyRule("(p1:B)AND(p2:B)AND(p3:B)AND(p4:B)AND(p6:B)AND(p7:B)AND(p8:B)AND(p9:B)THEN(e:D)");
		//edge.addFuzzyRule("(p1:D)AND(p2:D)AND(p3:D)AND(p4:D)AND(p6:D)AND(p7:D)AND(p8:D)AND(p9:D)THEN(e:D)");
		edge.addFuzzyRule("(p5:B)AND(s:D)THEN(e:B)");
		// Otherwise case
		edge.addFuzzyRule("ELSE(e:D)");

		// Generate the inference schedule before execution
		edge.configure();

		// Ends initialization
		if (h == 0){
			nvtxRangeEnd(t14);
			cout << "Sequential fuzzy processing started..." << endl;
			t17 = nvtxRangeStartEx(&events[17]);
		}
		else if (h == 1) {
			nvtxRangeEnd(t15);
			cout << "Heterogeneous fuzzy processing started..." << endl;
			t18 = nvtxRangeStartEx(&events[18]);
		}
		else {
			nvtxRangeEnd(t16);
			cout << "Heterogeneous with Streams fuzzy processing started..." << endl;
			t19 = nvtxRangeStartEx(&events[19]);
		}

		for (unsigned int i = 1; i < imSize[0] - 1; i++) {
			for (unsigned int j = 1; j < imSize[1] - 1; j++) {
				vector<float> pixelInputs = {
					first[i - 1][j - 1], first[i - 1][j + 0], first[i - 1][j + 1],
					first[i + 0][j - 1], first[i][j], first[i + 0][j + 1],
					first[i + 1][j - 1], first[i + 1][j + 0], first[i + 1][j + 1],
					second[i][j] };
				//vector<float> pixelInputs = { first[i][j], second[i][j] };
				vector<fs> x_primes(pixelInputs.size(), fs(isCUDA, isStream));
				for (unsigned int k = 0; k < pixelInputs.size(); k++) {
					ostringstream convert;
					convert << k;
					string str = convert.str();
					fs prime(range[0], range[1], 1.0, "p" + str + "_prime", "Gaussian", { pixelInputs[k], sigma }, isCUDA, isStream);
					x_primes[k] = prime;
				}
				fs u1(range[0], range[1], 1.0, "s_prime", "Gaussian", { pixelInputs[1], sigma }, isCUDA, isStream);
				x_primes[x_primes.size() - 1] = u1;

				// Starts execution
				// Fuzzification process
				edge.fuzzify(x_primes);
				// Execute inference process according to inference schedule
				edge.infer();
				//vector<fs> inferred = edge.getInferredSets();
				// Defuzzification process
				fuzzyEdges[i][j] = edge.defuzzify()[0];//inferred[0].getNormalization();
				cout << fixed << setprecision(2) << "\t\t" << "|" << first[i - 1][j - 1] << ",\t" << first[i - 1][j + 0] << ",\t" << first[i - 1][j + 1] << "|" << endl;
				cout << "(" << i << ", " << j << ") = \t" << "|" << first[i + 0][j - 1] << ",\t" << first[i + 0][j + 0] << ",\t" << first[i + 0][j + 1] << "| = \t" << fuzzyEdges[i][j] << endl;
				cout << "\t\t" << "|" << first[i + 1][j - 1] << ",\t" << first[i + 1][j + 0] << ",\t" << first[i + 1][j + 1] << "|" << endl;
				cout << "---------------------------------------------------------------------------" << endl;
			}
		}
		// Ends execution
		if (h == 0){
			nvtxRangeEnd(t17);
			cout << "Sequential processing finished." << endl;
		}
		else if (h == 1) {
			nvtxRangeEnd(t18);
			cout << "Heterogeneous processing finished." << endl;
		}
		else {
			nvtxRangeEnd(t19);
			cout << "Heterogeneous with Streams processing finished." << endl;
		}

		ofstream outputFile;
		ostringstream convert1, convert2, convert3, convert4, convert5, convert6;
		convert1 << h;
		convert2 << setfill('0') << setw(3) << edge.getRuleCount() + 1;
		convert3 << setfill('0') << setw(3) << offset1 * 100;
		convert4 << setfill('0') << setw(3) << offset2 * 100;
		convert5 << setfill('0') << setw(3) << ap1 * 100;
		convert6 << setfill('0') << setw(3) << ap2 * 100;
		// name: "lenagrayfull" + #rules: "001" + offset1: "0.65" + offset2: "0.75" + firstDerivAp: "0.10" + secondDerivAp: "0.20"
		// lenagrayfull00165751020.dat
		outputFile.open(fileName + convert1.str() + convert2.str() + convert3.str() + convert4.str() + convert5.str() + convert6.str() + ".dat");
		for (unsigned int i = 0; i < imSize[0]; i++) {
			for (unsigned int j = 0; j < imSize[1]; j++) {
				if (j != imSize[1] - 1)
					outputFile << fuzzyEdges[i][j] << "\t";
				else
					outputFile << fuzzyEdges[i][j] << endl;
			}
		}
		outputFile.close();
	}
	CUDAend();
	//cin.get();
	return 0;
}
