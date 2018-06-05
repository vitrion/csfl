// The number of samples defines the vector operations length
extern const unsigned int samples(100000);
// To start building CUDA-accelerated fuzzy systems import the library
#include "C:\Users\Arturo\OneDrive\Documentos\Catedras\UTM\Research\FLS\Programs\v1_1\fls.cuh"

int main()
{
	float pi = acos(-1);
	
	// This creates the CUDA information needed for operation
	CUDAinit();
	// These are the crisp input observations
	float e = -pi / 2;
	float c = pi / 6;
	// The sigma value is needed to build non-singleton sets for fuzzification
	float e_sigma = pi / 10;
	float c_sigma = pi / 30;
	// These flags determine the execution mode
	bool isCUDA = true;
	bool isStream = true;

	// Here the fls constructor defines the execution mode
	fls motor(isCUDA, isStream);

	// Set the fuzzy system application name
	motor.setName("Servomotor");
	// Set the inference model
	motor.setInferenceModel("Mamdani");

	// Set the variable ranges and names
	vector<float> range1 = { -pi / 2, pi / 2 };
	vector<float> range2 = { -pi / 6, pi / 6 };
	string name1 = "Error";
	string name2 = "Change";

	// Linguistic variable error has been created
	motor.addFuzzyVar("Input", name1, range1[0], range1[1]);
	// Linguistic variable change has been created
	motor.addFuzzyVar("Input", name2, range2[0], range2[1]);
	// Linguistic variable voltage has been created
	motor.addFuzzyVar("Output", "Voltage", -7.5, 7.5);
	
	//Three linguistic terms have been created for the error
	motor.addFuzzySet("Error", "Negative", "Z", { -pi / 2, 0 }, 1);
	motor.addFuzzySet("Error", "Zero", "Triangular", { -pi / 2, 0, pi / 2 }, 1);
	motor.addFuzzySet("Error", "Positive", "S", { 0, pi / 2 }, 1);
	//Three linguistic terms have been created for the change
	motor.addFuzzySet("Change", "Negative", "Z", { -pi / 6, 0 }, 1);
	motor.addFuzzySet("Change", "Zero", "Triangular", { -pi / 6, 0, pi / 6 }, 1);
	motor.addFuzzySet("Change", "Positive", "S", { 0, pi / 6 }, 1);
	//Three linguistic terms have been created for the voltage
	motor.addFuzzySet("Voltage", "Negative", "Z", { -5, 0 }, 1);
	motor.addFuzzySet("Voltage", "Zero", "Triangular", { -5, 0, 5 }, 1);
	motor.addFuzzySet("Voltage", "Positive", "S", { 0, 5 }, 1);

	// Rule set is comprised of 9 rules
	motor.addFuzzyRule("(Error: Negative) AND (Change: Negative) THEN (Voltage: Negative)");
	motor.addFuzzyRule("(Error: Negative) AND (Change: Zero) THEN (Voltage: Negative)");
	motor.addFuzzyRule("(Error: Negative) AND (Change: Positive) THEN (Voltage: Negative)");
	motor.addFuzzyRule("(Error: Zero) AND (Change: Negative) THEN (Voltage: Negative)");
	motor.addFuzzyRule("(Error: Zero) AND (Change: Zero) THEN (Voltage: Zero)");
	motor.addFuzzyRule("(Error: Zero) AND (Change: Positive) THEN (Voltage: Positive)");
	motor.addFuzzyRule("(Error: Positive) AND (Change: Negative) THEN (Voltage: Positive)");
	motor.addFuzzyRule("(Error: Positive) AND (Change: Zero) THEN (Voltage: Positive)");
	motor.addFuzzyRule("(Error: Positive) AND (Change: Positive) THEN (Voltage: Positive)");

	// Here, the execution plan is lazily evaluated and the fuzzy system is ready for making decisions
	motor.configure();
	
	// The non-singleton fuzzy observations have been created
	fs a(range1[0], range1[1], 1.0, name1 + "_prime", "Gaussian", { e, e_sigma }, true, true);
	fs b(range2[0], range2[1], 1.0, name2 + "_prime", "Gaussian", { c, c_sigma }, true, true);

	// Fuzzify by using the non-singleton fuzzy observations
	motor.fuzzify({ a, b });
	// Perform the inference process according to the execution plan
	motor.infer();
	// Defuzzify the implied set of voltage
	vector<float> result = motor.defuzzify();
	cout << "The inferred voltage generated from inputs " << e << "(rads) and " << c << "(rads/s) is " << result[0] << " volts.";
	// his deletes all the CUDA information used during operation
	CUDAend();

	cin.get();
    return 0;
}