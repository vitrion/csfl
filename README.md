# CUDA-Streams Fuzzy Logic
The CUDA-Streams Fuzzy Logic (CSFL) is the easiest way to design Non-Singleton Type-1 Fuzzy Logic Systems, which has the ability of computing large amounts of inputs and rules concurrently, by using one of the most interesting properties of CUDA: the Streams.

The project structure is based on the Microsoft Visual Studio 2015. The CSFL tool has been designed as an Object-Oriented C++11 implementation with CUDA 8.0. However it can also be imported in a NVIDIA Nsight Eclipse project. Also, it has been tested in the following GPUs: GTX650 Ti and Tesla K40C.

****************************************************************************************************************
HOW TO CITE THIS SOFTWARE.
****************************************************************************************************************
This software was developed under the sponsory of the Mexican Council for Science and Technology under project number 1170.
A journal article is derived from this software, which can be cited by the user as follows:

Arturo Téllez-Velázquez and Raúl Cruz-Barbosa, “A CUDA-Streams Inference Machine for Non-Singleton Fuzzy Systems,” Concurrency Computat.: Pract. Exper. (2017), vol. X, number X, 23 pages, 2016. doi:10.1155/2012/698062

****************************************************************************************************************
CONTACT INFORMATION.
****************************************************************************************************************

For further information about installing this software, please send us an email to the following recipients:

Dr. Arturo Téllez-Velázquez
atellezve@conacyt.mx

Dr. Raúl Cruz-Barbosa
rcruz@mixteco.utm.mx

UNIVERSIDAD TECNOLÓGICA DE LA MIXTECA
Carretera a Acatlima km. 2.5, Zip. Code 69000,
Huajuapan de León, Oaxaca, México, 2016.
Phone number: +52 953 532 0399 ext. 200.

****************************************************************************************************************
INSTALLATION GUIDELINES.
****************************************************************************************************************

Guidelines for installing the CUDA-Streams Fuzzy Library (CSFL) v1.1:

1.  Install the Programming IDE of your preference. If you use Windows 8.1, we recommend to install the Visual Studio 2013 Community first. If you use a Linux distribution, you do not have to install any programming IDE because it is included in the toolkit.
2.  Download and install the CUDA Toolkit 7.5 from the NVIDIA site. www.nvidia.com/cuda/download. It is very important that you install Visual Studio 2013 Community before the toolkit, in the case of you are using Windows 8.1. The following documentation can be consulted: http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#axzz4D5QFJ2gU or http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#axzz4D5QFJ2gU. Before installation, be sure of having installed in your system a CUDA-capable device.
3.  Download or clone this project.
6.  Create a new CUDA Runtime Project.
7.  If you are using Visual Studio, configure your project to allow the use of NVTX by following the guidelines: http://http.developer.nvidia.com/ParallelNsight/1.51/UserGuide/HTML/How_To_Use_the_NVIDIA_Tools_Extension_Library_(nvtx).html. If you are using Linux, follow the guidelines: https://stackoverflow.com/questions/35567621/what-is-the-correct-cuda-project-configuration-when-profiling-in-nsight-eclipse.
8.  Write your own code by using the classes fs and fls in order to build fuzzy sets and fuzzy systems, respectively. Please assure that your program addresses (with #include directive) the header file "fls.cuh" in the correct directory where you uncompressed the CSFL-v1_1.zip file (we recomment to include the full path string).
9.  Define the number of samples for processing the application as an external constant, i.e. an extern const unsigned int.
10.  Define the processing type by setting the flag isCUDA: false for sequential and true for heterogeneous by using CUDA processing. Also, if you prefer using CUDA-Streams wyou will need to set isStream to true; please be sure that isCUDA is set to true first. For instance, the code: bool isCUDA = true; will turn on the CUDA Fuzzy acceleration; bool isCUDA = true and bool isStream = true will turn on the CUDA Streams execution mode
11. Please be sure of calling the global funnctions CUDAinit() at the beggining of your code and CUDAend() at the end of your code.
12. Build your application and run.

****************************************************************************************************************
FILE DESCRIPTION IN THE GPCFL
****************************************************************************************************************

| FILE NAME       | DESCRPTION |
| ---------       | ---------- |
| cdev.h          | This file provides the class "cdev" that manages the CUDA device properties |
| fls.cuh         | This is the main header file that the user must include in the applications. Provides the whole methods to manage fuzzy set objects of class "fs", fuzzy variable objects of class "fvar" and fuzzy rule objects of class "frule." Its functionality help the user to build fuzzy sets and systems. |
| flsExcept.cuh   | This file provides the fuzzy system object exceptions of class "fls." |
| flsFriends.cuh  | This file provides the friend functions of classes "fls," "fs," "fvar," and "frule." |
| frule.cuh       | This file provides the fuzzy rule class and its functionality based on STL objects of class "string." |
| fruleExcept.cuh | This file provides the fuzzy rule object exceptions of class "frule." |
| fs.cuh          | This file is the main building block for the class "fls." Provides the fuzzy arithmetic and logic operations for fuzzy sets. Also it provides CUDA acceleration by enabling a flag called "isCUDA." |
| fsExcept.cuh    | This file provides the fuzzy set object exceptions of class "fs." |
| fvar.cuh        | This file provides the fuzzy variable class and its functionality to contain and manage fuzzy set objects of class "fs."                                                                                                               
| fvarExcept.cuh  | This file provides the fuzzy variable object exceptions of class "fvar." |

****************************************************************************************************************
BASIC FLS METHODS TO BUILD CONTROL SYSTEMS
****************************************************************************************************************
|-----------------------------------------|------------------------------------------------------------------------------------------------|
|               FLS METHOD                |                                      DESCRPTION                                                |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
|fls::fls()                               | Creates an initializes a fuzzy logic system.                                                   |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
|fls::seName(string name)                 | Sets the <name> of the fuzzy logic system.                                                     |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
|fls::setInferenceModel(string model)     | Sets the inference <model>: Mamdani.                                                           |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
|fls::setHetProc(bool processing)         | Sets the <processing> type: false for sequential and true for heterogeneous                    |
|                                         | by using CUDA.                                                                                 |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
|fls::setStream(bool processing)          | Sets the <CUDA Streams processing> type: false for sequential and true for heterogeneous       |
|                                         | by using CUDA.                                                                                 |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
|fls::addFuzzyVar(string type,            | Adds a variable of <type> "Input" or "Output." The string called <name> establishes the name   |
|  string name, vector<double> range)     | of the variable and the <range> is the interval of operation of the variable.                  |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
|fls::addFuzzySet(string variable,        | Adds a fuzzy set called <name>, in a specific <variable>. Also, you can specify the <shape>,   |
|  string name, string shape,             | <parameters> and <normalization> values of each set.                                           |
|  vector<double> parameters,             |                                                                                                |
|  double normalization)                  |                                                                                                |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
|fls::addFuzzyRule(string)                | Adds a string-based fuzzy rule in a specific syntax and order. Each rule is included in the    |
|                                         | rule set ordered in the way it is added with this method.                                      |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
|fls::configure()                         | It builds the execution plan based on the current fuzzy system configuration by using the lazy |
|                                         | evaluation of rules. Until here, no fuzzy rule has been executed, but an execution plan is     |
|                                         | ready for later execution. To execute the rule, use the method infer().                        |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
|fls::fuzzify(vector<double> crisp_inputs)| Fuzzify the crisp input values by using the Non-Singleton Fuzzification, according to the      |
|                                         | fuzzy system configuration.                                                                    |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
|fls::infer()                             | Executes the inference machine according to the execution plan. An inferred set is obtained    |
|                                         | for each output variable.                                                                      |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
|double crisp_outputs fls::defuzzify()    | Returns the defuzzified value from the inferred set obtained with the method infer. The length |
|                                         | of the resulting std::vector object that returns the method defuzzify() is defined by the      |
|                                         | number of the output variables.                                                                |
|-----------------------------------------|------------------------------------------------------------------------------------------------|

****************************************************************************************************************
BASIC FS OPERATORS
****************************************************************************************************************
|-----------------|----------------------------------------|
|    OPERATOR     |             DESCRPTION                 |
|-----------------|----------------------------------------|
|        +        | Fuzzy arithmetic sum. Example:         |
|                 | fs A, B, C;                            |
|                 | C = A + B;                             |
|-----------------|----------------------------------------|
|        -        | Fuzzy arithmetic subtraction. Example: |
|                 | fs A, B, C;                            |
|                 | C = A - B;                             |
|-----------------|----------------------------------------|
|        *        | Fuzzy arithmetic product. Example:     |
|                 | fs A, B, C;                            |
|                 | C = A * B;                             |
|-----------------|----------------------------------------|
|        /        | Fuzzy arithmetic division. Example:    |
|                 | fs A, B, C;                            |
|                 | C = A / B;                             |
|-----------------|----------------------------------------|
|        &        | Fuzzy logic intersection. Example:     |
|                 | fs A, B, C;                            |
|                 | C = A & B;                             |
|-----------------|----------------------------------------|
|        |        | Fuzzy logic union. Example:            |
|                 | fs A, B, C;                            |
|                 | C = A | B;                             |
|-----------------|----------------------------------------|
|        !        | Fuzzy logic complement. Example:       |
|                 | fs A, B;                               |
|                 | B = !A;                                |
|-----------------|----------------------------------------|

****************************************************************************************************************
BUILDING YOU OWN APPLICATION.
****************************************************************************************************************

Once you have installed all the required programs for the CSFL v1.1, please write the following code to build a simple Fuzzy System for the DC Servomotor Control. This example is expressed as a pseudo-code in the derived paper:

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
