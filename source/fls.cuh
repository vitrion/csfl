#ifndef FLS_CUH
#define FLS_CUH

// ********************************************************************************************
// ********************************* LOADING HEADER FILES *************************************
// ********************************************************************************************

#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include "nvToolsExt.h"
vector<nvtxEventAttributes_t> events;
vector<cudaStream_t> streams;
#include "fs.cuh"
#include "fvar.cuh"
#include "frule.cuh"
#include "flsExcept.cuh"
#include "flsFriends.cuh"
#include <fstream>

struct roh{
	unsigned int stage;
	string operation;
	fs *host_operand1;
	fs *host_operand2;
	fs *host_result;
};
vector<roh> execPlan;

// ********************************************************************************************
// ******************************** FUZZY SYSTEM CLASS DEFINITION ********************************
// ********************************************************************************************


class fls
{
public:
	// FUZZY SYSTEM: CONSTRUCTORS

	// Constructor 0: Default Fuzzy System Constructor
	fls(); // OK
	// Constructor 1: Fuzzy System Constructor that specifies its name
	fls(const string &);
	// Constructor 2: Fuzzy System Constructor that specifies its name and processing type
	fls(const string &, const bool &);
	fls(const bool &, const bool &);
	// Constructor 3: Fuzzy System Constructor that specifies its name and inference model
	fls(const string &, const string &);
	// Constructor 4: Fuzzy System Constructor that specifies its name, inference model and processing type
	fls(const string &, const string &, const bool &, const bool &);

	// FUZZY SYSTEM: MANAGEMENT FUNCTIONS

	// Adds a new variable into the current system
	void addFuzzyVar(const string &, const string &, const float &, const float &); // OK
	// Adds an existing variable into the current system
	void addFuzzyVar(fvar &); // OK
	// Deletes a variable from the current system using the fuzzy variable name
	void delFuzzyVar(const string &); // OK
	// Deletes several variables from the current system using several fuzzy variable names
	void delFuzzyVar(const vector<string> &); // OK

	// Adds a new set into an existing variable into the current system
	void addFuzzySet(const string &, const string &, const string &, const vector<float> &, const float &); // OK
	// Adds an existing set into an existing variable into the current system
	void addFuzzySet(const string &, fs &); // OK
	// Deletes a set from the current variable into the system using the fuzzy variable and set names
	void delFuzzySet(const string &, const string &); // OK
	// Deletes several sets from the current variable into the system using several fuzzy set names and its corresponding variable name
	void delFuzzySet(const string &, const vector<string> &); // OK

	// Adds a new rule in the system rule set
	void addFuzzyRule(const string &); // OK
	// Generates all the possible rules according to the available premises
	void addFuzzyRule(); // OK
	// Generates a specific number of rules
	void addFuzzyRule(const unsigned int &);
	// Rule generator
	bool ruleGen(const unsigned int &, const string &);
	// Rule generator with specific number of rules to generate
	bool ruleGen(const unsigned int &, const string &, const unsigned int &);
	// Adds an existing rule in the system rule set
	void addFuzzyRule(frule &); // OK
	// Deletes an existing rule in the system rule set addressing it by its rule number
	void delFuzzyRule(const unsigned int &); // OK
	// Deletes an several existing rules in the system rule set addressing it by its rule number
	void delFuzzyRule(const vector<unsigned int> &); // OK

	// Fuzzify all the input variables
	void fuzzify(vector<fs>);
	// Performs the inference step according to the rule set
	void infer();
	// Perform the inference plan before inference execution
	void configure();
	// Defuzify the inferred set according to the defuzzification method
	vector<float> defuzzify() const;

	// Assigns a fuzzy system to another
	const fls &operator=(const fls &);

	// Modifies the name of the fuzzy system
	void setName(const string &);
	// Modifies the processing type of the fuzzy system
	void setHetProc(const bool &);
	// Modifies the fuzzification type of the fuzzy system
	void setStreams(const bool &);
	// Modifies the inference model used for all the system
	void setInferenceModel(const string &);
	// Modifies the conjunction operation used for all the system
	void setConjOp(const string &);
	// Modifies the disjunction operation used for all the system
	void setDisjOp(const string &);
	// Modifies the aggregation method used for all the system
	void setAggregMethod(const string &);
	// Modifies the defuzzification method used for all the system
	void setDefuzzMethod(const string &);

	// Modifies an existing variable using its name, property and its new string value
	// This is very useful for methods: fvar::setType, fvar::setName, fvar::setConjOp,
	// fvar::setDisjOp, fvar::setAggregMethod, and fvar::setDefuzzMethod
	void setFuzzyVar(const string &, const string &, const string &);
	// Modifies an existing variable using its name, property and its new interval value
	// This is very useful for methods: fvar::setRange
	void setFuzzyVar(const string &, const string &, const vector<float> &);
	// Replaces an existing variable
	void setFuzzyVar(const string &, const fvar &);

	// Modifies an existing set in an existing variable using its name, property and its new string value
	// This is very useful for methods: fs::setName, fs::setShape
	void setFuzzySet(const string &, const string &, const string &, const string &);
	// Modifies an existing set in an existing variable using its name, property and its new numerical value
	// This is very useful for methods: fs::setNorm
	void setFuzzySet(const string &, const string &, const string &, const float &);
	// Modifies an existing set in an existing variable using its name, property and its new interval value
	// This is very useful for methods: fs::setRange and fs::setParams
	void setFuzzySet(const string &, const string &, const string &, const vector<float> &);
	// Replaces an existing set in an existing variable
	void setFuzzySet(const string &, const string &, const fvar &);

	// Modifies an existing rule in the system rule set addressing it by its rule number
	void setFuzzyRule(const unsigned int &, const string &);

	// Set a stage number to each binary operation in the Inference stage
	unsigned int setStage(const string &, const unsigned int &, const unsigned int &, vector<string> *) const;

	// Delivers the type of variable: input or output
	string getInferenceModel() const;
	// Delivers the variable name
	string getName() const;

	// Delivers an existing variable in the system
	fvar getFuzzyVar(const string &) const;
	unsigned int getInVarCount() const;
	unsigned int getOutVarCount() const;
	unsigned int getRuleCount() const;
	string getInVarName(const unsigned int &) const;
	string getOutVarName(const unsigned int &) const;
	// Delivers an exixting set in an existing variable
	fs getFuzzySet(const string &, const string &) const;

	// Delivers the fuzzy conjunction operations used for each variable set
	string getConjOp() const;
	// Delivers the fuzzy disjunction operations used for each variable set
	string getDisjOp() const;
	// Delivers the aggregation method used for each variable set
	string getAggregMethod() const;
	// Delivers the defuzzification method used for each variable set
	string getDefuzzMethod() const;

	// Modifies the fuzzification type of the fuzzy system
	bool getFuzzificationType() const;
	// Delivers the entire rule set
	vector<string> getRuleSet() const;
	// Delivers the current firing strengths
	vector<fs> getFiringStrengths() const;
	// Delivers the current promoted strengths
	vector<float> getPromotedStrengths() const;
	// Delivers the inferred sets for all the outputs
	vector<fs> getInferredSets() const;
	// Delivers the input variables
	vector<fvar> getInputVariables() const;
	// Delivers the output variables
	vector<fvar> getOutputVariables() const;
	// Delivers the control surface for two input variables versus one output variable for non-singleton fuzzification
	float **getSurface(const string &, const string &, const string &, const vector<float> &);
	// Delivers the control surface for two input variables versus one output variable for singleton fuzzification
	float **getSurface(const string &, const string &, const string &);
	// This function delivers a set of token divided by the separator strings
	friend vector<string> token(const string &, const vector<string> &, const bool &);

	vector<fs> getImpliedSets() const;

	void saveResultingRulesAsStr(const string &);
private:
	// Fuzzy system name
	string name;
	// Input variable number
	unsigned int inVarCount;
	// Output variable number
	unsigned int outVarCount;
	// Number of rules
	unsigned int ruleCount;
	// Processing type. TRUE means CUDA processing; FALSE means PC processing
	bool isCUDA;
	// Concurrency enables. TRUE means CUDA will use Streams to process fuzzification and inference stages
	bool isStream;
	// Fuzzification type. TRUE means that Singleton Fuzzification is going to be used; FALSE means Non-Singleton Fuzzification.
	// bool isSingleton;
	// Inference Models: Mamdani or Takagi-Sugeno-Kang (TSK)
	string inferenceModel;
	// Maximum number of rules in the system
	size_t maxRuleCount;
	// Maximum number of premises
	unsigned int premSize;
	// Crisp input values
	vector<float> crisp;
	// Auxiliary Rule String
	vector<string> ruleStr;

	// Conjunction operator
	string conjOperator;
	// Disjunction operator
	string disjOperator;
	// Aggregation method
	string aggregMethod;
	// Defuzzification method
	string defuzzMethod;

	// Input fuzzy variables
	vector<fvar> input_variables;
	// Output fuzzy variables
	vector<fvar> output_variables;
	// Input variables backup
	vector<fvar> input_backup;

	// Output fuzzy trimmers
	vector<fvar> outTrimSets;

	// Fuzzy system rule set
	vector<frule> ruleSet;
	// Consequence string queue
	vector<string> strConsequences;
	// Premises
	vector<fs> premises;
	// String
	vector<string> strPremises;
	// Inferred degree queue
	vector<fs> consequences;
	// Fuzzy firing strenghts
	vector<fs> firingStrengths;
	// Firing strength pointers
	vector<fs *> firePtrs;
	// Fuzzy inferred sets for all the output variables
	vector<fs> inferredSets;
	// Number of fuzzy sets per variable
	vector<unsigned int> numOfSets;
	// Fuzzy control surface
	float **fuzzySurf;
	// Related Set List
	vector<vector<fs *>> relatedSetList;
	// Result of inference operations
	vector<vector<fs>> resOfOps;
	// Stages
	vector<vector<roh>> stages;
	// Stages required for inference, previously scheduled
	vector<vector<roh>> inferStages;

	//// Delivers a pointer to an existing variable in the system
	//fvar *getVarPtr(const string &);
	//// Delivers an index to an existing variable and a boolean to determine if defined in input/output variables
	//void getVarIndex(bool *, unsigned int *, const string &);
	// Delivers the maximum rule count for the current fuzzy system configuration
	size_t getMaxRuleCount() const;
	// Updates the maximum fuzzy rule number according to the variables and sets
	void setMaxRuleNumber();
	// Verifies the rule syntax
	bool isRuleSyntaxOK(const string &);
	// Schedule the fuzzification process
	void fuzzSchedule(vector<fs> *);
	// Performs the fuzzification process with CUDA Streams
	void fuzzExecution();
	// Schedule the implication process with CUDA Streams
	void inferSchedule();
	// Perform the implication process with CUDA Streams
	void inferExecution();
	// Performs the inference aggregation of rules with CUDA
	void aggregation();

	// Performs asynchronous copy from Host to Device with Streams in Inference
	void uploadOperandsToDevice2Infer(const unsigned int &, const int &);
	// Performs asynchronous copy from Device to Host with Streams in Inference
	void downloadResultFromDevice2Infer(const unsigned int &, const int &);	
	// Performs T-Norm for Intersection with Streams in Inference
	void executeTNormKernel(const unsigned int &, const int &, const string &);
	// Performs S-Norm for Union with Streams in Inference
	void executeSNormKernel(const unsigned int &, const int &, const string &);
	// Performs Complement with Streams in Inference
	void executeComplementKernel(const unsigned int &, const unsigned &);

	// Performs asynchronous copy from Host to Device with Streams in Fuzzification
	void uploadOperandsToDevice2Fuzz(const int &);
	// Performs asynchronous copy from Device to Host with Streams in Fuzzification
	void downloadResultFromDevice2Fuzz(const int &);
	// Performs Minimum Intersection with Streams in Non-Singleton Fuzzification
	void executeMinimumKernel(const int &);

	// FUZZY SYSTEM: ERROR CHECKING FUNCTIONS

	// Verify the rule count fault
	void check4RuleCount();
	// Verify when deleting a rule in an empty rule set
	void check4MTRuleSet();
	// 
	void check4ExistingVarAndSet(const string &);
	// 
	fvar *getVarPtr(const string &);
	// 
	fs *getFSPtr(const string &);

	void initializeCounters4Profiling();
};

// ********************************************************************************************
// ********************************  FUZZY SYSTEM CLASS METHODS **********************************
// ********************************************************************************************

// FUZZY SYSTEM: CONSTRUCTORS

// Constructor 0
fls::fls() {
	initializeCounters4Profiling();
	name = "Untitled";
	inVarCount = 0;
	outVarCount = 0;
	ruleCount = 0;
	inferenceModel = "Mamdani";
	conjOperator = "Minimum";
	disjOperator = "Maximum";
	aggregMethod = "Maximum";
	defuzzMethod = "Centroid";
	input_variables = {};
	output_variables = {};
	maxRuleCount = 0;
	ruleSet = {};
	premises = {};
	strPremises = {};
	consequences = {};
	strConsequences = {};
	firingStrengths = {};
	outTrimSets = {};
	inferredSets = {};
	isCUDA = true;
	isStream = false;
}
// Constructor 1:
fls::fls(const string &proposedName) {
	initializeCounters4Profiling();
	name = proposedName;
	inVarCount = 0;
	outVarCount = 0;
	ruleCount = 0;
	inferenceModel = "Mamdani";
	conjOperator = "Minimum";
	disjOperator = "Maximum";
	aggregMethod = "Maximum";
	defuzzMethod = "Centroid";
	input_variables = {};
	output_variables = {};
	maxRuleCount = 0;
	ruleSet = {};
	premises = {};
	strPremises = {};
	consequences = {};
	strConsequences = {};
	firingStrengths = {};
	outTrimSets = {};
	inferredSets = {};
	isCUDA = true;
	isStream = false;
}
// Constructor 2:
fls::fls(const string &proposedName, const bool &procType) {
	initializeCounters4Profiling();
	name = proposedName;
	inVarCount = 0;
	outVarCount = 0;
	ruleCount = 0;
	inferenceModel = "Mamdani";
	conjOperator = "Minimum";
	disjOperator = "Maximum";
	aggregMethod = "Maximum";
	defuzzMethod = "Centroid";
	input_variables = {};
	output_variables = {};
	maxRuleCount = 0;
	ruleSet = {};
	premises = {};
	strPremises = {};
	consequences = {};
	strConsequences = {};
	firingStrengths = {};
	outTrimSets = {};
	inferredSets = {};
	isCUDA = procType;
	isStream = false;
}
fls::fls(const bool &procType, const bool &streamType) {
	initializeCounters4Profiling();
	name = "Untitled";
	inVarCount = 0;
	outVarCount = 0;
	ruleCount = 0;
	inferenceModel = "Mamdani";
	conjOperator = "Minimum";
	disjOperator = "Maximum";
	aggregMethod = "Maximum";
	defuzzMethod = "Centroid";
	input_variables = {};
	output_variables = {};
	maxRuleCount = 0;
	ruleSet = {};
	premises = {};
	strPremises = {};
	consequences = {};
	strConsequences = {};
	firingStrengths = {};
	outTrimSets = {};
	inferredSets = {};
	isCUDA = procType;
	isStream = streamType;
}
// Constructor 3:
fls::fls(const string &proposedName, const string &proposedModel) {
	initializeCounters4Profiling();
	name = proposedName;
	inVarCount = 0;
	outVarCount = 0;
	ruleCount = 0;
	inferenceModel = proposedModel;
	conjOperator = "Minimum";
	disjOperator = "Maximum";
	aggregMethod = "Maximum";
	defuzzMethod = "Centroid";
	input_variables = {};
	output_variables = {};
	maxRuleCount = 0;
	ruleSet = {};
	premises = {};
	strPremises = {};
	consequences = {};
	strConsequences = {};
	firingStrengths = {};
	outTrimSets = {};
	inferredSets = {};
	isCUDA = true;
	isStream = false;
}
// Constructor 4:
fls::fls(const string &proposedName, const string &proposedModel, const bool &procType, const bool &strm) {
	initializeCounters4Profiling();
	name = proposedName;
	inVarCount = 0;
	outVarCount = 0;
	ruleCount = 0;
	inferenceModel = proposedModel;
	conjOperator = "Minimum";
	disjOperator = "Maximum";
	aggregMethod = "Maximum";
	defuzzMethod = "Centroid";
	input_variables = {};
	output_variables = {};
	maxRuleCount = 0;
	ruleSet = {};
	premises = {};
	strPremises = {};
	consequences = {};
	strConsequences = {};
	firingStrengths = {};
	outTrimSets = {};
	inferredSets = {};
	isCUDA = procType;
	isStream = strm;
}

void fls::initializeCounters4Profiling() {
	nvtxEventAttributes_t e = {
		NVTX_VERSION,
		NVTX_EVENT_ATTRIB_STRUCT_SIZE,
		0,
		NVTX_COLOR_ARGB,
		0xFF000000,
		NVTX_PAYLOAD_TYPE_UNSIGNED_INT64,
		1,
		1,
		NVTX_MESSAGE_TYPE_ASCII,
		"Unspecified"
	};
	for (unsigned int i = 0; i < 30; i++)
		events.push_back(e);

	events[0].color = 0xFFFF0000;
	events[0].message.ascii = "Fuzzification";

	events[1].color = 0xFFFF7700;
	events[1].message.ascii = "Fuzzification Scheduling";

	events[2].color = 0xFFFF7777;
	events[2].message.ascii = "Fuzzification Execution";	

	events[10].color = 0xFF00FF00;
	events[10].message.ascii = "Inference";

	events[11].color = 0xFF777F00;
	events[11].message.ascii = "Configuration";

	events[12].color = 0xFF774477;
	events[12].message.ascii = "Implication";

	events[13].color = 0xFF7700AA;
	events[13].message.ascii = "Aggregation";

	events[14].message.ascii = "T1FLS initialization";

	events[17].message.ascii = "Sequential T1FLS execution";
	events[18].message.ascii = "Heterogeneous T1FLS execution";
	events[19].message.ascii = "Heterogeneous with Streams T1FLS execution";

	events[20].color = 0xFF0000FF;
	events[20].message.ascii = "Defuzzification";
}

// FUZZY SYSTEM: MANAGEMENT FUNCTIONS

void fls::addFuzzyVar(const string &proposedType, const string &proposedName, const float &lower, const float &upper) {
	if (proposedType.compare("Input") == 0) {
		fvar newVar(inVarCount, proposedType, proposedName, lower, upper, isCUDA, isStream);
		input_variables.push_back(newVar);
		inVarCount++;
	}
	else if (proposedType.compare("Output") == 0) {
		fvar newVar(outVarCount, proposedType, proposedName, lower, upper, isCUDA, isStream);
		output_variables.push_back(newVar);
		outVarCount++;
	}
}

void fls::addFuzzyVar(fvar &newVar) {
	newVar.setHetProc(isCUDA);
	if (newVar.getType().compare("Input") == 0) {
		newVar.setVarID(inVarCount);
		input_variables.push_back(newVar);
		inVarCount++;
	}
	else if (newVar.getType().compare("Output") == 0) {
		newVar.setVarID(outVarCount);
		output_variables.push_back(newVar);
		outVarCount++;
	}
}

void fls::delFuzzyVar(const string &name2search) {
	fvar *varPtr = getVarPtr(name2search);
	if (varPtr->getType().compare("Input") == 0){
		input_variables.erase(input_variables.begin() + varPtr->getVarID());
		inVarCount--;
	}
	else {
		output_variables.erase(output_variables.begin() + varPtr->getVarID());
		outVarCount--;
	}
}

void fls::delFuzzyVar(const vector<string> &names2search) {
	for (unsigned int i = 0; i < names2search.size(); i++)
		delFuzzyVar(names2search[i]);
}

void fls::addFuzzySet(const string &varName, const string &setName, const string &setShape, const vector<float> &setParams, const float &setNorm) {
	fvar *varPtr = getVarPtr(varName);
	vector<float> r;
	r = varPtr->getRange();
	fs newSet(r[0], r[1], setNorm, varName + ":" + setName, setShape, setParams, isCUDA, isStream);
	varPtr->addFuzzySet(newSet);
	maxRuleCount = getMaxRuleCount();
}

void fls::addFuzzySet(const string &varName, fs &newSet) {
	fvar *varPtr = getVarPtr(varName);
	vector<float> r;
	r = varPtr->getRange();
	newSet.isCUDA = isCUDA;
	newSet.setRange(r[0], r[1]);
	newSet.setName(varName + ":" + newSet.getName());
	varPtr->addFuzzySet(newSet);
	maxRuleCount = getMaxRuleCount();
}

void fls::delFuzzySet(const string &varName, const string &setName) {
	fvar *varPtr = getVarPtr(varName);
	varPtr->delFuzzySet(setName);
	maxRuleCount = getMaxRuleCount();
}

void fls::delFuzzySet(const string &varName, const vector<string> &setNames) {
	for (unsigned int i = 0; i < setNames.size(); i++) {
		delFuzzySet(varName, setNames[i]);
	}
	maxRuleCount = getMaxRuleCount();
}

void fls::addFuzzyRule(const string &ruleString) {
	if (ruleCount != ruleSet.size())
		throw elseStatementFault();
	frule r(ruleCount, ruleString);
	ruleSet.push_back(r);
	size_t index = ruleString.find("ELSE");
	if (index > ruleString.size())
		ruleCount++;
	try {
		check4RuleCount();
	}
	catch (ruleCountFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}

void fls::addFuzzyRule() {
	ruleStr = {};
	bool finished = ruleGen(0, "");
	if (finished) {
		for (unsigned int i = 0; i < outVarCount; i++) {
			unsigned int dr = floor(float(ruleStr.size()) / float(output_variables[i].getSetCount()));
			unsigned int l = 0;
			for (unsigned int k = 0; k < ruleStr.size(); k++) {
				if (l < output_variables[i].getSetCount()) {
					ruleStr[k] += " THEN (" + output_variables[i].sets[l].getName() + ")";
					addFuzzyRule(ruleStr[k]);
					if ((k + 1) % dr == 0 && k != 0)
						l++;
				}
			}
		}			
	}
}

void fls::addFuzzyRule(const unsigned int &numOfRules) {
	ruleStr = {};
	bool finished = ruleGen(0, "", numOfRules);
	if (finished) {
		for (unsigned int i = 0; i < outVarCount; i++) {
			unsigned int dr = floor(float(ruleStr.size()) / float(output_variables[i].getSetCount()));
			unsigned int l = 0;
			for (unsigned int k = 0; k < ruleStr.size(); k++) {
				if (l < output_variables[i].getSetCount()) {
					ruleStr[k] += " THEN (" + output_variables[i].sets[l].getName() + ")";
					addFuzzyRule(ruleStr[k]);
					if ((k + 1) % dr == 0 && k != 0)
						l++;
				}
			}
		}
	}
}

bool fls::ruleGen(const unsigned int &index, const string &str) {
	bool finished;
	if (index < inVarCount) {
		for (unsigned int i = 0; i < input_variables[index].getSetCount(); i++) {			
			if (str.size() == 0) {
				finished = ruleGen(index + 1, "(" + input_variables[index].sets[i].getName() + ")");
				//
			}
			else {
				finished = ruleGen(index + 1, str + " AND (" + input_variables[index].sets[i].getName() + ")");
			}
			if (!finished)
				ruleStr.push_back(str + " AND (" + input_variables[index].sets[i].getName() + ")");
		}
		return true;
	}
	else
		return false;
}

bool fls::ruleGen(const unsigned int &index, const string &str, const unsigned int &numOfRules) {
	bool finished;
	if (index < inVarCount) {
		for (unsigned int i = 0; i < input_variables[index].getSetCount(); i++) {
			if (str.size() == 0)
				finished = ruleGen(index + 1, "(" + input_variables[index].sets[i].getName() + ")", numOfRules);
			else
				finished = ruleGen(index + 1, str + " AND (" + input_variables[index].sets[i].getName() + ")", numOfRules);

			if (!finished)
				ruleStr.push_back(str + " AND (" + input_variables[index].sets[i].getName() + ")");			

			if (ruleStr.size() >= numOfRules)
				return true;
		}
		return true;
	}
	else
		return false;
}

void fls::check4RuleCount() {
	/*if (ruleCount > maxRuleCount)
		throw ruleCountFault();*/
}

void fls::addFuzzyRule(frule &inRule) {
	inRule.setRuleID(ruleCount);
	ruleSet.push_back(inRule);
	ruleCount++;
	try {
		check4RuleCount();
	}
	catch (ruleCountFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}

void fls::delFuzzyRule(const unsigned int &id) {
	try {
		check4MTRuleSet();
		if (id <= ruleCount) {
			ruleSet.erase(ruleSet.begin() + id);
			for (unsigned int i = id; i < ruleSet.size(); i++) {
				unsigned int a = ruleSet[i].getRuleID() - 1;
				ruleSet[i].setRuleID(a);
			}
			ruleCount--;
		}
		else {
			throw unexistRuleFault();
		}
	}
	catch (MTDelRuleFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	catch (unexistRuleFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}

void fls::check4MTRuleSet() {
	if (ruleCount == 0)
		throw MTDelRuleFault();
}

void fls::delFuzzyRule(const vector<unsigned int> &ids) {
	for (unsigned int i = 0; i < ids.size(); i++)
		delFuzzyRule(ids[i]);
}

void fls::check4ExistingVarAndSet(const string &t) {
	fs *f = getFSPtr(t);
}

fvar *fls::getVarPtr(const string &varName) {
	bool found = false;
	fvar *varPtr = NULL;
	for (unsigned int i = 0; i < input_variables.size(); i++)
		if (varName.compare(input_variables[i].getName()) == 0) {
			varPtr = &input_variables[i];
			found = true;
			break;
		}
	if (!found){
		for (unsigned int i = 0; i < output_variables.size(); i++)
			if (varName.compare(output_variables[i].getName()) == 0) {
				varPtr = &output_variables[i];
				found = true;
				break;
			}
	}
	if (!found)
		throw unexistingVarNameFault();
	return varPtr;
}

fs *fls::getFSPtr(const string &fsName) {
	vector<string> t = token(fsName, { ":" }, true);
	fvar *varPtr = getVarPtr(t[0]);
	return varPtr->getFSPtr(fsName);
}

void fls::fuzzify(vector<fs> in_crisps) {
	nvtxRangeId_t t000 = nvtxRangeStartEx(&events[0]);

	// Assign the initial crisp input values
	vector<float> c(inVarCount);
	crisp = c;

	// ***************************
	// BEGIN FUZZIFICATION PROCESS
	// ***************************
	// Get the total number of premise sets
	premSize = 0;
	// Initialize the Premise Sets array
	for (unsigned int i = 0; i < inVarCount; i++) {
		crisp[i] = in_crisps[i].getDiscourse()[in_crisps[i].idxOfMax];
		premSize += input_variables[i].getSetCount();
	}
	if (isStream) {		
		if (premSize > ruleCount)
			streams.resize(premSize);
		else
			streams.resize(ruleCount);
	}

	// Backup the imput variables before fuzzification execution
	input_backup = input_variables;

	//nvtxRangeId_t t001 = nvtxRangeStartEx(&events[1]);
	// Schedule the Execution Plan for Fuzzification
	fuzzSchedule(&in_crisps);
	//nvtxRangeEnd(t001);	

	//nvtxRangeId_t t002 = nvtxRangeStartEx(&events[2]);
	// Execute the fuzzification process based on the Execution Plan
	fuzzExecution();
	//nvtxRangeEnd(t002);
	// ***************************

	// ***********************************************************
	// INFERRED SETS AND TRIMMER SETS INITIALIZATION FOR INFERENCE
	// ***********************************************************
	// This information is necessary for Inference Machine operation
	// The Output Trimmer Variables provide the Trimmer Sets for each variable.
	outTrimSets = output_variables;
	// Begin trimmer set initialization
	for (unsigned int i = 0; i < outVarCount; i++) {
		// Get all the consequence sets of the i-th output variable
		vector<string> names = output_variables[i].getFuzzySetNames();
		// Get the i-th variable range
		vector<float> r = output_variables[i].getRange();
		// Create an Empty Inferred Set with the i-th output variable range
		fs empty(r[0], r[1], 0, "MT", "Unknown", {}, isCUDA, isStream);
		// Output Trimmer Sets initialization
		for (unsigned int j = 0; j < names.size(); j++) {
			// Update the Set Name
			empty.setName(names[j]);
			// Replace the current set of the Output Trimmer Sets array with an Empty set
			outTrimSets[i].replaceFuzzySet(j, empty);
		}
		// Inferred Sets initialization with an Empty Set per output variable
		inferredSets.push_back(empty);
	}
	// ***********************************************************

	// Streams are erased
	// streams.clear();
	// The Fuzzification execution plan is erased
	execPlan.clear();
	nvtxRangeEnd(t000);
}

void fls::infer() {
	//nvtxRangeId_t t10 = nvtxRangeStartEx(&events[10]);	
	inferExecution();
	aggregation();
	//nvtxRangeEnd(t10);
}

// ***********************************************************
// FUZZIFICATION SCHEDULING 
// ***********************************************************
void fls::fuzzSchedule(vector<fs> *inCrisp) {
	execPlan.clear();
	for (unsigned int i = 0; i < inVarCount; i++) {
		fvar *currentVar = &input_variables[i];
		// Shifts the input set to center it with respect its range center
		(*inCrisp)[i].setOffset(-crisp[i]);
		for (unsigned int j = 0; j < currentVar->getSetCount(); j++) {
			fs *currentSet = &currentVar->sets[j];
			// Shifts the premise sets to center it with respect its range center
			currentSet->setOffset(-crisp[i]);
			// Add the operands references to the Execution Plan for Non-Singleton Fuzzification
			roh u{ 0, "tnorm:Minimum", &(*inCrisp)[i], currentSet, currentSet };
			execPlan.push_back(u);
			// If the Fuzzy System isStream is enabled, then add a CUDA Stream
			// Streams are important for concurrent execution of Fuzzification
		}
	}
}

// ***********************************************************
// FUZZIFICATION EXECUTION 
// ***********************************************************
void fls::fuzzExecution() {
	// If Fuzzy System isStream is enabled, then the concurrent execution of Fuzzification is performed
	if (isStream) {
		// Enable all the CUDA Streams previously created in fls::fuzzSchedule() for concurrent execution
		for (unsigned int i = 0; i < streams.size(); i++) {
			cudaStatus = cudaStreamCreate(&streams[i]);
			if (cudaStatus != cudaSuccess) {
				cerr << "CUDA Stream create failed in fls::fuzzExecution in line 820: " << cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
		}
		
		// Overlap every CUDA execution stage according to the Execution Plan for Non-Singleton Fuzzification
		for (int i = 0; i < execPlan.size() + 2; i++) {
			// If device has Dual Copy Engine, then Pipeline is 3
			// Otherwise, Pipeline is 2
			//
			// U0 K0 D0
			//    U1 K1 D1
			//       U2 K2 D2
			//          U3 K3 D3
			//             U4 K4 D4
			//                U5 K5 D5
			// where
			// U: is uploadOperandsToDevice(i)
			// K: is executeMinimumKernel(i - 1)
			// D: is downloadResultFromDevice(i - 2)
			// Every Stream must be synchronized to assure data consistency
			// i.e. Operands cannot be processed before they are fully transfered to the device
			if (i == 0) {
				uploadOperandsToDevice2Fuzz(i);
				cudaStatus = cudaStreamSynchronize(streams[i]);
				if (cudaStatus != cudaSuccess) {
					cerr << "CUDA Stream synchronization failed in fls::fuzzExecution in line 847: " << cudaGetErrorString(cudaStatus) << endl;
					cin.get();
					exit(1);
				}
			}
			else if (i == 1) {
				executeMinimumKernel(i - 1);
				cudaStatus = cudaStreamSynchronize(streams[i - 1]);
				if (cudaStatus != cudaSuccess) {
					cerr << "CUDA Stream synchronization failed in fls::fuzzExecution in line 856: " << cudaGetErrorString(cudaStatus) << endl;
					cin.get();
					exit(1);
				}
				uploadOperandsToDevice2Fuzz(i);
				cudaStatus = cudaStreamSynchronize(streams[i]);
				if (cudaStatus != cudaSuccess) {
					cerr << "CUDA Stream synchronization failed in fls::fuzzExecution in line 863: " << cudaGetErrorString(cudaStatus) << endl;
					cin.get();
					exit(1);
				}
			}
			else {
				downloadResultFromDevice2Fuzz(i - 2);
				cudaStatus = cudaStreamSynchronize(streams[i - 2]);
				if (cudaStatus != cudaSuccess) {
					cerr << "CUDA Stream synchronization failed in fls::fuzzExecution in line 872: " << cudaGetErrorString(cudaStatus) << endl;
					cin.get();
					exit(1);
				}
				executeMinimumKernel(i - 1);
				if ((i - 1) < streams.size()) {
					cudaStatus = cudaStreamSynchronize(streams[i - 1]);
					if (cudaStatus != cudaSuccess) {
						cerr << "CUDA Stream synchronization failed in fls::fuzzExecution in line 880: " << cudaGetErrorString(cudaStatus) << endl;
						cin.get();
						exit(1);
					}
				}
				uploadOperandsToDevice2Fuzz(i);
				if (i < streams.size()) {
					cudaStatus = cudaStreamSynchronize(streams[i]);
					if (cudaStatus != cudaSuccess) {
						cerr << "CUDA Stream synchronization failed in fls::fuzzExecution in line 889: " << cudaGetErrorString(cudaStatus) << endl;
						cin.get();
						exit(1);
					}
				}
			}
		}
		for (unsigned int i = 0; i < execPlan.size(); i++) {
			// Once the resulting set membership is computed, update the
			// Range
			execPlan[i].host_result->setRange(-1, 1);
			// Name
			string name = execPlan[i].host_operand2->getName();
			execPlan[i].host_result->setName(name);
			// Normalization value
			execPlan[i].host_result->setNorm();
			// Shape
			execPlan[i].host_result->setShape("Unknown");
			// Parameters
			execPlan[i].host_result->setParams({});
			// Support
			execPlan[i].host_result->setSupport();
			// and Crisp value
			execPlan[i].host_result->defuzzification();
		}
	}
	// Otherwise, the Sequential or Heterogeneous execution of Fuzzification is performed
	else {
		for (unsigned int i = 0; i < execPlan.size(); i++) {
			string name = execPlan[i].host_operand2->getName();
			// Perform the Non-Singleton Fuzzification according to the Execution Plan
			(*execPlan[i].host_result) = (*execPlan[i].host_operand1) && (*execPlan[i].host_operand2);
			// Once the resulting set membership is computed, update the
			// Range
			execPlan[i].host_result->setRange(-1, 1);
			// Name
			execPlan[i].host_result->setName(name);
		}
	}
}

// ***********************************************************
// INFERENCE SCHEDULING 
// ***********************************************************
void fls::inferSchedule() {
	size_t index;
	unsigned int n;
	string str;
	fs *op1, *op2;
	firePtrs.resize(ruleCount);
	vector<string> premInRule, conseqInRule;
	// The Binary Operation reference
	vector<string> binOpList;
	resOfOps.resize(ruleCount);
	if (ruleCount != ruleSet.size())
		firingStrengths.resize(ruleCount + 1, fs(-1, 1, "fire", isCUDA, isStream));
	else
		firingStrengths.resize(ruleCount, fs(-1, 1, "fire", isCUDA, isStream));

	for (unsigned int i = 0; i < ruleCount; i++) {
		unsigned int binOpsReq = ruleSet[i].getOperations().size();
		unsigned int numOpsInRule = ruleSet[i].getPremises().size();
		vector<fs> a;		
		for (unsigned int j = 0; j < (binOpsReq - numOpsInRule - 1); j++) {
			ostringstream convert1, convert2;
			convert1 << i; convert2 << j;
			fs b(-1, 1, "res" + convert1.str() + "-" + convert2.str(), isCUDA, isStream);
			a.push_back(b);
		}
		resOfOps[i] = a;
	}
	vector<vector<fs *>> relatedSetList(ruleCount);

	for (unsigned int i = 0; i < ruleCount; i++) {
		// Get the premise set string from each rule 
		premInRule = ruleSet[i].getPremises();
		// Get the consequence set string from each rule
		conseqInRule = ruleSet[i].getConsequences();

		// Verify if the input variable and rule premise sets exist in the Fuzzy System
		for (unsigned int j = 0; j < premInRule.size(); j++)// {
			check4ExistingVarAndSet(premInRule[j]);
		// Verify if the output variable and rule consequence sets exist in the Fuzzy System
		for (unsigned int j = 0; j < conseqInRule.size(); j++)// {
			check4ExistingVarAndSet(conseqInRule[j]);

		// *****************************************************************
		// SCHEDULE THE FUZZY OPERATIONS FOR IMPLICATION
		// *****************************************************************

		// Get the Binary Operation References from the rule set
		binOpList = ruleSet[i].getOperations();		
		// Create a Binary Operation List based on the Binary Operation References
		// Map every related set acording to rules
		vector<fs *> beta(binOpList.size());
		for (unsigned int j = 0; j < premInRule.size(); j++) {
			// Get the premise index from the Binary Operation List
			stringstream convert2Num(binOpList[j]);
			convert2Num >> n;
			// Get the fuzzy set according to its name
			beta[j] = getFSPtr(premInRule[n]);
		}
		relatedSetList[i] = beta;
		// Set the operation results pointers
		for (unsigned int j = premInRule.size(); j < binOpList.size(); j++) {
			// For intermediate operations, results are stored in intermediate sets;
			// otherwise, the final result is stored in the firing strength set
			if (j < binOpList.size() - 1)
				relatedSetList[i][j] = &resOfOps[i][j - premInRule.size()];
			else
				relatedSetList[i][j] = &firingStrengths[i];
		}
		
		// Starts computing the rule based on the Binary Operation List
		vector<string> t;
		for (unsigned int j = premInRule.size(); j < binOpList.size(); j++) {
			// First, it tries with the Fuzzy Complement
			t = token(binOpList[j], { "NOT" }, true);
			// If a Fuzzy Complement is found, the Inference Machine searches the ONLY related premise from the Premise String Queue
			if (t[0].size() == 0) {
				index = t[1].find_first_of("()");
				while (index <= t[1].size()) {
					t[1].erase(index, 1);
					index = t[1].find_first_of("()", index + 1);
				}
				stringstream convert2Num(t[1]);
				convert2Num >> n;
				str = premInRule[n];

				// Select the SINGLE related operand
				op1 = relatedSetList[i][n];
			}
			// Otherwise, it must be a Fuzzy Intersection or Union
			else {
				// Secondly, it tries with the Fuzzy Intersection				
				t = token(binOpList[j], { "AND" }, true);
				// If a Fuzzy Intersection is found, the Inference Machine searches the TWO related premises from the Premise String Queue
				if (t[0].size() != binOpList[j].size()) {
					for (unsigned int k = 0; k < 2; k++) {
						index = t[k].find_first_of("()");
						while (index <= t[k].size()) {
							t[k].erase(index, 1);
							index = t[k].find_first_of("()", index + 1);
						}
						stringstream convert2Num(t[k]);
						convert2Num >> n;
						// Select the related operands
						if (k == 0)
							op1 = relatedSetList[i][n];
						else
							op2 = relatedSetList[i][n];
					}
				}
				else {
					// Otherwise, it must be a Fuzzy Union
					t = token(binOpList[j], { "OR" }, true);
					if (t[0].size() != binOpList[j].size()) {
						for (unsigned int k = 0; k < 2; k++) {
							index = t[k].find_first_of("()");
							while (index <= t[k].size()) {
								t[k].erase(index, 1);
								index = t[k].find_first_of("()", index + 1);
							}
							stringstream convert2Num(t[k]);
							convert2Num >> n;
							// Select the related operands
							if (k == 0)
								op1 = relatedSetList[i][n];
							else
								op2 = relatedSetList[i][n];
						}
					}
				}
			}
			// Assign a stage to each rule operation
			unsigned int stg = setStage(binOpList[j], premInRule.size(), 0, &binOpList);
			// Once the operands and the fuzzy operation are defined, the rule must be computed			
			index = binOpList[j].find_first_of("ANO");
			// If an Intersection Operation is involved, the T-norm operation will be computed
			if (binOpList[j][index] == 'A'){
				// If isStream is enabled, operations will not be executed until the execution plan is ready. Otherwise, operations will be heterogeneously or sequentially executed.
				roh u{ stg, "tnorm:" + conjOperator, op1, op2, relatedSetList[i][j] };
				execPlan.push_back(u);
			}
			// If an Union Operation is involved, the S-norm operation will be computed
			else if (binOpList[j][index] == 'O'){
				// If isStream is enabled, operations will not be executed until the execution plan is ready. Otherwise, operations will be heterogeneously or sequentially executed.
				roh u{ stg, "snorm:" + disjOperator, op1, op2, relatedSetList[i][j] };
				execPlan.push_back(u);
			}
			// If a Complement Operation is involved, the NOT operation will be computed
			else if (binOpList[j][index] == 'N') {
				// Fuzzy Complement is not CUDA accelerated. Therefore, Streams are not applicable
				roh u{ stg, "complement", op1, NULL, relatedSetList[i][j] };
				execPlan.push_back(u);
			}
		}
	}
	// Enable all the CUDA Streams previously created in fls::inferSchedule() for concurrent execution
	unsigned int numOfStages = 0;
	for (unsigned int i = 0; i < execPlan.size(); i++) {
		if (execPlan[i].stage > numOfStages)
			numOfStages = execPlan[i].stage;
	}
	numOfStages++;

	// Separate every operation by stages
	inferStages.resize(numOfStages, vector<roh>(0, { 0, "", NULL, NULL, NULL }));
	for (int h = 0; h < numOfStages; h++) {
		for (int i = 0; i < execPlan.size(); i++) {
			if (execPlan[i].stage == h)
				inferStages[h].push_back(execPlan[i]);
		}
	}
}

// ***********************************************************
// INFERENCE EXECUTION
// ***********************************************************
void fls::inferExecution() {
	nvtxRangeId_t t12 = nvtxRangeStartEx(&events[12]);	
	// If Fuzzy System isStream is enabled, then the concurrent execution of Inference is performed by stages
	if (isStream) {
		// Overlap every CUDA execution stage according to the Execution Plan for Inference
		for (int h = 0; h < inferStages.size(); h++) {
			for (int i = 0; i < inferStages[h].size() + 2; i++) {
				// If device has Dual Copy Engine, then Pipeline is 3
				// Otherwise, Pipeline is 2
				//
				// U0 K0 D0
				//    U1 K1 D1
				//       U2 K2 D2
				//          U3 K3 D3
				//             U4 K4 D4
				//                U5 K5 D5
				// where
				// U: is uploadOperandsToDevice(i)
				// K: is executeMinimumKernel(i - 1)
				// D: is downloadResultFromDevice(i - 2)
				// Every Stream must be synchronized to assure data consistency
				// i.e. Operands cannot be processed before they are fully transfered to the device		

				if (i == 0) {
					uploadOperandsToDevice2Infer(h, i);
					cudaStatus = cudaStreamSynchronize(streams[i]);
					if (cudaStatus != cudaSuccess) {
						cerr << "CUDA Stream synchronization failed in fls::inferExecution in line 1144: " << cudaGetErrorString(cudaStatus) << endl;
						cin.get();
						exit(1);
					}
				}
				else {
					vector<string> t;
					string op;
					string type;
					if ((i - 1) < inferStages[h].size()) {
						t = token(inferStages[h][i - 1].operation, { ":" }, true);
						if (t[0].size() != inferStages[h][i - 1].operation.size()) {
							op = t[0];
							type = t[1];
						}
						else {
							op = inferStages[h][i - 1].operation;
							type = "";
						}
					}
					if (i == 1) {
						if (op.compare("tnorm") == 0)
							executeTNormKernel(h, i - 1, type);
						else if (op.compare("snorm") == 0)
							executeSNormKernel(h, i - 1, type);
						else
							executeComplementKernel(h, i - 1);
						cudaStatus = cudaStreamSynchronize(streams[i - 1]);
						if (cudaStatus != cudaSuccess) {
							cerr << "CUDA Stream synchronization failed in fls::inferExecution in line 1173: " << cudaGetErrorString(cudaStatus) << endl;
							cin.get();
							exit(1);
						}
						uploadOperandsToDevice2Infer(h, i);
						cudaStatus = cudaStreamSynchronize(streams[i]);
						if (cudaStatus != cudaSuccess) {
							cerr << "CUDA Stream synchronization failed in fls::inferExecution in line 1180: " << cudaGetErrorString(cudaStatus) << endl;
							cin.get();
							exit(1);
						}
					}
					else {
						downloadResultFromDevice2Infer(h, i - 2);
						cudaStatus = cudaStreamSynchronize(streams[i - 2]);
						if (cudaStatus != cudaSuccess) {
							cerr << "CUDA Stream synchronization failed in fls::inferExecution in line 1189: " << cudaGetErrorString(cudaStatus) << endl;
							cin.get();
							exit(1);
						}
						if (op.compare("tnorm") == 0)
							executeTNormKernel(h, i - 1, type);
						else if (op.compare("snorm") == 0)
							executeSNormKernel(h, i - 1, type);
						else
							executeComplementKernel(h, i - 1);
						if ((i - 1) < streams.size()) {
							cudaStatus = cudaStreamSynchronize(streams[i - 1]);
							if (cudaStatus != cudaSuccess) {
								cerr << "CUDA Stream synchronization failed in fls::inferExecution in line 1202: " << cudaGetErrorString(cudaStatus) << endl;
								cin.get();
								exit(1);
							}
						}
						uploadOperandsToDevice2Infer(h, i);
						if (i < streams.size()) {
							cudaStatus = cudaStreamSynchronize(streams[i]);
							if (cudaStatus != cudaSuccess) {
								cerr << "CUDA Stream synchronization failed in fls::inferExecution in line 1211: " << cudaGetErrorString(cudaStatus) << endl;
								cin.get();
								exit(1);
							}
						}
					}
				}
			}
		}

		// Assure destroying every Stream after Inference
		for (int i = 0; i < streams.size(); i++) {
			cudaStatus = cudaStreamDestroy(streams[i]);
			if (cudaStatus != cudaSuccess) {
				cerr << "CUDA Stream destroy failed in fls::inferExecution in line 1151: " << cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
		}

		vector<float> dsc = linearSpace(-1, 1, samples);
		fs full(-1, 1, 1, "FULL", "Interval", {-1, 1}, isCUDA, isStream);
		for (int i = 0; i < firingStrengths.size(); i++) {
			if (i != firingStrengths.size() - 1) {
				// Once the resulting set membership is computed, update the
				// Range
				firingStrengths[i].discourse = dsc;
				// Normalization value
				firingStrengths[i].setNorm();
				// Name
				ostringstream convert;
				convert << i;
				string str = convert.str();
				firingStrengths[i].setName("f_" + str);
				// Shape
				firingStrengths[i].setShape("Unknown");
				// Parameters
				firingStrengths[i].setParams({});
				// Support
				firingStrengths[i].setSupport();
				// and Crisp value
				firingStrengths[i].defuzzification();
				if (ruleCount != ruleSet.size())
					full = full & (!firingStrengths[i]);
			}
			else {
				if (ruleCount != ruleSet.size()) {
					firingStrengths[i] = full;
					firingStrengths[i].setName("else");
				}
				else {
					// Range
					firingStrengths[i].discourse = dsc;
					// Normalization value
					firingStrengths[i].setNorm();
					// Name
					ostringstream convert;
					convert << i;
					string str = convert.str();
					firingStrengths[i].setName("f_" + str);
					// Shape
					firingStrengths[i].setShape("Unknown");
					// Parameters
					firingStrengths[i].setParams({});
					// Support
					firingStrengths[i].setSupport();
					// and Crisp value
					firingStrengths[i].defuzzification();
				}
			}
		}
		//for (int h = 0; h < stages.size(); h++) {
		//	for (int i = 0; i < stages[h].size(); i++) {
		//		// Once the resulting set membership is computed, update the
		//		// Range
		//		stages[h][i].host_result->discourse = dsc;
		//		// Normalization value
		//		stages[h][i].host_result->setNorm();
		//		// Name
		//		string op;
		//		vector<string> t = token(stages[h][i].operation, { ":" }, true);
		//		if (t[0].size() != stages[h][i].operation.size())
		//			op = t[0];
		//		else
		//			op = stages[h][i].operation;

		//		if (op.compare("tnorm") == 0)
		//			stages[h][i].host_result->setName("(" + stages[h][i].host_operand1->getName() + ")" + " AND (" + stages[h][i].host_operand2->getName() + ")");
		//		else if (op.compare("snorm") == 0)
		//			stages[h][i].host_result->setName("(" + stages[h][i].host_operand1->getName() + ")" + " OR (" + stages[h][i].host_operand2->getName() + ")");
		//		else
		//			stages[h][i].host_result->setName("NOT(" + stages[h][i].host_operand1->getName() + ")");

		//		// Shape
		//		stages[h][i].host_result->setShape("Unknown");
		//		// Parameters
		//		stages[h][i].host_result->setParams({});
		//		// Support
		//		stages[h][i].host_result->setSupport();
		//		// and Crisp value
		//		stages[h][i].host_result->defuzzification();
		//	}
		//}
	}
	// Otherwise, the Sequential or Heterogeneous execution of Fuzzification is performed
	else {
		for (int h = 0; h < inferStages.size(); h++) {
			for (int i = 0; i < inferStages[h].size(); i++) {
				string op;
				vector<string> t = token(inferStages[h][i].operation, { ":" }, true);
				if (t[0].size() != inferStages[h][i].operation.size())
					op = t[0];
				else
					op = inferStages[h][i].operation;

				if (op.compare("tnorm") == 0)
					(*inferStages[h][i].host_result) = (*inferStages[h][i].host_operand1) & (*inferStages[h][i].host_operand2);
				else if (op.compare("snorm") == 0)
					(*inferStages[h][i].host_result) = (*inferStages[h][i].host_operand1) | (*inferStages[h][i].host_operand2);
				else
					(*inferStages[h][i].host_result) = !(*inferStages[h][i].host_operand1);
			}
		}
		if (ruleCount != ruleSet.size()) {
			fs full(-1, 1, 1, "FULL", "Interval", { -1, 1 }, isCUDA, isStream);
			for (unsigned int i = 0; i < firingStrengths.size(); i++)			 {				
				if (i != firingStrengths.size() - 1)
					full = full & (!firingStrengths[i]);
				else
					firingStrengths[i] = full;
			}
		}
	}
	execPlan.clear();
	nvtxRangeEnd(t12);
}

void fls::aggregation() {
	nvtxRangeId_t t13 = nvtxRangeStartEx(&events[13]);	
	vector<string> conseqInRule;
	fvar *varPtr = NULL;
	// Once we know the firing strength for the i-th rule, we must define its corresponding consequence
	for (unsigned int i = 0; i < ruleSet.size(); i++) {
		conseqInRule = ruleSet[i].getConsequences();		
		for (unsigned int j = 0; j < conseqInRule.size(); j++) {
			// Separates the variable name and set name from the Rule Consequence Set String
			vector<string> t = token(conseqInRule[j], { ":" }, true);
			// Get the variable pointer to an existing output variable
			varPtr = getVarPtr(t[0]);
			// Searching for consequent set
			for (unsigned int k = 0; k < outTrimSets[varPtr->getVarID()].getSetCount(); k++) {
				string name = outTrimSets[varPtr->getVarID()].sets[k].getName();				
				// If the consequent set name is found in the Fuzzy System, then update the Output Trimmer Sets
				if (varPtr->getFuzzySet(k).getName().compare(conseqInRule[j]) == 0) {
					// First, aggregate every Firing Strengths according to the rule specification
					outTrimSets[varPtr->getVarID()].sets[k] = outTrimSets[varPtr->getVarID()].sets[k] || firingStrengths[i];
					// Update the Trimmer Set name
					if (outTrimSets[varPtr->getVarID()].sets[k].getNormalization() == 0)
						outTrimSets[varPtr->getVarID()].sets[k].setName("(" + firingStrengths[i].getName() + " THEN (" + name + "))");
					else
						outTrimSets[varPtr->getVarID()].sets[k].setName("(" + firingStrengths[i].getName() + " THEN (" + name + ")) AND ");
					break;
				}
			}			
		}
	}

	// Trimmer Set Overlapping
	for (unsigned int j = 0; j < outTrimSets.size(); j++) {
		for (unsigned int k = 0; k < outTrimSets[j].getSetCount(); k++) {
			// Finally the resulting Trimmer Sets must be shifted in order to overlap its corresponding Consequencec Set
			// Get the Consequence Set shape
			string shp = output_variables[j].sets[k].getShape();
			// Get the Consequence Set parameters
			vector<float> p = output_variables[j].sets[k].getParameters();
			// Shift the Trimmer Set according to the Consequence Set parameters
			if (outTrimSets[j].sets[k].getShape().compare("Unknown") == 0) {
				if (shp.compare("Interval") == 0)
					outTrimSets[j].sets[k].setOffset((p[0] + p[1]) / 2);
				else if (shp.compare("Triangular") == 0)
					outTrimSets[j].sets[k].setOffset(p[1]);
				else if (shp.compare("Trapezoidal") == 0)
					outTrimSets[j].sets[k].setOffset((p[1] + p[2]) / 2);
				else if (shp.compare("S") == 0)
					outTrimSets[j].sets[k].setOffset(p[1]);
				else if (shp.compare("Z") == 0)
					outTrimSets[j].sets[k].setOffset(p[0]);
				else if (shp.compare("Gaussian") == 0)
					outTrimSets[j].sets[k].setOffset(p[0]);
				else
					outTrimSets[j].sets[k].setOffset(p[0]);
			}
		}
	}

	// Compute the Inferred Sets
	vector<fvar> result = output_variables;
	for (unsigned int i = 0; i < result.size(); i++) {
		string name = output_variables[i].getName();
		// Trim every consequence set in every output variable
		result[i].trimSets(outTrimSets[i].getFuzzySet());
		// Merge by union operation all the trimmed sets to get the Inferred Sets
		result[i].unionMerging();
		// Update the Inferred Sets
		inferredSets[i] = result[i].getFuzzySet(i);
		inferredSets[i].setName("Inferred_" + name);
	}
	input_variables = input_backup;
	nvtxRangeEnd(t13);
}

void fls::configure() {
	nvtxRangeId_t t11 = nvtxRangeStartEx(&events[11]);
	inferSchedule();
	nvtxRangeEnd(t11);
}

unsigned int fls::setStage(const string &op, const unsigned int &condition, const unsigned int &initialStage, vector<string> *list) const{
	vector<string> t, availableOps = { "AND", "OR", "NOT" };
	size_t index; unsigned int n, newStage, l, m1 = initialStage, m2 = initialStage;
	t = token(op, availableOps, true);
	if (t[0].size() == 0){
		t[0] = t[1];
		l = 1;
	}
	else
		l = 2;
	for (unsigned int k = 0; k < l; k++) {
		index = t[k].find_first_of("()");
		while (index <= t[k].size()) {
			t[k].erase(index, 1);
			index = t[k].find_first_of("()", index + 1);
		}
		stringstream convert2Num(t[k]);
		convert2Num >> n;

		if (k == 0) {
			if (n >= condition) {
				unsigned int stage = setStage((*list)[n], condition, m1 + 1, list);
				m1 = stage;
			}
		}
		else {
			if (n >= condition) {
				unsigned int stage = setStage((*list)[n], condition, m2 + 1, list);
				m2 = stage;
			}
		}
	}
	if (m1 > m2)
		newStage = m1;
	else
		newStage = m2;
	return newStage;
}

void fls::uploadOperandsToDevice2Infer(const unsigned int &stg, const int &index) {
	if (index >= 0 && index < inferStages[stg].size()) {
		fs *sourceSet1 = inferStages[stg][index].host_operand1;
		float *host_op1 = &sourceSet1->membership[0];
		cudaStatus = cudaMemcpyAsync(dev_op1, host_op1, samples * sizeof(float), cudaMemcpyHostToDevice, streams[index]);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fls::uploadOperandstoDevice (operand 1): " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		if (inferStages[stg][index].host_operand2 != NULL) {
			fs *sourceSet2 = inferStages[stg][index].host_operand2;
			float *host_op2 = &sourceSet2->membership[0];
			cudaStatus = cudaMemcpyAsync(dev_op2, host_op2, samples * sizeof(float), cudaMemcpyHostToDevice, streams[index]);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fls::uploadOperandstoDevice (operand 2): " << cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
		}
	}
}

void fls::downloadResultFromDevice2Infer(const unsigned int &stg, const int &index) {
	if (index >= 0 && index < inferStages[stg].size()) {
		fs *destinySet = inferStages[stg][index].host_result;
		float *host_result1 = &destinySet->membership[0];
		cudaStatus = cudaMemcpyAsync(host_result1, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost, streams[index]);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to Host data transfer failed in fls::downloadResultFromDevice (result): " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
}

void fls::executeTNormKernel(const unsigned int &stg, const int &index, const string &type) {
	if (index >= 0 && index < inferStages[stg].size()) {
		if (type.compare("Minimum") == 0)
			_minimumIntersection <<<grids, blocks, 0, streams[index]>>> (dev_result1, dev_op1, dev_op2);
		else if (type.compare("Product") == 0)
			_productIntersection <<<grids, blocks, 0, streams[index]>>> (dev_result1, dev_op1, dev_op2);
		else if (type.compare("Bounded") == 0)
			_boundedIntersection <<<grids, blocks, 0, streams[index]>>> (dev_result1, dev_op1, dev_op2, float(1.0));
		else if (type.compare("Drastic") == 0)
			_drasticIntersection <<<grids, blocks, 0, streams[index]>>> (dev_result1, dev_op1, dev_op2, float(1.0));
		if (cudaGetLastError() != cudaSuccess) {
			cerr << "Kernel execution failed in fls::executeTNormKernel: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
}

void fls::executeSNormKernel(const unsigned int &stg, const int &index, const string &type) {
	if (index >= 0 && index < inferStages[stg].size()) {
		if (type.compare("Maximum") == 0)
			_maximumUnion <<<grids, blocks, 0, streams[index]>>> (dev_result1, dev_op1, dev_op2);
		else if (type.compare("Algebraic Sum") == 0)
			_algebraicUnion <<<grids, blocks, 0, streams[index]>>> (dev_result1, dev_op1, dev_op2);
		else if (type.compare("Bounded") == 0)
			_boundedUnion <<<grids, blocks, 0, streams[index]>>> (dev_result1, dev_op1, dev_op2, float(1.0));
		else if (type.compare("Drastic") == 0)
			_drasticUnion <<<grids, blocks, 0, streams[index]>>> (dev_result1, dev_op1, dev_op2, float(1.0));
		if (cudaGetLastError() != cudaSuccess) {
			cerr << "Kernel execution failed in fls::executeSNormKernel: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
}

void fls::executeComplementKernel(const unsigned int &stg, const unsigned &index) {
	if (index >= 0 && index < inferStages[stg].size()) {
		_complement <<<grids, blocks, 0, streams[index]>>> (dev_result1, dev_op1);
		if (cudaGetLastError() != cudaSuccess) {
			cerr << "Kernel execution failed in fls::executeComplementKernel: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
}

void fls::uploadOperandsToDevice2Fuzz(const int &index) {
	//nvtxRangeId_t t003 = nvtxRangeStartEx(&events[3]);
	if (index >= 0 && index < execPlan.size()) {
		fs *sourceSet1 = execPlan[index].host_operand1;
		float *host_op1 = &sourceSet1->membership[0];
		cudaStatus = cudaMemcpyAsync(dev_op1, host_op1, samples * sizeof(float), cudaMemcpyHostToDevice, streams[index]);
		if (cudaStatus != cudaSuccess) {
			cerr << "Host to device data transfer failed in fls::uploadOperandstoDevice (operand 1): " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
		if (execPlan[index].host_operand2 != NULL) {
			fs *sourceSet2 = execPlan[index].host_operand2;
			float *host_op2 = &sourceSet2->membership[0];
			cudaStatus = cudaMemcpyAsync(dev_op2, host_op2, samples * sizeof(float), cudaMemcpyHostToDevice, streams[index]);
			if (cudaStatus != cudaSuccess) {
				cerr << "Host to device data transfer failed in fls::uploadOperandstoDevice (operand 2): " << cudaGetErrorString(cudaStatus) << endl;
				cin.get();
				exit(1);
			}
		}
	}
	//nvtxRangeEnd(t003);
}

void fls::downloadResultFromDevice2Fuzz(const int &index) {
	//nvtxRangeId_t t004 = nvtxRangeStartEx(&events[4]);
	if (index >= 0 && index < execPlan.size()) {
		fs *destinySet = execPlan[index].host_result;
		float *host_result1 = &destinySet->membership[0];
		cudaStatus = cudaMemcpyAsync(host_result1, dev_result1, samples * sizeof(float), cudaMemcpyDeviceToHost, streams[index]);
		if (cudaStatus != cudaSuccess) {
			cerr << "Device to Host data transfer failed in fls::downloadResultFromDevice (result): " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
	//nvtxRangeEnd(t004);
}

void fls::executeMinimumKernel(const int &index) {
	if (index >= 0 && index < execPlan.size()) {
		_minimumIntersection <<<grids, blocks, 0, streams[index]>>> (dev_result1, dev_op1, dev_op2);
		if (cudaGetLastError() != cudaSuccess) {
			cerr << "Kernel execution failed in fls::executeMinimumKernel: " << cudaGetErrorString(cudaStatus) << endl;
			cin.get();
			exit(1);
		}
	}
}

vector<float> fls::defuzzify() const{
	nvtxRangeId_t t200 = nvtxRangeStartEx(&events[20]);
	vector<float> result(inferredSets.size());
	for (unsigned int i = 0; i < result.size(); i++)
		result[i] = inferredSets[i].defuzzify();
	nvtxRangeEnd(t200);
	return result;
}

float **fls::getSurface(const string &inVar1, const string &inVar2, const string &outVar, const vector<float> &sigmas) {
	fuzzySurf = new float *[samples];
	for (unsigned int i = 0; i < samples; i++)
		fuzzySurf[i] = new float[samples];
	vector<float> result;
	vector<float> discourse1, discourse2, range1, range2;
	string name1, name2;
	fvar *varPtr = NULL;
	for (unsigned int i = 0; i < 3; i++){
		if (i == 0) {
			varPtr = getVarPtr(inVar1);
			discourse1 = varPtr->getFuzzySet(0).getDiscourse();
			range1 = varPtr->getRange();
			name1 = varPtr->getName();
		}
		else if (i == 1) {
			varPtr = getVarPtr(inVar2);
			discourse2 = varPtr->getFuzzySet(0).getDiscourse();
			range2 = varPtr->getRange();
			name2 = varPtr->getName();
		}
		else
			varPtr = getVarPtr(outVar);
	}

	for (unsigned int i = 0; i < samples; i++) {
		for (unsigned int j = 0; j < samples; j++) {
			fs a(range1[0], range1[1], 1.0, name1 + "_prime", "Gaussian", { discourse1[i], sigmas[0] }, isCUDA, isStream);
			fs b(range2[0], range2[1], 1.0, name2 + "_prime", "Gaussian", { discourse2[j], sigmas[1] }, isCUDA, isStream);
			fuzzify({ a, b });
			infer();
			result = defuzzify();
			fuzzySurf[i][j] = result[0];
		}
	}
	return fuzzySurf;
}

float **fls::getSurface(const string &inVar1, const string &inVar2, const string &outVar) {
	fuzzySurf = new float *[samples];
	for (unsigned int i = 0; i < samples; i++)
		fuzzySurf[i] = new float[samples];
	vector<float> result;
	vector<float> discourse1, discourse2, range1, range2;
	string name1, name2;
	fvar *varPtr = NULL;
	for (unsigned int i = 0; i < 3; i++){
		if (i == 0) {
			varPtr = getVarPtr(inVar1);
			discourse1 = varPtr->getFuzzySet(0).getDiscourse();
			range1 = varPtr->getRange();
			name1 = varPtr->getName();
		}
		else if (i == 1) {
			varPtr = getVarPtr(inVar2);
			discourse2 = varPtr->getFuzzySet(0).getDiscourse();
			range2 = varPtr->getRange();
			name2 = varPtr->getName();
		}
		else
			varPtr = getVarPtr(outVar);
	}

	for (unsigned int i = 0; i < samples; i++) {
		for (unsigned int j = 0; j < samples; j++) {
			fs a(range1[0], range1[1], 1.0, name1 + "_prime", "Singleton", { discourse1[i] }, isCUDA, isStream);
			fs b(range2[0], range2[1], 1.0, name2 + "_prime", "Singleton", { discourse2[j] }, isCUDA, isStream);
			fuzzify({ a, b });
			infer();
			result = defuzzify();
			fuzzySurf[i][j] = result[0];
		}
	}
	return fuzzySurf;
}

// FUZZY SYSTEM: SET FUNCTIONS

void fls::setName(const string &proposedName) {
	name = proposedName;
}

void fls::setInferenceModel(const string &proposedModel) {
	inferenceModel = proposedModel;
}

void fls::setHetProc(const bool &proc) {
	isCUDA = proc;
	for (unsigned int i = 0; i < inVarCount; i++)
		input_variables[i].setHetProc(proc);
	for (unsigned int i = 0; i < outVarCount; i++)
		output_variables[i].setHetProc(proc);
}

void fls::setStreams(const bool &s) {
	if (!isCUDA && s)
		throw invalidStreamEnableFault();
	isStream = s;
	for (unsigned int i = 0; i < inVarCount; i++)
		input_variables[i].setStream(s);
	for (unsigned int i = 0; i < outVarCount; i++)
		output_variables[i].setStream(s);
}

// FUZZY SYSTEM: GET FUNCTIONS

size_t fls::getMaxRuleCount() const {
	size_t u = input_variables[0].getSetCount();
	for (size_t i = 1; i < input_variables.size(); i++) {
		u *= input_variables[i].getSetCount();
	}
	return u;
}

vector<fs> fls::getInferredSets() const {
	return inferredSets;
}

vector<fs> fls::getFiringStrengths() const {
	return firingStrengths;
}

vector<fs> fls::getImpliedSets() const {
	vector<fs> result;
	for (unsigned int i = 0; i < inVarCount; i++)
		for (unsigned int j = 0; j < input_variables[i].getSetCount(); j++)
			result.push_back(input_variables[i].sets[j]);
	return result;
}

const fls &fls::operator=(const fls &right) {
	if (&right != this) {
		this->name = right.name;
		this->inVarCount = right.inVarCount;
		this->outVarCount = right.outVarCount;
		this->ruleCount = right.ruleCount;
		this->isCUDA = right.isCUDA;
		this->isStream = right.isStream;
		this->inferenceModel = right.inferenceModel;
		this->maxRuleCount = right.maxRuleCount;
		this->crisp = right.crisp;
		this->conjOperator = right.conjOperator;
		this->disjOperator = right.disjOperator;
		this->aggregMethod = right.aggregMethod;
		this->defuzzMethod = right.defuzzMethod;
		this->input_variables = right.input_variables;
		this->output_variables = right.output_variables;
		this->outTrimSets = right.outTrimSets;
		this->ruleSet = right.ruleSet;
		this->firingStrengths = right.firingStrengths;
		this->firePtrs = right.firePtrs;
		this->inferredSets = right.inferredSets;
		this->numOfSets = right.numOfSets;
		this->fuzzySurf = right.fuzzySurf;
		this->relatedSetList = right.relatedSetList;
		this->resOfOps = right.resOfOps;
		this->stages = right.stages;
	}
	return *this;
}

unsigned int fls::getInVarCount() const {
	return inVarCount;
}

unsigned int fls::getOutVarCount() const {
	return outVarCount;
}

unsigned int fls::getRuleCount() const {
	return ruleCount;
}

string fls::getInVarName(const unsigned int &index) const {
	return input_variables[index].getName();
}

string fls::getOutVarName(const unsigned int &index) const {
	return output_variables[index].getName();
}

void fls::saveResultingRulesAsStr(const string &filePath) {
	ofstream outputFile;
	outputFile.open(filePath);
	unsigned int n;
	vector<string> premInRule, conseqInRule;
	// The Binary Operation reference
	vector<string> binOpList;	
	for (unsigned int i = 0; i < ruleCount; i++) {
		// Get the premise set string from each rule 
		premInRule = ruleSet[i].getPremises();
		// Get the consequence set string from each rule
		conseqInRule = ruleSet[i].getConsequences();

		// Get the Binary Operation References from the rule set
		binOpList = ruleSet[i].getOperations();
		
		vector<string> beta(binOpList.size());
		for (unsigned int j = 0; j < premInRule.size(); j++) {
			stringstream convert2Num(binOpList[j]);
			convert2Num >> n;
			beta[j] = premInRule[n];
		}		
		
		vector<string> t;
		string str, op1, op2, operation;
		unsigned int index;
		for (unsigned int j = premInRule.size(); j < binOpList.size(); j++) {
			t = token(binOpList[j], { "NOT" }, true);
			// If a Fuzzy Complement is found, the Inference Machine searches the ONLY related premise from the Premise String Queue
			if (t[0].size() == 0) {
				index = t[1].find_first_of("()");
				while (index <= t[1].size()) {
					t[1].erase(index, 1);
					index = t[1].find_first_of("()", index + 1);
				}
				stringstream convert2Num(t[1]);
				convert2Num >> n;
				str = premInRule[n];

				// Select the SINGLE related operand
				operation = "NOT(" + beta[n] + ")";
			}
			// Otherwise, it must be a Fuzzy Intersection or Union
			else {
				// Secondly, it tries with the Fuzzy Intersection				
				t = token(binOpList[j], { "AND" }, true);
				// If a Fuzzy Intersection is found, the Inference Machine searches the TWO related premises from the Premise String Queue
				if (t[0].size() != binOpList[j].size()) {
					for (unsigned int k = 0; k < 2; k++) {
						index = t[k].find_first_of("()");
						while (index <= t[k].size()) {
							t[k].erase(index, 1);
							index = t[k].find_first_of("()", index + 1);
						}
						stringstream convert2Num(t[k]);
						convert2Num >> n;
						// Select the related operands
						if (k == 0)
							op1 = beta[n];
						else
							op2 = beta[n];
					}
					operation = "(" + op1 + ")AND(" + op2 + ")";
				}
				else {
					// Otherwise, it must be a Fuzzy Union
					t = token(binOpList[j], { "OR" }, true);
					if (t[0].size() != binOpList[j].size()) {
						for (unsigned int k = 0; k < 2; k++) {
							index = t[k].find_first_of("()");
							while (index <= t[k].size()) {
								t[k].erase(index, 1);
								index = t[k].find_first_of("()", index + 1);
							}
							stringstream convert2Num(t[k]);
							convert2Num >> n;
							// Select the related operands
							if (k == 0)
								op1 = beta[n];
							else
								op2 = beta[n];
						}
					}
					operation = "(" + op1 + ")OR(" + op2 + ")";
				}
			}
			beta[j] = operation;
		}
		string result = "RULE " + to_string(i) + ":= " + ruleSet[i].getRule() + " ---> " + beta[binOpList.size() - 1] + "THEN(" + conseqInRule[0] + ")";
		outputFile << result << endl << endl;
	}
	outputFile.close();
}
#endif