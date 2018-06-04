#ifndef FVAR_CUH
#define FVAR_CUH

// ********************************************************************************************
// ********************************* LOADING HEADER FILES *************************************
// ********************************************************************************************

#include "fs.cuh"
#include "fvarExcept.cuh"
#include "nvToolsExt.h"

// ********************************************************************************************
// ******************************** FUZZY VARIABLE CLASS DEFINITION ********************************
// ********************************************************************************************

class fvar
{
public:
	// FUZZY VARIABLE CONSTRUCTORS

	// Constructor 0: Default Fuzzy Variable Constructor (type: input, name: "Undefined", range: [0, 1])
	fvar(); // OK

	// Constructor 1: Fuzzy Variable Constructor that specifies the processing type
	fvar(const bool &); // OK

	// Constructor 2: Fuzzy Variable Constructor that specifies the variable name
	fvar(const string &); // OK

	// Constructor 3: Fuzzy Variable Constructor that specifies the variable name and its processing type
	fvar(const string &, const bool &); // OK

	// Constructor 4: Fuzzy Variable Constructor that specifies the variable type and name
	fvar(const string &, const string &); // OK

	// Constructor 5: Fuzzy Variable Constructor that specifies the variable type, name and its processing type
	fvar(const string &, const string &, const bool &); // OK

	// Constructor 6: Fuzzy Variable Constructor that specifies the variable name and range
	fvar(const string &, const float &, const float &); // OK

	// Constructor 7: Fuzzy Variable Constructor that specifies the variable name, range and its processing type
	fvar(const string &, const float &, const float &, const bool &); // OK

	// Constructor 8: Fuzzy Variable Constructor that specifies the variable type, name and range
	fvar(const string &, const string &, const float &, const float &); // OK

	// Constructor 9: Fuzzy Variable Constructor that specifies the variable type, name, range and its processing type
	fvar(const string &, const string &, const float &, const float &, const bool &); // OK

	// Constructor 10: Fuzzy Variable Constructor that specifies the variable ID, type, name, range and its processing type
	fvar(const unsigned int &, const string &, const string &, const float &, const float &, const bool &); // OK

	// Constructor 11: Fuzzy Variable Constructor that specifies the variable ID, type, name, range, processing type and streams
	fvar(const unsigned int &, const string &, const string &, const float &, const float &, const bool &, const bool &); // OK

	// Constructor 12: Fuzzy Variable Constructor that specifies the processing type
	fvar(const bool &, const bool &); // OK

	// MANAGEMENT FUNCTIONS

	// Evaluates the membership degrees in all the sets contained in the variable
	vector<fs> fuzzify(fs); // OK
	// Determinates the crisp value in a single or in all the sets contained in the variable
	vector<float> defuzzify(); // OK

	// Adds a new set into the current variable
	void addFuzzySet(const string &, const string &, const vector<float> &, const float &); // OK
	// Adds an existing set into the current variable
	void addFuzzySet(fs &); // OK
	// Deletes a set from the current variable using the fuzzy set name
	void delFuzzySet(const string &); // OK
	// Deletes several sets from the current variable using several fuzzy set names
	void delFuzzySet(const vector<string> &); // OK
	// Deletes all the sets from the current variable
	void delFuzzySet(); // OK
	// Assigns a fuzzy variable to another
	const fvar &operator=(const fvar &); // OK

	// SET FUNCTIONS:

	//void not();
	// Modifies the type of variable: input or output
	void setType(const string &); // OK
	// Modifies the ID of the current variable
	void setVarID(const unsigned int &); // OK
	// Modifies the processing type of the variable
	void setHetProc(const bool &); // OK
	// Modifies the Stream processing of the variable
	void setStream(const bool &); // OK
	// Modifies the variable name
	void setName(const string &); // OK
	// Modifies the variable range
	void setRange(const float &, const float &); // OK
	// Modifies the conjunction operation used for each variable set
	void setConjOp(const string &); // OK
	// Modifies the disjunction operation used for each variable set
	void setDisjOp(const string &); // OK
	// Modifies the aggregation method used for each variable set
	void setAggregMethod(const string &); // OK
	// Modifies the defuzzification method used for each variable set
	void setDefuzzMethod(const string &); // OK
	// Modifies an existing set in the variable

	// Modifies the name of an existing fuzzy set in the variable
	void setFuzzySetName(const string &, const string &); // OK
	// Modifies the shape of an existing fuzzy set in the variable
	void setFuzzySetShape(const string &, const string &); // OK
	// Modifies the membership function parameters of an existing fuzzy set in the variable
	void setFuzzySetParams(const string &, const vector<float> &); // OK
	// Modifies the normalization value of an existing fuzzy set in the variable
	void setFuzzysetNorm(const string &, const float &); // OK
	// Modifies the parameters of an existing fuzzy set in the variable
	void setFuzzySet(const string &, const string &, const string &, const vector<float> &, const float &); // OK
	// Replaces an existing fuzzy set in an existing variable
	void replaceFuzzySet(const unsigned int &, const fs &);

	// GET FUNCTIONS:

	// Delivers the type of variable: input or output
	string getType() const; // OK
	// Delivers the current variable ID
	unsigned int getVarID() const; // OK
	// Delivers the count of the sets in the variable
	size_t getSetCount() const;
	// Delivers the processing type of the variable
	bool getProcType() const;
	// Delivers the variable name
	string getName() const; // OK
	// Delivers the variable range
	vector<float> getRange() const; // OK
	// Delivers an existing set in the variable by name
	fs getFuzzySet(const string &) const; // OK
	// Delivers an existing set in the variable by index
	fs getFuzzySet(const unsigned int &) const;
	// Delivers all the existing fuzzy sets in the variable
	vector<fs> getFuzzySet() const;
	// Delivers all the existing fuzzy sets names in the variable
	vector<string> getFuzzySetNames() const;
	// Deliver a pointer to a fuzzy set searching by name
	fs *getFSPtr(const string &);
	// Delivers the fuzzy conjunction operations used for each variable set
	string getConjOp() const; // OK
	// Delivers the fuzzy disjunction operations used for each variable set
	string getDisjOp() const; // OK
	// Delivers the aggregation method used for each variable set
	string getAggregMethod() const; // OK
	// Delivers the defuzzification method used for each variable set
	string getDefuzzMethod() const; // OK

	// MERGING OPERATIONS:

	// Intersects all the sets in variable creating a new set
	void intersectionMerging(); // OK
	// Joins all the sets in variable creating a new set
	void unionMerging(); // OK
	// Sums all the sets in variable creating a new set
	void sumMerging(); // OK
	// Subtracts all the sets in variable creating a new set
	void subMerging(); // OK
	// Multiplies all the sets in variable creating a new set
	void mulMerging(); // OK
	// Divides all the sets in variable creating a new set
	void divMerging(); // OK
	// Modulates all the sets in variable creating a new set
	void modMerging(); // OK

	// TRIMMING FUNCTIONS:

	// Trim a set of sets in variable with an alpha level set
	void trimSets(const vector<float> &); // OK
	// Trim a set of sets in variable with an interval or a singleton set
	void trimSets(const vector<fs> &); // OK
	// Variable fuzzy sets
	vector<fs> sets; // OK

private:
	// Variable type: input or output
	string type; // OK
	// Variable identifier
	unsigned int ID;
	// Variable processing type
	bool isCUDA;
	// Streaming processing type
	bool isStream;
	// Variable name
	string name; // OK
	// Variable range
	vector<float> range; // OK
	// Number of sets in the variable
	size_t setCount; // OK

	// Conjunction operator
	string conjOperator; // OK
	// Disjunction operator
	string disjOperator; // OK
	// Aggregation method
	string aggregMethod; // OK
	// Defuzzification method
	string defuzzMethod; // OK

	// ERROR CHECKING FUNCTIONS

	// Prevents the fuzzification or the defuzzification of a variable without sets
	void check4MTVars(); // OK
	// Prevents to address an unexisting set in the current variable
	unsigned int check4ExistingSetName(const string &) const; // OK
	// Prevents to perform the trimming operation with different alpha set number
	void check4SizeMismatch4Trimming(const vector<float> &); // OK
	// Prevents to perform the trimming operation with different interval set number
	void check4SizeMismatch4Trimming(const vector<fs> &); // OK
	// Prevents the use of any other membership function than the interval set for the trimming operation
	void check4IntervalSet4Trimming(const vector<fs> &); // OK
	// Prevents to assign a different variable type than the existing (input ot output)
	void check4VarType(const string &); // OK
};

// ********************************************************************************************
// ********************************  FUZZY VARIABLE CLASS METHODS **********************************
// ********************************************************************************************

// FUZZY VARIABLE: CONSTRUCTORS

// Constructor 0
fvar::fvar() {
	type = "Input";
	ID = 0;
	name = "Undefined";
	range = {0.0, 1.0};
	sets = {};
	setConjOp("Minimum");
	setDisjOp("Maximum");
	setAggregMethod("Maximum");
	setDefuzzMethod("Centroid");
	isCUDA = false;
	isStream = false;
	setCount = 0;
}

// Constructor 1
fvar::fvar(const bool &proc) {
	type = "Input";
	ID = 0;
	name = "Undefined";
	range = { 0.0, 1.0 };
	sets = {};
	setConjOp("Minimum");
	setDisjOp("Maximum");
	setAggregMethod("Maximum");
	setDefuzzMethod("Centroid");
	isCUDA = proc;
	isStream = false;
	setCount = 0;
}

// Constructor 2
fvar::fvar(const string &varName) {
	type = "Input";
	ID = 0;
	range = { 0.0, 1.0 };
	sets = {};
	name = varName;
	setConjOp("Minimum");
	setDisjOp("Maximum");
	setAggregMethod("Maximum");
	setDefuzzMethod("Centroid");
	isCUDA = false;
	isStream = false;
	setCount = 0;
}

// Constructor 3
fvar::fvar(const string &varName, const bool &proc) {
	type = "Input";
	ID = 0;
	range = { 0.0, 1.0 };
	sets = {};
	name = varName;
	setConjOp("Minimum");
	setDisjOp("Maximum");
	setAggregMethod("Maximum");
	setDefuzzMethod("Centroid");
	isCUDA = proc;
	isStream = false;
	setCount = 0;
}

// Constructor 4
fvar::fvar(const string &varType, const string &varName) {
	try {
		check4VarType(varType);
		type = varType;
	}
	catch (varTypeFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	ID = 0;
	name = varName;
	range = { 0.0, 1.0 };
	sets = {};
	setConjOp("Minimum");
	setDisjOp("Maximum");
	setAggregMethod("Maximum");
	setDefuzzMethod("Centroid");
	isCUDA = false;
	isStream = false;
	setCount = 0;
}

// Constructor 5
fvar::fvar(const string &varType, const string &varName, const bool &proc) {
	try {
		check4VarType(varType);
		type = varType;
	}
	catch (varTypeFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	ID = 0;
	name = varName;
	range = { 0.0, 1.0 };
	sets = {};
	setConjOp("Minimum");
	setDisjOp("Maximum");
	setAggregMethod("Maximum");
	setDefuzzMethod("Centroid");
	isCUDA = proc;
	isStream = false;
	setCount = 0;
}

// Constructor 6
fvar::fvar(const string &varName, const float &lower, const float &upper) {
	type = "Input";
	ID = 0;
	name = varName;
	range = { lower, upper };
	sets = {};
	setConjOp("Minimum");
	setDisjOp("Maximum");
	setAggregMethod("Maximum");
	setDefuzzMethod("Centroid");
	isCUDA = false;
	isStream = false;
	setCount = 0;
}

// Constructor 7
fvar::fvar(const string &varName, const float &lower, const float &upper, const bool &proc) {
	type = "Input";
	ID = 0;
	name = varName;
	range = { lower, upper };
	sets = {};
	setConjOp("Minimum");
	setDisjOp("Maximum");
	setAggregMethod("Maximum");
	setDefuzzMethod("Centroid");
	isCUDA = proc;
	isStream = false;
	setCount = 0;
}

// Constructor 8
fvar::fvar(const string &varType, const string &varName, const float &lower, const float &upper) {
	try {
		check4VarType(varType);
		type = varType;
	}
	catch (varTypeFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	ID = 0;
	name = varName;
	range = { lower, upper };
	sets = {};
	setConjOp("Minimum");
	setDisjOp("Maximum");
	setAggregMethod("Maximum");
	setDefuzzMethod("Centroid");
	isCUDA = false;
	isStream = false;
	setCount = 0;
}

// Constructor 9
fvar::fvar(const string &varType, const string &varName, const float &lower, const float &upper, const bool &proc) {
	try {
		check4VarType(varType);
		type = varType;
	}
	catch (varTypeFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	ID = 0;
	name = varName;
	range = { lower, upper };
	sets = {};
	setConjOp("Minimum");
	setDisjOp("Maximum");
	setAggregMethod("Maximum");
	setDefuzzMethod("Centroid");
	isCUDA = proc;
	isStream = false;
	setCount = 0;
}

// Constructor 10
fvar::fvar(const unsigned int &varID, const string &varType, const string &varName, const float &lower, const float &upper, const bool &proc) {
	try {
		check4VarType(varType);
		type = varType;
	}
	catch (varTypeFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	ID = varID;
	name = varName;
	range = { lower, upper };
	sets = {};
	setConjOp("Minimum");
	setDisjOp("Maximum");
	setAggregMethod("Maximum");
	setDefuzzMethod("Centroid");
	isCUDA = proc;
	isStream = false;
	setCount = 0;
}

// Constructor 11
fvar::fvar(const unsigned int &varID, const string &varType, const string &varName, const float &lower, const float &upper, const bool &proc, const bool &s) {
	try {
		check4VarType(varType);
		type = varType;
	}
	catch (varTypeFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	ID = varID;
	name = varName;
	range = { lower, upper };
	sets = {};
	setConjOp("Minimum");
	setDisjOp("Maximum");
	setAggregMethod("Maximum");
	setDefuzzMethod("Centroid");
	isCUDA = proc;
	setCount = 0;
	isStream = s;
}

// Constructor 1
fvar::fvar(const bool &proc, const bool &s) {
	type = "Input";
	ID = 0;
	name = "Undefined";
	range = { 0.0, 1.0 };
	sets = {};
	setConjOp("Minimum");
	setDisjOp("Maximum");
	setAggregMethod("Maximum");
	setDefuzzMethod("Centroid");
	isCUDA = proc;
	isStream = s;
	setCount = 0;
}

// FUZZY SET: ERROR CHECKING FUNTIONS
void fvar::check4MTVars() {
	if (sets.size() == 0)
		throw mtVarFault();
}

unsigned int fvar::check4ExistingSetName(const string &name) const {
	unsigned int result;
	for (unsigned int i = 0; i < sets.size(); i++)
		if (sets[i].getName().compare(name) == 0) {
			result = i;
			break;
		}
		else
			if (i == (sets.size() - 1))
				throw unexistingSetNameFault();
	return result;
}

void fvar::check4SizeMismatch4Trimming(const vector<float> &alphaLevels) {
	 if (alphaLevels.size() != sets.size())
		 throw sizeMismatch4TrimmingFault();
}

void fvar::check4SizeMismatch4Trimming(const vector<fs> &intervalSets) {
	if (intervalSets.size() != sets.size())
		throw sizeMismatch4TrimmingFault();
}

void fvar::check4IntervalSet4Trimming(const vector<fs> &intervalSets) {
	// for (unsigned int i = 0; i < intervalSets.size(); i++)
		// if (intervalSets[i].getShape().compare("Interval") != 0)
			// throw nonIntervalSetFault();
}

void fvar::check4VarType(const string &varType) {
	if (!(varType.compare("Input") == 0 || varType.compare("Output") == 0))
		throw varTypeFault();
}

// FUZZY VARIABLE: VARIABLE MANAGEMENT

vector<fs> fvar::fuzzify(fs in_crisp) {
	try {
		check4MTVars();
	}
	catch (mtVarFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	//nvtxRangeId_t t02 = nvtxRangeStartEx(&events[2]);
	vector<fs> result(sets.size(), fs(isCUDA, isStream));
	fs b(isCUDA, isStream);
	float offset = ((range[0] + range[1]) / 2) - in_crisp.getParameters()[0];
	in_crisp.setOffset(offset);
	for (unsigned int i = 0; i < sets.size(); i++) {		
		sets[i].setOffset(offset);
		result[i] = in_crisp && sets[i];
		result[i].setRange(-0.5, 0.5, samples);
		sets[i].setOffset(-offset);
	}
	//nvtxRangeEnd(t02);
	return result;
}

vector<float> fvar::defuzzify() {
	try {
		check4MTVars();
	}
	catch (mtVarFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	vector<float> result(sets.size());
	for (unsigned int i = 0; i < sets.size(); i++) {
		result[i] = sets[i].defuzzify();
	}
	return result;
}

void fvar::addFuzzySet(const string &name, const string &shape, const vector<float> &params, const float &normalization) {
	fs s(range[0], range[1], normalization, name, shape, params, isCUDA, isStream);
	sets.push_back(s);
	setCount++;
}
void fvar::addFuzzySet(fs &inputSet) {
	inputSet.setRange(range[0], range[1]);
	inputSet.isCUDA = isCUDA;
	sets.push_back(inputSet);
	setCount++;
}

void fvar::delFuzzySet(const string &name) {
	unsigned int index;
	try {
		index = check4ExistingSetName(name);
		sets.erase(sets.begin() + index);
		setCount--;
	}
	catch (unexistingSetNameFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}

void fvar::delFuzzySet(const vector<string> &names2Del) {
	for (unsigned int i = 0; i < names2Del.size(); i++){
		delFuzzySet(names2Del[i]);
		setCount--;
	}
}

void fvar::delFuzzySet() {
	sets = {};
	setCount = 0;
}

const fvar &fvar::operator=(const fvar &right) {// ASSIGNMENT
	if (&right != this) {
		this->type = right.type;
		this->name = right.name;
		this->range = right.range;
		this->sets = right.sets;
		this->conjOperator = right.conjOperator;
		this->disjOperator = right.disjOperator;
		this->aggregMethod = right.aggregMethod;
		this->defuzzMethod = right.defuzzMethod;
	}
	return *this;
}

// FUZZY VARIABLE: SET FUNCTIONS

void fvar::setType(const string &varType) {
	try {
		check4VarType(varType);
		type = varType;
	}
	catch (varTypeFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}
void fvar::setVarID(const unsigned int &varID) {
	ID = varID;
}

void fvar::setHetProc(const bool &proc) {
	isCUDA = proc;
	for (unsigned int i = 0; i < sets.size(); i++)
		sets[i].isCUDA = proc;
}

void fvar::setStream(const bool &s) {
	isStream = s;
	for (unsigned int i = 0; i < sets.size(); i++)
		sets[i].isStream = s;
}

void fvar::setName(const string &varName) {
	name = varName;
}

void fvar::setRange(const float &lower, const float &upper) {
	range = { lower, upper };
	for (unsigned int i = 0; i < sets.size(); i++) {
		sets[i].setRange(lower, upper);
		sets[i].fuzzify();
	}
}

void fvar::setFuzzySetName(const string &setName, const string &proposedName) {
	unsigned int index;
	try {
		index = check4ExistingSetName(setName);
		sets[index].setName(proposedName);
	}
	catch (unexistingSetNameFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}

void fvar::setFuzzySetShape(const string &setName, const string &proposedShape) {
	unsigned int index;
	try {
		index = check4ExistingSetName(setName);
		sets[index].setShape(proposedShape);
		sets[index].fuzzify();
	}
	catch (unexistingSetNameFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}

void fvar::setFuzzySetParams(const string &setName, const vector<float> &proposedParameters) {
	unsigned int index;
	try {
		index = check4ExistingSetName(setName);
		sets[index].setParams(proposedParameters);
		sets[index].fuzzify();
	}
	catch (unexistingSetNameFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}

void fvar::setFuzzysetNorm(const string &setName, const float &proposedNormalization) {
	unsigned int index;
	try {
		index = check4ExistingSetName(setName);
		sets[index].setNorm(proposedNormalization);
		sets[index].fuzzify();
	}
	catch (unexistingSetNameFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}

void fvar::setConjOp(const string &op) {
	conjOperator = op;
	for (unsigned int i = 0; i < sets.size(); i++)
		sets[i].setConjOp(op);
}

void fvar::setDisjOp(const string &op) {
	disjOperator = op;
	for (unsigned int i = 0; i < sets.size(); i++)
		sets[i].setDisjOp(op);
}

void fvar::setAggregMethod(const string &agg) {
	aggregMethod = agg;
	for (unsigned int i = 0; i < sets.size(); i++)
		sets[i].setAggregMethod(agg);
}

void fvar::setDefuzzMethod(const string &def) {
	defuzzMethod = def;
	for (unsigned int i = 0; i < sets.size(); i++)
		sets[i].setDefuzzMethod(def);
}

void fvar::setFuzzySet(const string &setName, const string &proposedName, const string &proposedShape,
	const vector<float> &proposedParameters, const float &proposedNormalization) {
	unsigned int index;
	try {
		index = check4ExistingSetName(setName);
		sets[index].setName(proposedName);
		sets[index].setShape(proposedShape);
		sets[index].setParams(proposedParameters);
		sets[index].setNorm(proposedNormalization);
		sets[index].fuzzify();
	}
	catch (unexistingSetNameFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}

void fvar::replaceFuzzySet(const unsigned int &index, const fs &newSet) {
	if (index <= setCount)
	{
		sets[index] = newSet;
	}
	else {
		// **************** EXCEPTION
	}
}

// FUZZY VARIABLE: GET FUNCTIONS

string fvar::getType() const {
	return type;
}

unsigned int fvar::getVarID() const {
	return ID;
}

size_t fvar::getSetCount() const {
	return setCount;
}

bool fvar::getProcType() const {
	return isCUDA;
}

string fvar::getName() const {
	return name;
}

vector<float> fvar::getRange() const {
	return range;
}

fs fvar::getFuzzySet(const string &setName) const {
	unsigned int index;
	fs result(isCUDA, isStream);
	try {
		index = check4ExistingSetName(setName);
		result = sets[index];
	}
	catch (unexistingSetNameFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	return result;
}

fs fvar::getFuzzySet(const unsigned int &index) const{
	fs result(isCUDA, isStream);
	try {
		if (index <= setCount)
			result = sets[index];
		else {
			throw unexistingSetNameFault();
		}
	}
	catch (unexistingSetNameFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	return result;
}

vector<fs> fvar::getFuzzySet() const {
	return sets;
}

vector<string> fvar::getFuzzySetNames() const {
	vector<string> result(setCount);
	for (unsigned int i = 0; i < setCount; i++)
		result[i] = sets[i].getName();
	return result;
}

fs *fvar::getFSPtr(const string &fsName) {
	fs *setPtr = NULL;
	for (unsigned int i = 0; i < setCount; i++)
		if (fsName.compare(sets[i].getName()) == 0) {
			setPtr = &sets[i];
			break;
		}
	if (setPtr == NULL)
		throw unexistingSetNameFault();
	return setPtr;
}

string fvar::getConjOp() const {
	return conjOperator;
}

string fvar::getDisjOp() const {
	return disjOperator;
}

string fvar::getAggregMethod() const {
	return aggregMethod;
}

string fvar::getDefuzzMethod() const {
	return defuzzMethod;
}

// FUZZY VARIABLE: MERGING FUNCTIONS

void fvar::intersectionMerging() {
	vector<fs> result(1, fs(isCUDA, isStream));
	try {
		check4MTVars();
	}
	catch (mtVarFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	result[0] = sets[0];
	for (unsigned int i = 1; i < sets.size(); i++)
		result[0] = result[0] && sets[i];
	sets = result;
	setCount = sets.size();
}

void fvar::unionMerging() {
	vector<fs> result;
	try {
		check4MTVars();
	}
	catch (mtVarFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	result.push_back(sets[0]);
	for (unsigned int i = 1; i < sets.size(); i++) {
		//nvtxRangeId_t t13 = nvtxRangeStartEx(&events[13]);
		result[0] = result[0] || sets[i];
		//nvtxRangeEnd(t13);
	}
	sets = result;
	setCount = sets.size();
}

void fvar::sumMerging() {
	vector<fs> result(1, fs(isCUDA, isStream));
	try {
		check4MTVars();
	}
	catch (mtVarFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	result[0] = sets[0];
	for (unsigned int i = 1; i < sets.size(); i++)
		result[0] = result[0] + sets[i];
	range = result[0].getRange();
	sets = result;
	setCount = sets.size();
}

void fvar::subMerging() {
	vector<fs> result(1, fs(isCUDA, isStream));
	try {
		check4MTVars();
	}
	catch (mtVarFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	result[0] = sets[0];
	for (unsigned int i = 1; i < sets.size(); i++)
		result[0] = result[0] - sets[i];
	range = result[0].getRange();
	sets = result;
	setCount = sets.size();
}

void fvar::mulMerging() {
	vector<fs> result(1, fs(isCUDA, isStream));
	try {
		check4MTVars();
	}
	catch (mtVarFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	result[0] = sets[0];
	for (unsigned int i = 1; i < sets.size(); i++)
		result[0] = result[0] * sets[i];
	range = result[0].getRange();
	sets = result;
	setCount = sets.size();
}

void fvar::divMerging() {
	vector<fs> result(1, fs(isCUDA, isStream));
	try {
		check4MTVars();
	}
	catch (mtVarFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	result[0] = sets[0];
	for (unsigned int i = 1; i < sets.size(); i++)
		result[0] = result[0] / sets[i];
	range = result[0].getRange();
	sets = result;
	setCount = sets.size();
}

void fvar::modMerging() {
	vector<fs> result(1, fs(isCUDA, isStream));
	try {
		check4MTVars();
	}
	catch (mtVarFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	result[0] = sets[0];
	for (unsigned int i = 1; i < sets.size(); i++)
		result[0] = result[0] % sets[i];
	range = result[0].getRange();
	sets = result;
	setCount = sets.size();
}

// FUZZY VARIABLE: TRIMMING FUNCTIONS

void fvar::trimSets(const vector<float> &alphaLevels) {
	try {
		check4MTVars();
	}
	catch (mtVarFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	try {
		check4SizeMismatch4Trimming(alphaLevels);
	}
	catch (sizeMismatch4TrimmingFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	for (unsigned int i = 0; i < sets.size(); i++) {
		//nvtxRangeId_t t12 = nvtxRangeStartEx(&events[12]);
		sets[i] = sets[i] && alphaLevels[i];
		//nvtxRangeEnd(t12);
	}
}

void fvar::trimSets(const vector<fs> &intervalSets) {
	try {
		check4MTVars();
	}
	catch (mtVarFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	try {
		check4SizeMismatch4Trimming(intervalSets);
	}
	catch (sizeMismatch4TrimmingFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	try {
		check4IntervalSet4Trimming(intervalSets);
	}
	catch (nonIntervalSetFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	for (unsigned int i = 0; i < sets.size(); i++) {
		//nvtxRangeId_t t12 = nvtxRangeStartEx(&events[12]);
		sets[i] = sets[i] && intervalSets[i];
		//nvtxRangeEnd(t12);
		//sets[i].setNorm(intervalSets[i].getNormalization());
	}
}

#endif
