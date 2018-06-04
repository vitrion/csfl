#ifndef FSEXCEPT_H
#define FSEXCEPT_H

#include <stdexcept>
using std::runtime_error;

#include <string>
using std::string;

class maxNormFault : public runtime_error
{
public:
	maxNormFault() : runtime_error("Fuzzy Set Exception. Normalization values must be in interval [0, 1]") {};
};

class ordinalityFault : public runtime_error
{
public:
	ordinalityFault(const string &name, const string &mf) : runtime_error("Fuzzy Set Exception. Invalid Parameter Condition for set " + name +  ". Parameters must be set in ascending order for " + mf + " Membership Function.") {};
	
};

class paramNumbFault : public runtime_error
{
public:
	paramNumbFault(const string &name, const string &mf, const string &proposed, const string &required) : runtime_error("Fuzzy Set Exception. Invalid Parameter Number for set " + name + " with " + mf + " Membership Function. " + "Proposed: " + proposed + ". Required: " + required + '.') {};
};

class paramNaNFault : public runtime_error
{
public:
	paramNaNFault(const string &name, const string &mf, const string &expr) : runtime_error("Fuzzy Set Exception. Invalid Parameters for set " + name + " with " + mf + " Membership Function. " + "Verify that " + expr) {};
};

class mfShapeFault : public runtime_error
{
public:
	mfShapeFault(const string &name) : runtime_error("Fuzzy Set Exception. " + name + " is not an available Membership Function Shape.") {};
};

class conjOpFault : public runtime_error
{
public:
	conjOpFault(const string &name) : runtime_error("Fuzzy Set Exception. " + name + " is not an available T-Norm for the Conjunction Operation.") {};
};

class disjOpFault : public runtime_error
{
public:
	disjOpFault(const string &name) : runtime_error("Fuzzy Set Exception. " + name + " is not an available S-Norm for the Disjunction Operation.") {};
};

class aggregMethodFault : public runtime_error
{
public:
	aggregMethodFault(const string &name) : runtime_error("Fuzzy Set Exception. " + name + " is not an available S-Norm for the Aggregation Method.") {};
};

class defuzzMethodFault : public runtime_error
{
public:
	defuzzMethodFault(const string &name) : runtime_error("Fuzzy Set Exception. " + name + " is not an available Defuzzification Method.") {};
};

//class fuzzFault : public runtime_error
//{
//public:
//	fuzzFault(const string &name) : runtime_error("Fuzzy Set Exception. You're attempting to fuzzify a set with " + name + " shape.") {};
//};

class defuzzFault : public runtime_error
{
public:
	defuzzFault(const string &name) : runtime_error("Fuzzy Set Exception. You're attempting to defuzzify set " + name + " with an empty co-domain (Membership Values).") {};
};

class sizeMismatchFault : public runtime_error
{
public:
	sizeMismatchFault() : runtime_error("Fuzzy Set Exception. You're attempting to perform a fuzzy operation with discourse size mismatch.") {};
};

class invalidShape4ShiftFault : public runtime_error
{
public:
	invalidShape4ShiftFault() : runtime_error("Fuzzy Set Exception: Unknown shape definition is not valid to shift a fuzzy set") {};
};

#endif