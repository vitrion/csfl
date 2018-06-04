#ifndef FVAREXCEPT_H
#define FVAREXCEPT_H

#include <stdexcept>
using std::runtime_error;

#include <string>
using std::string;

class mtVarFault : public runtime_error
{
public:
	mtVarFault() : runtime_error("Fuzzy Variable Exception. You are trying to fuzzify a crisp value in a fuzzy variable without sets.") {};
};

class nonIntervalSetFault : public runtime_error
{
public:
	nonIntervalSetFault() : runtime_error("Fuzzy Variable Exception. Trimming operation must be performed only with Interval Sets.") {};
};

class unexistingSetNameFault : public runtime_error
{
public:
	unexistingSetNameFault() : runtime_error("Fuzzy Variable Exception. You are trying to a address an unexisting fuzzy set in current variable.") {};
};

class sizeMismatch4TrimmingFault : public runtime_error
{
public:
	sizeMismatch4TrimmingFault() : runtime_error("Fuzzy Variable Exception. Trimming operation requires the number of sets to be equal to the number of trimming alpha levels or interval sets.") {};
};

class varTypeFault : public runtime_error
{
public:
	varTypeFault() : runtime_error("Fuzzy Variable Exception. Invalid variable type selection. Only possible values: Input or Output") {};
};
#endif