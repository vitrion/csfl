#ifndef FLSEXCEPT_H
#define FLSEXCEPT_H

#include <stdexcept>
using std::runtime_error;

#include <string>
using std::string;

class elseStatementFault : public runtime_error
{
public:
	elseStatementFault() : runtime_error("Fuzzy System Exception: An ELSE statement has been previously defined. If you need to adding more rules, please eliminate the ELSE statement") {};
};

class ruleCountFault : public runtime_error
{
public:
	ruleCountFault() : runtime_error("Fuzzy System Exception: Rule count exceeds the maximum rule number available with the current configuration.") {};
};

class MTDelRuleFault : public runtime_error
{
public:
	MTDelRuleFault() : runtime_error("Fuzzy System Exception: You are trying to delete a rule in an empty rule set.") {};
};

class unexistRuleFault : public runtime_error
{
public:
	unexistRuleFault() : runtime_error("Fuzzy System Exception: You are trying to delete an unexisting rule in the rule set.") {};
};

class unexistingVarNameFault : public runtime_error
{
public:
	unexistingVarNameFault() : runtime_error("Fuzzy System Exception: You are trying to access an unexisting variable.") {};
};

class invalidSigmaCountFault : public runtime_error
{
public:
	invalidSigmaCountFault() : runtime_error("Fuzzy System Exception: The non-singleton fuzzification requires that the number of sigma values are equal to the number of input variables.") {};
};

class invalidStreamEnableFault : public runtime_error
{
public:
	invalidStreamEnableFault() : runtime_error("Fuzzy System Exception: Streams are available only when flag isCUDA is enabled.") {};
};
#endif