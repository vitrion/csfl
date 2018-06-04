#ifndef FRULEEXCEPT_H
#define FRULEEXCEPT_H

#include <stdexcept>
using std::runtime_error;

#include <string>
using std::string;

class thenFault : public runtime_error
{
public:
	thenFault() : runtime_error("Fuzzy Rule Exception: Rule must have a THEN statement. Try writting the THEN statement in CAPS.") {};
};

class parenthesisFault : public runtime_error
{
public:
	parenthesisFault() : runtime_error("Fuzzy Rule Exception: Expected parenthesis closure.") {};
};

class ruleOpFault : public runtime_error
{
public:
	ruleOpFault() : runtime_error("Fuzzy Rule Exception: Unexpected operator. Possible operators are: NOT, AND and OR. Try writting the operator statement in CAPS") {};
};
#endif