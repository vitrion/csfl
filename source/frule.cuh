#ifndef FRULE_CUH
#define FRULE_CUH

// ********************************************************************************************
// ********************************* LOADING HEADER FILES *************************************
// ********************************************************************************************

#include "fruleExcept.cuh"
#include "flsFriends.cuh"

#include <iostream>
#include <string>
#include <vector>
using namespace std;

#include <sstream>
using std::stringstream;

// ********************************************************************************************
// ******************************** FUZZY RULE CLASS DEFINITION ********************************
// ********************************************************************************************

class frule
{
public:
	// FUZZY RULE: CONSTRUCTORS

	// Constructor 1: Fuzzy rule constructor specifying the rule in a string
	frule(const unsigned int &, const string &);

	// FUZZY RULE: SET FUNCTIONS

	// This function sets an existing rule specifying the rule in a string
	void setRule(const unsigned int &, const string &);
	// This function sets the ID of an existing rule
	void setRuleID(const unsigned int &);

	// FUZZY RULE: GET FUNCTIONS

	// This function gets an existing rule specifying its rule index
	string getRule() const;
	// This functions gets the ID of an existing rule
	unsigned int getRuleID() const;
	// Delivers the premises based on the rule
	vector<string> getPremises() const;
	// Delivers the consequences based on the rule
	vector<string> getConsequences() const;
	// Delivers the operations used in the rule
	vector<string> getOperations() const;

	// This function delivers a set of token divided by the separator strings
	friend vector<string> token(const string &, const vector<string> &, const bool &);

private:
	// FUZZY RULE : PROPERTIES

	// Rule ID
	unsigned int ID;
	// Rule string
	string rule;
	// This is the list of all the premise sets specified in the rule
	vector<string> premises;
	// This is the list of all the consequence sets specified in the rule
	vector<string> consequences;
	// Stores the left part of the inference model
	string left;
	// Stores the right part of the inference model
	string right;
	// Utility index for nesting operation purposes
	size_t last_inner;
	// Utility index for defining the nesting level
	unsigned int level;
	// This is the list of all the operations used in the rule
	vector<string> operations;
	// Available operators for rules
	vector<string> availableOperators;

	// FUZZY RULE: ERROR CHECKING FUNCTIONS

	// This function assures the inference model definition by operator => THEN
	void check4ThenStatement();
	// 
	void check4Parenthesis();
	// 
	void check4RuleOp();

	// FUZZY RULE : UTILITY FUNCTIONS

	//// This function delivers a set of token divided by the separator string
	//vector<string> token(const string &, const vector<string> &, const bool &) const;
	// This function compiles the rule and sets the premise, consequence and rule operations
	void compile();
	// This function searches for all the premise sets related in the rule
	size_t setPremises(const size_t &);
	// This function searches for all the consequence sets related in the rule
	size_t setConsequences(const size_t &);
};

// ********************************************************************************************
// *******************************  FUZZY RULE CLASS METHODS **********************************
// ********************************************************************************************

// FUZZY RULE: CONSTRUCTORS

frule::frule(const unsigned int & identifier, const string &ruleString) {
	ID = identifier;
	rule = ruleString;
	for (unsigned int j = 0; j < rule.size(); j++) {
		if (rule[j] == ' ')
			rule.erase(j, 1);
	}
	operations = {};
	premises = {};
	consequences = {};
	// Set available operators for rules
	availableOperators.push_back("NOT");
	availableOperators.push_back("AND");
	availableOperators.push_back("OR");

	try {
		check4ThenStatement();
	}
	catch (thenFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	compile();
}

// FUZZY RULE: SET FUNCTIONS

void frule::setRule(const unsigned int & identifier, const string &ruleString) {
	ID = identifier;
	rule = ruleString;
	for (unsigned int j = 0; j < rule.size(); j++) {
		if (rule[j] == ' ')
			rule.erase(j, 1);
	}
	operations = {};
	premises = {};
	consequences = {};
	// Set available operators for rules
	availableOperators.push_back("NOT");
	availableOperators.push_back("AND");
	availableOperators.push_back("OR");

	try {
		check4ThenStatement();
	}
	catch (thenFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
	compile();
}

void frule::setRuleID(const unsigned int &identifier) {
	ID = identifier;
}

// FUZZY RULE: GET FUNCTIONS

string frule::getRule() const {
	return rule;
}

unsigned int frule::getRuleID() const {
	return ID;
}

vector<string> frule::getPremises() const {
	return premises;
}

vector<string> frule::getConsequences() const {
	return consequences;
}

vector<string> frule::getOperations() const {
	return operations;
}

// FUZZY RULE: RULE FLAGS

// FUZZY RULE : UTILITY FUNCTIONS

size_t frule::setPremises(const size_t &open) {
	size_t inner_open, inner_close, close;
	if (last_inner < open) {
		inner_open = left.find("(", open + 1);
		close = left.find(")", open + 1);
	}
	else {
		inner_open = left.find("(", last_inner + 1);
		close = left.find(")", last_inner + 1);
	}
	if (close < inner_open) {
		level--;
		last_inner = open;
		operations.push_back(left.substr(open + 1, close - open - 1));
		return close;
	}
	else {
		level++;
		last_inner = setPremises(inner_open);
		if (level != 0)
			inner_close = setPremises(open);
		return inner_close;
	}
}

size_t frule::setConsequences(const size_t &open) {
	size_t inner_open, inner_close, close;
	string str;
	if (last_inner < open) {
		inner_open = right.find("(", open + 1);
		close = right.find(")", open + 1);
	}
	else {
		inner_open = right.find("(", last_inner + 1);
		close = right.find(")", last_inner + 1);
	}
	if (close < inner_open) {
		level--;
		last_inner = open;
		str = right.substr(open + 1, close - open - 1);
		consequences.push_back(str);
		return close;
	}
	else {
		level++;
		last_inner = setConsequences(inner_open);
		if (level != 0)
			inner_close = setConsequences(open);
		return inner_close;
	}
}

void frule::compile() {
	try {
		check4Parenthesis();
	}
	catch (parenthesisFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}

	if (left.size() != 0) {
		// Find the nested operations and operands
		size_t close, open = left.find("(");
		bool cond = (open <= left.size());
		if (cond)
			level = 1;
		else
			level = 0;
		last_inner = open;
		while (cond) {
			close = setPremises(open);
			open = left.find("(", close + 1);
			cond = (open <= left.size());
			if (cond)
				level = 1;
			else
				level = 0;
		}
		operations.push_back(left);

		// Find the related premises in rule
		size_t index;
		vector<size_t> d;
		for (unsigned int i = 0; i < operations.size(); i++) {
			index = operations[i].find_first_of("()");
			if (index > operations[i].size()) {
				premises.push_back(operations[i]);
				d.push_back(i);
			}
		}
		// Reorder the singular operands, i.e. the premises
		for (unsigned int h = 0; h < d.size() - 1; h++) {
			vector<string>::iterator newPos, delPos;
			if (d[h + 1] - d[h] != 1) {
				newPos = operations.begin() + d[h + 1] - d[h];
				operations.insert(newPos, operations[d[h + 1]]);
				delPos = operations.begin() + d[h + 1] + 1;
				operations.erase(delPos);

				d.clear();
				for (unsigned int i = 0; i < operations.size(); i++) {
					index = operations[i].find_first_of("()");
					if (index > operations[i].size()) {
						d.push_back(i);
					}
				}
			}
		}
		// Reduce the operand string reference by numerical indexing
		string str;
		for (unsigned int i = 0; i < premises.size(); i++) {
			stringstream number;
			number << i;
			str = number.str();
			for (unsigned int j = 0; j < operations.size(); j++) {
				index = operations[j].find(premises[i]);
				if (index <= operations[j].size())
					operations[j] = operations[j].replace(index, premises[i].size(), str);
			}
		}
		for (size_t i = premises.size(); i < operations.size(); i++) {
			stringstream number;
			number << i;
			str = number.str();
			for (size_t j = i + 1; j < operations.size(); j++) {
				index = operations[j].find(operations[i]);
				if (index <= operations[j].size())
					operations[j] = operations[j].replace(index, operations[i].size(), str);
			}
		}
		// Find the consequences related in the rule
		open = right.find("(");
		cond = (open <= right.size());
		if (cond)
			level = 1;
		else
			level = 0;
		last_inner = open;
		while (cond) {
			close = setConsequences(open);
			open = right.find("(", close + 1);
			cond = (open <= right.size());
			if (cond)
				level = 1;
			else
				level = 0;
		}

		// Decompose operations to binary operations
		vector<string> t;
		size_t op1Index, op2Index, opIndex, newOpIndex;
		vector<string>::iterator newPos;
		for (unsigned int i = 0; i < operations.size(); i++) {
			bool ovride = false;
			stringstream number;
			number << i;
			str = number.str();
			t = token(operations[i], availableOperators, true);
			if (t.size() > 2) {
				if (t[0].size() != 0) {
					for (unsigned int j = 0; j < t.size() - 1; j++) {
						op1Index = operations[i].find(t[j], 0);
						if (op1Index <= operations[i].size())
							op2Index = operations[i].find(t[j + 1], op1Index);
						else
							op2Index = op1Index;
						for (unsigned int k = 0; k < availableOperators.size(); k++) {
							opIndex = operations[i].find(availableOperators[k], 0);
							if (opIndex <= operations[i].size()) {
								if (opIndex >= op1Index && opIndex <= op2Index) {
									newPos = operations.begin() + i;
									operations.insert(newPos, t[j] + availableOperators[k] + t[j + 1]);
									newOpIndex = operations[i + 1].find(operations[i], 0);
									operations[i + 1].replace(newOpIndex, operations[i].size(), "(" + str + ")");
									ovride = true;
									break;
								}
								else
									ovride = false;
							}
						}
						size_t strStart, strEnd;
						bool sw = false;
						string numStr;
						if (i + 2 < operations.size()) {
							index = operations[i + 2].find_first_of("()");
							while (index <= operations.size()) {
								if (!sw)
									strStart = index + 1;
								else{
									strEnd = index;
									numStr = operations[i + 2].substr(strStart, strEnd - strStart);
									operations[i + 2].erase(strStart, numStr.size());
									stringstream convert2Num(numStr);
									unsigned int n;
									convert2Num >> n;
									if (n >= premises.size()) {
										n++;
										stringstream convert2Str;
										convert2Str << n;
										numStr = convert2Str.str();
									}
									operations[i + 2].insert(strStart, numStr);
								}
								index = operations[i + 2].find_first_of("()", index + 1);
								sw = !sw;
							}
						}
						if (ovride)
							break;
					}
				}
			}
			if (ovride)
				continue;
		}
	}
	else {
		level = 0;
		premises.push_back(left);
		string str = right;
		size_t index = right.find_first_of("()");
		while (index <= right.size()) {
			str.erase(index, 1);
			index = str.find_first_of("()", index + 1);
		}
		consequences.push_back(str);
		operations.push_back(rule);
	}
	try {
		check4RuleOp();
	}
	catch (ruleOpFault &e) {
		cerr << e.what() << endl;
		cin.get();
		exit(1);
	}
}

// ERROR CHECKING FUNCTIONS

void frule::check4ThenStatement() {
	vector<string> fragments;
	fragments = token(rule, { "THEN" }, true);
	if (fragments.size() == 2) {
		left = fragments[0];
		right = fragments[1];
	}
	else {
		fragments = token(rule, { "ELSE" }, true);
		if (fragments.size() == 1)
			throw thenFault();
		else {
			left = "";
			right = fragments[1];
		}
	}
}

void frule::check4Parenthesis() {
	unsigned int u = 0, v = 0;
	size_t index = rule.find("(");
	while (index <= rule.size()) {
		u++;
		index = rule.find("(", index + 1);
	}
	index = rule.find(")");
	while (index <= rule.size()) {
		v++;
		index = rule.find(")", index + 1);
	}
	if (u != v)
		throw parenthesisFault();
}

void frule::check4RuleOp() {
	string L = left;
	for (unsigned int i = 0; i < premises.size(); i++) {
		size_t begin = L.find(premises[i]), s = premises[i].size();
		L.erase(begin, s);
	}

	vector<string> t;
	bool d = false;
	for (unsigned int i = 0; i < availableOperators.size(); i++) {
		t = token(left, { availableOperators[i] }, true);
		d = d | (t.size() != 1);
	}

	if (premises.size() != 1)
		if (!d)
			throw ruleOpFault();
}
#endif