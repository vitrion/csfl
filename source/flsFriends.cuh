#ifndef FLSFRIENDS_CUH
#define FLSFRIENDS_CUH

#include <string>
#include <iostream>
#include <vector>
using namespace std;

vector<float> linearSpace(const float &lower, const float &upper, const unsigned int N) {
	vector<float> result(N);
	float dx = (upper - lower) / (N - 1);
	float summation = lower;
	for (unsigned int i = 0; i < N; i++) {
		result[i] = summation;
		summation += dx;
	}
	return result;
}

vector<string> token(const string &str2sep, const vector<string> &separator, const bool &elimEndSpaces) {
	vector<string> result;
	vector<size_t> found;
	vector<size_t> sepSize;
	size_t sep;
	for (size_t i = 0; i < separator.size(); i++) {
		sep = str2sep.find(separator[i]);
		while (sep <= str2sep.size()) {
			found.push_back(sep);
			sepSize.push_back(separator[i].size());
			sep = str2sep.find(separator[i], sep + 1);
		}
	}
	if (found.size() > 1) {
		bool ordered = false;
		vector<bool> a(found.size() - 1);
		for (unsigned int i = 0; i < found.size() - 1; i++)
			a[i] = (found[i] < found[i + 1]);
		ordered = a[0];
		for (unsigned int i = 1; i < a.size(); i++)
			ordered = ordered & a[i];
		while (!ordered) {
			for (unsigned int i = 0; i < found.size() - 1; i++) {
				size_t b = found[i];
				size_t c = sepSize[i];
				if (found[i] > found[i + 1]) {
					found[i] = found[i + 1];
					found[i + 1] = b;
					sepSize[i] = sepSize[i + 1];
					sepSize[i + 1] = c;
				}
			}
			for (unsigned int i = 0; i < found.size() - 1; i++)
				a[i] = (found[i] < found[i + 1]);
			ordered = a[0];
			for (unsigned int i = 1; i < a.size(); i++)
				ordered = ordered & a[i];

		}
	}
	/*else if (found.size() == 1) {

	}*/
	for (unsigned int i = 0; i < found.size(); i++) {
		if (i == 0)
			if (found[i] >= 0)
				if (found[i] == 0)
					result.push_back("");
				else
					result.push_back(str2sep.substr(0, found[i]));
			else
				result.push_back("");
		else
			result.push_back(str2sep.substr(found[i - 1] + sepSize[i - 1], found[i] - found[i - 1] - sepSize[i - 1]));
	}
	if (found.size() > 0)
		//if (found[0] > 0)
			result.push_back(str2sep.substr(found[found.size() - 1] + sepSize[found.size() - 1], str2sep.size()));
		/*else {
			if (found[0] > 0)
			result.push_back(str2sep);
		}*/
	else
		result = { str2sep };

	if (elimEndSpaces && (found.size() != 0)){
		size_t k;
		for (size_t i = 0; i < result.size(); i++) {
			for (size_t j = 0; j < result[i].size(); j++)
				if (result[i][j] == ' ')
					result[i].erase(j, 1);
				else
					break;
			for (size_t j = 0; j < result[i].size(); j++) {
				k = result[i].size() - j - 1;
				if (result[i][k] == ' ')
					result[i].erase(k, 1);
				else
					break;
			}
		}
	}
	return result;
}

#endif