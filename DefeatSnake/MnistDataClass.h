#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

class MnistDataClass{
private:
	int numImages = 0;
	int imageRow = 0;
	int imageCol = 0;

	vector< vector<double> > inputData, correctNumber;

	void readHeader(ifstream &file);
public:
	MnistDataClass();
	MnistDataClass(string dataFilename, string answerFilename);
	~MnistDataClass();
	int getNumImages();
	vector<double> getPixelData(int imageNum);
	vector<double> getImageNumber(int imageNum);
};

