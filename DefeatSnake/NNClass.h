#pragma once

#include "MatrixClass.h"
//#include <stdlib.h>
//#include <time.h>

const int SIGMOID = 0;
const int RELU = 1;

class NNClass {
	friend class NN_TrainerClass;
private:
	int numLayers;
	vector< int > nodesInLayer;
	vector< double(*)(double) > activationType, activationTypePrime;
	vector< MatrixClass<double> > nodeLayer, weight, bias;

public:
	NNClass() {};
	NNClass(int numInputNodes, int inputNodeType, int numOutputNodes, int outputNodeType);
	void addHiddenLayer(int numNodes, int nodeType);
	void init();

	int compute(vector<double> input);
};

double random(double x);
double sigmoid(double x);
double sigmoidPrime(double x);
double relu(double x);
double reluPrime(double x);

