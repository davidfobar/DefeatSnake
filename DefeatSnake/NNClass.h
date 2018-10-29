#pragma once

#include "MatrixClass.h"
#include <stdlib.h>
#include <time.h>

class NNClass
{
private:
	double learningRate;
	int numWeights, numBiases;
	int numHiddenLayers;
	int miniBatchSize;
	int numOutputs;
	MatrixClass<double> inputLayer, outputLayer, *dJdB, *dJdW;
	MatrixClass<double> *hLayer, *weight, *bias, *dJdBc, *dJdWc;
public:
	NNClass();
	NNClass(int layers, 
			int *nodesInLayer, 
			double learningRate);
	int compute(vector<double> input);
	void backPropogate(vector<double> expectedOutput);
	void updateWeightsAndBiases(int miniBatchSize);
	~NNClass();
};

double random(double x);
double sigmoid(double x);
double sigmoidPrime(double x);

