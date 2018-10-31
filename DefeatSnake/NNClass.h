#pragma once

#include "MatrixClass.h"
#include <stdlib.h>
#include <time.h>

const bool AGENT_PLAYING = true;
const bool AGENT_TRAINING = false;
const double DEFAULT_LEARNING_RATE = 3.0;

const int SIGMOID = 0;
const int RELU = 1;

class NNClass
{
private:
	double learningRate;
	double momentumFactor;
	int miniBatchSize;
	int numLayers;
	vector<int> nodesInLayer;
	vector<double(*)(double)> activationType, activationTypePrime;
	vector< MatrixClass<double> > nodeLayer, dJdB, dJdW, weight, bias, dJdBc, dJdWc, prev_dJdBc, prev_dJdWc;
public:
	NNClass();
	int compute(vector<double> input);

	void backPropogate(vector<double> expectedOutput);
	void updateWeightsAndBiases();
	void setLearningRate(double in);
	void setMiniBatchSize(int in);
	~NNClass();

	NNClass(int numInputNodes, int inputNodeType, int numOutputNodes, int outputNodeType);
	void addHiddenLayer(int numNodes, int nodeType);
	void init();
	void enableMomentum(double in);
};

double random(double x);
double sigmoid(double x);
double sigmoidPrime(double x);
double relu(double x);
double reluPrime(double x);

