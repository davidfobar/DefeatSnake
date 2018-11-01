#include "NNClass.h"

int NNClass::compute(vector<double> input) {

	nodeLayer[0] = MatrixClass<double>({ input });
	
	for (int i = 1; i < numLayers; i++) {
		nodeLayer[i] = ((nodeLayer[i-1].dot(weight[i-1]) + bias[i-1]).applyFunction(activationType[i-1]));
	}

	//return the output node index with the highest activation
	return nodeLayer[numLayers - 1].getGreatest1DIndex();
}

NNClass::NNClass(int numInputNodes, int inputNodeType, int numOutputNodes, int outputNodeType) {
	nodesInLayer.push_back(numInputNodes);
	if (inputNodeType == SIGMOID) {
		activationType.push_back(sigmoid);
		activationTypePrime.push_back(sigmoidPrime);
	}
	else if (inputNodeType == RELU) {
		activationType.push_back(relu);
		activationTypePrime.push_back(reluPrime);
	}

	nodesInLayer.push_back(numOutputNodes);
	if (outputNodeType == SIGMOID) {
		activationType.push_back(sigmoid);
		activationTypePrime.push_back(sigmoidPrime);
	}
	else if (outputNodeType == RELU) {
		activationType.push_back(relu);
		activationTypePrime.push_back(reluPrime);
	}

	numLayers = 2;
}

void NNClass::addHiddenLayer(int numNodes, int nodeType) {
	nodesInLayer.insert(nodesInLayer.end()-1, numNodes);
	if (nodeType == SIGMOID) {
		activationType.insert(activationType.end()-1, sigmoid);
		activationTypePrime.insert(activationTypePrime.end()-1, sigmoidPrime);
	}
	else if (nodeType == RELU) {
		activationType.insert(activationType.end()-1, relu);
		activationTypePrime.insert(activationTypePrime.end()-1, reluPrime);
	}
	numLayers++;
}

void NNClass::init() {
	nodeLayer.push_back(MatrixClass<double>(1, nodesInLayer[0]));
	for (int i = 1; i < numLayers; i++) {
		nodeLayer.push_back(MatrixClass<double>(1, nodesInLayer[i]));

		//dJdB, dJdW, hLayer, weight, bias, dJdBc, dJdWc;
		weight.push_back(MatrixClass<double>(nodesInLayer[i-1], nodesInLayer[i]));
		bias.push_back(MatrixClass<double>(1, nodesInLayer[i]));

		weight[i-1] = weight[i-1].applyFunction(random);
		bias[i-1] = bias[i-1].applyFunction(random);
	}
}

double random(double x) {
	double num = (double)(rand() % 10000 + 1) / 10000 - 0.5;
	return num;
}

double sigmoid(double x) {
	return (1.0 / (1.0 + exp(-x)));
}

double sigmoidPrime(double x) {
	return sigmoid(x)*(1 - sigmoid(x));
}

double relu(double x) {
	if (x > 0) return x;
	else return 0;
}

double reluPrime(double x) {
	if (x > 0) return 1;
	else return 0;
}