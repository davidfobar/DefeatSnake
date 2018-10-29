#include "NNClass.h"

int NNClass::compute(vector<double> input) {
	inputLayer = MatrixClass<double>({ input });
	
	//feed forward for first layer
	hLayer[0] = ((inputLayer.dot(weight[0]) + bias[0]).applyFunction(sigmoid));
	
	//feed forward for all hidden layers
	if (numHiddenLayers > 1) {
		for (int i = 0; i < numHiddenLayers-1; i++) {
			hLayer[i + 1] = (hLayer[i].dot(weight[i + 1]) + bias[i + 1]).applyFunction(sigmoid);
		}
	}

	//feed forward for last layer
	outputLayer = (hLayer[numHiddenLayers-1].dot(weight[numHiddenLayers]) 
			+ bias[numHiddenLayers]).applyFunction(sigmoid);
	
	//return the node number with the highest activation
	return outputLayer.getGreatest1DIndex();
}

void NNClass::backPropogate(vector<double> expectedOutput) {
	MatrixClass<double> expected({ expectedOutput });

	dJdB[numBiases-1] = (outputLayer - expected) * ( (hLayer[numHiddenLayers-1].dot(weight[numWeights-1])) + bias[numBiases-1] ).applyFunction(sigmoidPrime);
	for (int i = numBiases - 2; i > 0; i--) {
		dJdB[i] = dJdB[i+1].dot(weight[i+1].transpose()) * (hLayer[i-1].dot(weight[i]) + bias[i]).applyFunction(sigmoidPrime);
	}
	dJdB[0] = (dJdB[1].dot(weight[1].transpose())) * (inputLayer.dot(weight[0]) + bias[0]).applyFunction(sigmoidPrime);

	dJdW[numWeights-1] = (hLayer[numHiddenLayers-1].transpose()).dot(dJdB[numBiases-1]);
	for (int i = numWeights-1; i > 1; i--) {
		dJdW[i] = (hLayer[i-1].transpose()).dot(dJdB[i]);
	}
	dJdW[0] = inputLayer.transpose().dot(dJdB[0]);
	

	for (int i = 0; i < numBiases; i++) {
		dJdBc[i] = dJdBc[i] + dJdB[i];
	}
	
	for (int i = 0; i < numWeights; i++) {
		dJdWc[i] = dJdWc[i] + dJdW[i];
	}
}

void NNClass::updateWeightsAndBiases(int miniBatchSize) {
	for (int i = 0; i < numWeights; i++) {
		weight[i] = weight[i] - dJdWc[i] * (learningRate / miniBatchSize);
		dJdWc[i].clear();
	}

	for (int i = 0; i < numBiases; i++) {
		bias[i] = bias[i] - dJdBc[i] * (learningRate / miniBatchSize);
		dJdBc[i].clear();
	}
}

NNClass::NNClass(){}

NNClass::NNClass(int layers, int *nodesInLayer,	double learningRate) {
	this->learningRate = learningRate;

	numWeights = layers - 1;
	numBiases = layers - 1;
	numHiddenLayers = layers - 2;
	numOutputs = nodesInLayer[layers];
	weight	= new MatrixClass<double>[numWeights];
	dJdWc	= new MatrixClass<double>[numWeights];
	dJdW	= new MatrixClass<double>[numWeights];
	bias	= new MatrixClass<double>[numBiases];
	dJdBc	= new MatrixClass<double>[numBiases];
	dJdB	= new MatrixClass<double>[numBiases];
	hLayer	= new MatrixClass<double>[numHiddenLayers];

	for (int i = 0; i < numWeights; i++) {
		weight[i] = MatrixClass<double>(nodesInLayer[i], nodesInLayer[i+1]);
		dJdWc[i] = MatrixClass<double>(nodesInLayer[i], nodesInLayer[i+1]);
		dJdW[i] = MatrixClass<double>(nodesInLayer[i], nodesInLayer[i+1]);
		weight[i] = weight[i].applyFunction(random);
		dJdWc[i].clear();
		dJdW[i].clear();
	}

	for (int i = 0; i < numBiases; i++) {
		bias[i] = MatrixClass<double>(1, nodesInLayer[i+1]);
		dJdBc[i] = MatrixClass<double>(1, nodesInLayer[i+1]);
		dJdB[i] = MatrixClass<double>(1, nodesInLayer[i+1]);
		bias[i] = bias[i].applyFunction(random);
		dJdBc[i].clear();
		dJdB[i].clear();
	}
	
	for (int i = 0; i < numHiddenLayers; i++) {
		hLayer[i] = MatrixClass<double>(1, nodesInLayer[i+1]);
		hLayer[i].clear();
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

NNClass::~NNClass(){}
