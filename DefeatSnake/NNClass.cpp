#include "NNClass.h"

int NNClass::compute(vector<double> input) {
	nodeLayer[0] = MatrixClass<double>({ input });
	
	for (int i = 1; i < numLayers; i++) {
		nodeLayer[i] = ((nodeLayer[i-1].dot(weight[i-1]) + bias[i-1]).applyFunction(activationType[i-1]));
	}

	//return the output node index with the highest activation
	return nodeLayer[numLayers-1].getGreatest1DIndex();
}

void NNClass::backPropogate(vector<double> expectedOutput) {
	MatrixClass<double> expected({ expectedOutput });

	//for a matrix with 2 hidden layers, i would expect to have 3 bias vectors.
	//I subtract by one due to the zero based system.
	int numB = bias.size()-1;

	//for numB = 2
	//dJdB[2] =  (nodeLayer[3]      - expected) * ((nodeLayer[2]   .dot(weight[2]))    + bias[2])   .applyFunction(activationTypePrime[3]);
	dJdB[numB] = (nodeLayer[numB+1] - expected) * ((nodeLayer[numB].dot(weight[numB])) + bias[numB]).applyFunction(activationTypePrime[numB+1]);

	for (int i = numB-1; i >= 0; i--) {
		//for numB = 1
		//dJdB[1] = dJdB[2].dot(weight[2].transpose())   * (nodeLayer[1].dot(weight[1])   + bias[1]).applyFunction(activationTypePrime[2]);
		dJdB[i] = dJdB[i+1].dot(weight[i+1].transpose()) * (nodeLayer[i].dot(weight[i])   + bias[i]).applyFunction(activationTypePrime[i+1]);
	}

	for (int i = 0; i < numB; i++) {
		//compute the change in the weight matrix
		dJdW[i] = (nodeLayer[i]).transpose().dot(dJdB[i]);

		//add cummulative adjuestments for future updates to the weights and biases.
		dJdBc[i] = dJdBc[i] + dJdB[i];
		dJdWc[i] = dJdWc[i] + dJdW[i];
	}	
}

void NNClass::updateWeightsAndBiases() {
	for (int i = 0; i < numLayers-1; i++) {
		weight[i] = weight[i] - (dJdWc[i] * (learningRate / miniBatchSize) + prev_dJdWc[i] * momentumFactor);
		dJdWc[i].clear();
	}

	for (int i = 0; i < numLayers-1; i++) {
		bias[i] = bias[i] - (dJdBc[i] * (learningRate / miniBatchSize) + prev_dJdBc[i] * momentumFactor);
		dJdBc[i].clear();
	}

	prev_dJdBc = dJdBc;
	prev_dJdWc = dJdWc;
}

NNClass::NNClass(){}

void NNClass::setLearningRate(double in) {
	learningRate = in;
}

void NNClass::setMiniBatchSize(int in) {
	miniBatchSize = in;
}

void NNClass::enableMomentum(double in) {
	momentumFactor = in;
}

NNClass::NNClass(int numInputNodes, int inputNodeType, int numOutputNodes, int outputNodeType) {
	momentumFactor = 0;
	learningRate = DEFAULT_LEARNING_RATE;

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
		dJdWc.push_back(MatrixClass<double>(nodesInLayer[i-1], nodesInLayer[i]));
		dJdW.push_back(MatrixClass<double>(nodesInLayer[i-1], nodesInLayer[i]));

		bias.push_back(MatrixClass<double>(1, nodesInLayer[i]));
		dJdBc.push_back(MatrixClass<double>(1, nodesInLayer[i]));
		dJdB.push_back(MatrixClass<double>(1, nodesInLayer[i]));
		
		weight[i-1] = weight[i-1].applyFunction(random);
		dJdWc[i-1].clear();
		dJdW[i-1].clear();

		bias[i-1] = bias[i-1].applyFunction(random);
		dJdBc[i-1].clear();
		dJdB[i-1].clear();
	}

	prev_dJdBc = dJdBc;
	prev_dJdWc = dJdWc;
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

NNClass::~NNClass(){}
