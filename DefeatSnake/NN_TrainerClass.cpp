#include "NN_TrainerClass.h"

NN_TrainerClass::NN_TrainerClass(NNClass &inNN) {
	nn = &inNN;

	momentumFactor = 0;
	curEpisodeNum = 0;
	learningRate = DEFAULT_LEARNING_RATE;
	miniBatchSize = DEFAULT_MINIBATCH_SIZE;
	discountFactor = DEFAULT_DISCOUNT_FACTOR;

	dJdW = nn->weight;
	dJdWc = nn->weight;
	prev_dJdWc = dJdWc;

	dJdB = nn->bias;
	dJdBc = nn->bias;
	prev_dJdBc = dJdBc;
}

void NN_TrainerClass::backPropogate(vector<double> expectedOutput) {
	MatrixClass<double> expected({ expectedOutput });

	//for a matrix with 2 hidden layers, i would expect to have 3 bias vectors.
	//I subtract by one due to the zero based system.
	int numB = nn->bias.size() - 1;

	//for numB = 2
	//dJdB[2] =  (nodeLayer[3]      - expected) * ((nodeLayer[2]   .dot(weight[2]))    + bias[2])   .applyFunction(activationTypePrime[3]);
	dJdB[numB] = (nn->nodeLayer[numB + 1] - expected) * ((nn->nodeLayer[numB].dot(nn->weight[numB])) + nn->bias[numB]).applyFunction(nn->activationTypePrime[numB + 1]);

	for (int i = numB - 1; i >= 0; i--) {
		//for numB = 1
		//dJdB[1] = dJdB[2].dot(weight[2].transpose())   * (nodeLayer[1].dot(weight[1])   + bias[1]).applyFunction(activationTypePrime[2]);
		dJdB[i] = dJdB[i + 1].dot(nn->weight[i + 1].transpose()) * (nn->nodeLayer[i].dot(nn->weight[i]) + nn->bias[i]).applyFunction(nn->activationTypePrime[i + 1]);
	}

	for (int i = 0; i < numB; i++) {
		//compute the change in the weight matrix
		dJdW[i] = (nn->nodeLayer[i]).transpose().dot(dJdB[i]);

		//add cummulative adjuestments for future updates to the weights and biases.
		dJdBc[i] = dJdBc[i] + dJdB[i];
		dJdWc[i] = dJdWc[i] + dJdW[i];
	}
}

void NN_TrainerClass::updateWeightsAndBiases() {
	for (int i = 0; i < nn->numLayers - 1; i++) {
		nn->weight[i] = nn->weight[i] - (dJdWc[i] * (learningRate / miniBatchSize) + prev_dJdWc[i] * momentumFactor);
		dJdWc[i].clear();
	}

	for (int i = 0; i < nn->numLayers - 1; i++) {
		nn->bias[i] = nn->bias[i] - (dJdBc[i] * (learningRate / miniBatchSize) + prev_dJdBc[i] * momentumFactor);
		dJdBc[i].clear();
	}

	prev_dJdWc = dJdWc;
	prev_dJdBc = dJdBc;
}

void NN_TrainerClass::setLearningRate(double in) {
	learningRate = in;
}

void NN_TrainerClass::setMiniBatchSize(int in) {
	miniBatchSize = in;
}

void NN_TrainerClass::enableMomentum(double in) {
	momentumFactor = in;
}
