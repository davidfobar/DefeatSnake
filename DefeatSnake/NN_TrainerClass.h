#pragma once
#include "NNClass.h"

const double DEFAULT_LEARNING_RATE = 3.0;
const double DEFAULT_MINIBATCH_SIZE = 1;
const double DEFAULT_DISCOUNT_FACTOR = 1;

class NN_TrainerClass {
private:
	double learningRate;
	double momentumFactor;
	double discountFactor;
	int miniBatchSize;
	int curEpisodeNum;

	NNClass *nn;
	vector< MatrixClass<double> > dJdB, dJdW, dJdBc, dJdWc, prev_dJdBc, prev_dJdWc;

public:
	NN_TrainerClass() {};
	NN_TrainerClass(NNClass &inNN);
	void backPropogate(vector<double> expectedOutput);
	void updateWeightsAndBiases();
	void setLearningRate(double in);
	void setMiniBatchSize(int in);
	void enableMomentum(double in);
	
};

