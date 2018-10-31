#include "MnistDataClass.h"
#include "NNClass.h"
#include <iostream>
#include <time.h>
#include <stdlib.h>

using namespace std;

const double LEARNING_RATE = 2.5;
const double MOMENTUM_FACTOR = 0.25;
const int MINI_BATCH_SIZE = 50;
const int NUM_EPOCHS = 2;

int main(){
	cout << "Loading MNIST database" << endl;
	time_t t;
	srand((unsigned)time(&t));

	MnistDataClass data("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	MnistDataClass test("t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte");
	
	cout << "Initiate Neural Network" << endl;
	NNClass nn(784, RELU, 10, SIGMOID);
	nn.addHiddenLayer(40, RELU);
	nn.addHiddenLayer(20, RELU);
	nn.addHiddenLayer(16, RELU);
	nn.setLearningRate(LEARNING_RATE);
	nn.setMiniBatchSize(MINI_BATCH_SIZE);
	nn.enableMomentum(MOMENTUM_FACTOR);
	nn.init();

	//training
	for (int i = 0; i < NUM_EPOCHS; i++) {
		for (int j = 0; j < data.getNumImages() / MINI_BATCH_SIZE; j++) {
			for (int k = 0; k < MINI_BATCH_SIZE; k++) {
				nn.compute(data.getPixelData(j*MINI_BATCH_SIZE + k));
				nn.backPropogate(data.getImageNumber(j*MINI_BATCH_SIZE + k));
			}
			nn.updateWeightsAndBiases();
			cout << "Batch " << j << " of epoch " << i+1 << " complete" << endl;
		}
	}

	//assessment
	int count = 0;
	for (int i = 0; i < test.getNumImages(); i++) {
		int correctNum = 0;
		for (int j = 0; j < 10; j++){
			if (test.getImageNumber(i)[j] == 1) correctNum = j;
		}
		if (correctNum != nn.compute(test.getPixelData(i))) count++;
	}
	cout << count << " of " << test.getNumImages() << " failed: " 
		<< 100*(test.getNumImages() - count) / test.getNumImages() << " pass rate." << endl;
		
	cout << "done" << endl;
	while (1){}
	return 1;
}
