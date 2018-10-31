#include "MnistDataClass.h"

MnistDataClass::MnistDataClass(){
}

int MnistDataClass::getNumImages() {
	return numImages;
}

vector<double> MnistDataClass::getPixelData(int imageNum) {
	return inputData[imageNum];
}

vector<double> MnistDataClass::getImageNumber(int imageNum) {
	return correctNumber[imageNum];
}

MnistDataClass::MnistDataClass(string dataFilename, string answerFilename) {
	ifstream dataFile;
	dataFile.open(dataFilename, ifstream::binary | ifstream::in);

	if (dataFile.good()) {
		readHeader(dataFile);
		
		//numImages = 20;
		//set vector size
		inputData.resize(numImages);
		
		//read in pixel data for each image
		for (int i = 0; i < numImages; i++) {
			for (int j = 0; j < imageRow*imageCol; j++) {
				if (double(dataFile.get() & 0xff) > 0) inputData[i].push_back(1.0);
				else inputData[i].push_back(0);
				//inputData[i].push_back(double(uint8_t(dataFile.get() & 0xff)));
			}
		}
	}
	dataFile.close();

	dataFile.open(answerFilename, ifstream::binary | ifstream::in);
	if (dataFile.good()) {
		//dump the first 8 bytes
		for (int i = 0; i < 8; i++) {
			dataFile.get();
		}

		correctNumber.resize(numImages);

		//read in answer data for each image
		for (int i = 0; i < numImages; i++) {
			correctNumber[i].resize(10);
			correctNumber[i][double(dataFile.get() & 0xff)] = 1;
		}
	}
	dataFile.close();

}

void MnistDataClass::readHeader(ifstream &file) {
	char c;
	int magicNumber = 0;
	for (int i = 0; i < 4; i++) {
		c = file.get();
		magicNumber = (magicNumber << 8) | (c & 0xff);
	}

	for (int i = 0; i < 4; i++) {
		c = file.get();
		numImages = (numImages << 8) | (c & 0xff);
	}

	for (int i = 0; i < 4; i++) {
		c = file.get();
		imageRow = (imageRow << 8) | (c & 0xff);
	}

	for (int i = 0; i < 4; i++) {
		c = file.get();
		imageCol = (imageCol << 8) | (c & 0xff);
	}
}

MnistDataClass::~MnistDataClass()
{
}
