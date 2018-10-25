#include <assert.h>
#include <sstream>
#include "MatrixClass.h"

MatrixClass::MatrixClass() {}

MatrixClass::MatrixClass(int inHeight, int inWidth){
	height = inHeight;
	width = inWidth;
	array = std::vector<std::vector<double>>(inHeight, std::vector<double>(inWidth));
}

void MatrixClass::clear() {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			array[i][j] = 0;
		}
	}
}

MatrixClass::MatrixClass(std::vector<std::vector<double>> const &inArray){
	assert(inArray.size() != 0);
	height = inArray.size();
	width = inArray[0].size();
	array = inArray;
}

MatrixClass MatrixClass::multiply(double const &value){
	MatrixClass result(height, width);

	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			result.array[i][j] = array[i][j] * value;
		}
	}

	return result;
}

MatrixClass MatrixClass::add(MatrixClass const &m) const {
	assert(height == m.height && width == m.width);
	MatrixClass result(height, width);

	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			result.array[i][j] = array[i][j] + m.array[i][j];
		}
	}

	return result;
}

MatrixClass MatrixClass::subtract(MatrixClass const &m) const {
	assert(height == m.height && width == m.width);
	MatrixClass result(height, width);

	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			result.array[i][j] = array[i][j] - m.array[i][j];
		}
	}

	return result;
}

MatrixClass MatrixClass::multiply(MatrixClass const &m) const {
	assert(height == m.height && width == m.width);
	MatrixClass result(height, width);

	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			result.array[i][j] = array[i][j] * m.array[i][j];
		}
	}
	return result;
}

MatrixClass MatrixClass::dot(MatrixClass const &m) const {
	assert(width == m.height);
	MatrixClass result(height, m.width);
	double w = 0;
	
	for(int i = 0; i < height; i++){
		for(int j = 0; j < m.width; j++){
			for(int h = 0; h < width; h++){
				w += array[i][h] * m.array[h][j];
			}
			result.array[i][j] = w;
			w = 0;
		}
	}

	return result;
}

MatrixClass MatrixClass::transpose() const { // ij element becomes ji element
	MatrixClass result(width, height);

	for(int i = 0; i < width; i++){
		for(int j = 0; j < height; j++){
			result.array[i][j] = array[j][i];
		}
	}
	return result;
}

MatrixClass MatrixClass::applyFunction(double(*function)(double)) const {
	MatrixClass result(height, width);

	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			result.array[i][j] = (*function)(array[i][j]);
		}
	}

	return result;
}

void MatrixClass::print(std::ostream &flux) const { // pretty print, taking into account the space between each element of the matrix
	int *maxLength;
	maxLength = new int [width];
	std::stringstream ss;
	
	for(int i = 0; i < width; i++){
		maxLength[i] = 0;
	}
	
	for(int i = 0; i < height; i++) {
		for(int j = 0; j < width; j++){
			ss << array[i][j];
			if(maxLength[j] < ss.str().size()){
				maxLength[j] = ss.str().size();
			}
			ss.str(std::string());
		}
	}
	
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			flux << array[i][j];
			ss << array[i][j];
			for(int k = 0; k < maxLength[j] - ss.str().size() + 1; k++){
				flux << " ";
			}
			ss.str(std::string());
		}
		flux << std::endl;
	}
	
	delete maxLength;
}

std::ostream& operator<<(std::ostream &flux, MatrixClass const &m){
	m.print(flux);
	return flux;
}
