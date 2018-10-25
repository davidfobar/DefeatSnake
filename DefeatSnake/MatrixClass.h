#pragma once

#include <vector>
#include <iostream>

class MatrixClass
{
public:
	MatrixClass();
	MatrixClass(int height, int width);
	MatrixClass(std::vector<std::vector<double>> const &array);

	MatrixClass multiply	(double const &value); // scalar multiplication
	MatrixClass add			(MatrixClass const &m) const; // addition
	MatrixClass subtract	(MatrixClass const &m) const; // subtraction
	MatrixClass multiply	(MatrixClass const &m) const; // hadamard product
	MatrixClass dot			(MatrixClass const &m) const; // dot product
	MatrixClass transpose	() const; // transposed matrix
	MatrixClass applyFunction(double(*function)(double)) const; // to apply a function to every element of the matrix

	void print(std::ostream &flux) const; // pretty print of the matrix
	void clear();

private:
	std::vector<std::vector<double>> array;
	int height;
	int width;
};

std::ostream& operator<<(std::ostream &flux, MatrixClass const &m); // overloading << operator to print easily
