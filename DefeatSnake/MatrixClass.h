#pragma once

#include <vector>
#include <iostream>
#include <assert.h>
#include <sstream>

using namespace std;

template<class T>
class MatrixClass {
public:
	MatrixClass();
	MatrixClass(int height, int width);
	MatrixClass( vector< vector<T> > const &array);

	MatrixClass times		(double const &value); // scalar multiplication
	MatrixClass times		(MatrixClass const &m) const; // hadamard product
	MatrixClass plus		(MatrixClass const &m) const; // addition
	MatrixClass minus		(MatrixClass const &m) const; // subtraction
	MatrixClass dot			(MatrixClass const &m) const; // dot product
	MatrixClass transpose	() const; // transposed matrix
	MatrixClass applyFunction(double(*function)(double)) const; // to apply a function to every element of the matrix

	int getGreatest1DIndex() const;
	void print(std::ostream &flux) const; // pretty print of the matrix
	void clear();

	MatrixClass operator*(MatrixClass const &m) const;
	MatrixClass operator*(double const value);
	MatrixClass operator+(MatrixClass const &m) const;
	MatrixClass operator-(MatrixClass const &m) const;
private:
	std::vector< vector<T> > array;
	int height;
	int width;
};

template <class T>
ostream& operator<<(ostream &flux, MatrixClass<T> const &m);

#include "MatrixClass.inl"
