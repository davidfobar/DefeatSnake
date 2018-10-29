
template<class T>
MatrixClass<T>::MatrixClass() {}

template<class T>
int MatrixClass<T>::getGreatest1DIndex() const {
	assert(height == 1);
	double test = 0;
	int result = 0;
	for (int i = 0; i < width; i++) {
		if (array[0][i] > test) {
			test = array[0][i];
			result = i;
		}
	}

	return result;
}

template<class T>
MatrixClass<T>::MatrixClass(int inHeight, int inWidth) {
	height = inHeight;
	width = inWidth;
	array = vector< vector<T> >(height, vector<T>(width));
}

template<class T>
void MatrixClass<T>::clear() {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			array[i][j] = 0;
		}
	}
}

template<class T>
MatrixClass<T>::MatrixClass(vector< vector<T> > const &inArray) {
	assert(inArray.size() != 0);
	height = inArray.size();
	width = inArray[0].size();
	array = inArray;
}

template<class T>
MatrixClass<T> MatrixClass<T>::times(double const &value) {
	MatrixClass result(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			result.array[i][j] = array[i][j] * value;
		}
	}

	return result;
}

template<class T>
MatrixClass<T> MatrixClass<T>::operator*(double const value) {
	return this->times(value);
}

template<class T>
MatrixClass<T> MatrixClass<T>::plus(MatrixClass<T> const &m) const {
	assert(height == m.height && width == m.width);
	MatrixClass<T> result(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			result.array[i][j] = array[i][j] + m.array[i][j];
		}
	}

	return result;
}

template<class T>
MatrixClass<T> MatrixClass<T>::operator+(MatrixClass<T> const &m) const {
	return this->plus(m);
}

template<class T>
MatrixClass<T> MatrixClass<T>::minus(MatrixClass<T> const &m) const {
	assert(height == m.height && width == m.width);
	MatrixClass result(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			result.array[i][j] = array[i][j] - m.array[i][j];
		}
	}

	return result;
}

template<class T>
MatrixClass<T> MatrixClass<T>::operator-(MatrixClass<T> const &m) const {
	return this->minus(m);
}

template<class T>
MatrixClass<T> MatrixClass<T>::times(MatrixClass<T> const &m) const {
	assert(height == m.height && width == m.width);
	MatrixClass result(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			result.array[i][j] = array[i][j] * m.array[i][j];
		}
	}
	return result;
}

template<class T>
MatrixClass<T> MatrixClass<T>::operator*(MatrixClass<T> const &m) const {
	return this->times(m);
}

template<class T>
MatrixClass<T> MatrixClass<T>::dot(MatrixClass<T> const &m) const {
	assert(width == m.height);
	MatrixClass result(height, m.width);
	double w = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < m.width; j++) {
			for (int h = 0; h < width; h++) {
				w += array[i][h] * m.array[h][j];
			}
			result.array[i][j] = w;
			w = 0;
		}
	}

	return result;
}

template<class T>
MatrixClass<T> MatrixClass<T>::transpose() const { // ij element becomes ji element
	MatrixClass result(width, height);

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			result.array[i][j] = array[j][i];
		}
	}
	return result;
}

template<class T>
MatrixClass<T> MatrixClass<T>::applyFunction(double(*function)(double)) const {
	MatrixClass result(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			result.array[i][j] = (*function)(array[i][j]);
		}
	}

	return result;
}

template<class T>
void MatrixClass<T>::print(std::ostream &flux) const { // pretty print, taking into account the space between each element of the matrix
	int *maxLength;
	maxLength = new int[width];
	std::stringstream ss;

	for (int i = 0; i < width; i++) {
		maxLength[i] = 0;
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			ss << array[i][j];
			if (maxLength[j] < ss.str().size()) {
				maxLength[j] = ss.str().size();
			}
			ss.str(std::string());
		}
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			flux << array[i][j];
			ss << array[i][j];
			for (int k = 0; k < maxLength[j] - ss.str().size() + 1; k++) {
				flux << " ";
			}
			ss.str(std::string());
		}
		flux << std::endl;
	}

	delete maxLength;
}

template<class T>
std::ostream& operator<<(std::ostream &flux, MatrixClass<T> const &m) {
	m.print(flux);
	return flux;
}
