#include "main.h"
#include <iostream>

int main(){
	data_t input[SEQ][DIM];
	data_t output[SEQ][DIM];
	project_top(input, output);
	std::cout << "Done" << std::endl;

	return 0;
}
