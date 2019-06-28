#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctime>
#include <cassert>
#include <omp.h>
using namespace std;

const double EPSI = 1e-4;

unsigned long long now_rand = 1;

double get_time() {    
	clock_t curr_time = clock();
	return (double)(curr_time / CLOCKS_PER_SEC);
}

void set_rand_seed(unsigned long long seed)
{
    now_rand = seed;
}

unsigned long long get_rand()
{
    now_rand = ((now_rand * 6364136223846793005ULL + 1442695040888963407ULL) >> 1);
    return now_rand;
}

inline double sqr(const double &x) {
    return x * x;
}

inline int sign(const double &val) {
    if (val > EPSI) return 1;
    else if (val < -EPSI) return -1;
    else return 0;
}

// cal rmse for evaluation
double cal_rmse(vector<double> &pred, vector<double> &gt) {
    assert(pred.size() == gt.size());
    double rmse = 0;
    for (int i = 0; i < pred.size(); i++) {
        rmse += sqr(pred[i] - gt[i]);
    }
    rmse = sqrt(rmse / pred.size());
    return rmse;
}

// cal auc for evaluation
double cal_r2_score(vector<double> &pred, vector<double> &gt) {
    assert(pred.size() == gt.size());
	double mean_label = 0.0, up_square = 0.0, down_square = 0.0, r2_score = 0.0;
	for (int index = 0; index < gt.size(); ++index) {
		mean_label += gt[index];
	}
	mean_label /= gt.size();
	
	for (int index = 0; index < gt.size(); ++index) {
		up_square += (gt[index] - pred[index]) * (gt[index] - pred[index]);
		down_square += (gt[index] - mean_label) * (gt[index] - mean_label);
	}
	r2_score = 1 - (up_square / down_square);
	return r2_score;
}

#endif