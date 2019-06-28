/*
* Data_Mining -- Project Two
* Parallel GBDT Regression Problem
* Author : Palette25
*/
#include <float.h>

#include "GradientBoostedTree.hpp"

using namespace std;

const string prefix = "../../../../";
const int train_times = 6;

vector<BoostedForest> forests;

double predictResult(vector<double>& feats, int size) {
	double pred = 0.0f;
	for (int i = 0; i < size; i++) {
		pred += forests[i].predictForest(feats);
	}
	pred /= size;
	return pred;
}

int main(){
	for (int i = 0; i < 5; i++) {
		BoostedForest temp =  BoostedForest();
		forests.push_back(temp);
	}
	
	// Read In dataMat and labelMat
	FILE *feature_fp, *label_fp, *test_fp;
	vector< DFeature > vals_features;
	vector< vector<double> > train_feats;
	vector<double> train_labels;

	for(int index_=1; index_<=5; index_++){
		cout << "Reading dataSet, index: " << index_ << endl;
		string feature_file = prefix + "dataSet/train" + to_string(index_) + ".csv",
			   label_file = prefix + "dataSet/label" + to_string(index_) + ".csv";
		auto err = fopen_s(&feature_fp, feature_file.c_str(), "r");
		auto err1 = fopen_s(&label_fp, label_file.c_str(), "r");
		if (err != 0 || err1 != 0) {
			cout << "Open Files Failed!" << endl;
			return -1;
		}
		while(true){
			// Check end of file
			if (feof(feature_fp) || feof(label_fp)) {
				break;
			}

			vector<double> feats(13, 0);
			double label = 0.0;
			// First Read features csv
			fscanf_s(feature_fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",&feats[0],
					&feats[1],&feats[2],&feats[3],&feats[4],&feats[5],&feats[6],
					&feats[7],&feats[8],&feats[9],&feats[10],&feats[11],&feats[12]);
			// Then Read labels csv
			fscanf_s(label_fp, "%lf", &label);
			vals_features.push_back(DFeature(feats, label));
			// Push into final train datas
			train_labels.push_back(label);
			train_feats.push_back(feats);
		}

		// Set params

		// Build Forest
		forests[index_ - 1].buildForest(vals_features, true);

		// Calculate R2_Score base on training datas
		cout << "Start Calculating result's r2_score...." << endl;
		double mean_label = 0.0, up_square = 0.0, down_square = 0.0, r2_score = 0.0;
		for (int index = 0; index < train_labels.size(); ++index) {
			mean_label += train_labels[index];
		}
		mean_label /= train_labels.size();
		cout << "Mean Label Value: " << mean_label << endl;
		for (int index = 0; index < train_feats.size(); ++index) {
			// Combine pre-train forest result
			double pred = predictResult(train_feats[index], index_);
			up_square += (train_labels[index] - pred) * (train_labels[index] - pred);
			down_square += (train_labels[index] - mean_label) * (train_labels[index] - mean_label);
		}
		r2_score = 1 - (up_square / down_square);

		cout << "Current r2_score: " << r2_score << endl;

		// Clear Memory
		vals_features.clear();
		vector<DFeature>().swap(vals_features);
		
		err = fclose(feature_fp);
		err1 = fclose(label_fp);
		if (err || err1) {
			cout << "Close Files Failed!" << endl;
			return -1;
		}
	}
	
	cout << "Reading DataSet And Training Model Ending...." << endl;

	// Predict with test datas
	ofstream outFile;
	outFile.open(prefix + "result/gbdt_result.csv", ios::out);
	outFile << "ID" << ',' << "Predicted" << endl;

	int counter = 1;
	for (int index = 1; index <= 6; index++) {
		cout << "Start round " << index << " predicting..." << endl;
		string test_file = prefix + "dataSet/test" + to_string(index) + ".csv";
		auto err = fopen_s(&test_fp, test_file.c_str(), "r");
		if (err != 0) {
			cout << "Open Files Failed!" << endl;
			return -1;
		}
		
		while (true) {
			// End of file
			if (feof(test_fp))
				break;

			vector<double> feats(13, 0.0);
			// Read test features csv
			fscanf_s(test_fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &feats[0],
				&feats[1], &feats[2], &feats[3], &feats[4], &feats[5], &feats[6],
				&feats[7], &feats[8], &feats[9], &feats[10], &feats[11], &feats[12]);

			// Judge valid reading
			double sum = 0.0;
			for (int i = 0; i < feats.size(); i++) {
				sum += feats[i];
			}
			if (sum == 0.0)
				break;

			double predict_res = predictResult(feats, 5);

			outFile << counter << ',' << predict_res << endl;
			++counter;

		}

		cout << "Current Counter: " << counter << endl;
	}

	cout << "End of Predicting..." << endl;
	outFile.close();

	return 0;
}