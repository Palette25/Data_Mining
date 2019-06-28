#include "Utils.hpp"
#include "DecisionTree.hpp"

// Definition Of Gradient Boosting Forest
class BoostedForest 
{
public:
    vector<DecisionTree*> trees;
    int depth, max_feature, max_pos, min_children, nthread, num_tree;
    double bootstrap, step;
    vector<double> cur_vals, ori_vals;
    vector<double> steps;
    vector<DFeature> *val_features_ptr;

    BoostedForest() {
        val_features_ptr = NULL;
        step = 0.08;
        depth = 6;
		max_feature = -1;
		max_pos = -1;
        min_children = 10;
        nthread = 8;
        bootstrap = 0.0;
		num_tree = 170;
    }
    
    void set_val_data(vector<DFeature> &data) {
        val_features_ptr = &data;
    }

    void set_params(const char *name, const char *val) {
        if (!strcmp(name, "num_tree")) num_tree = atoi(val);
        if (!strcmp(name, "depth")) depth = atoi(val);
        if (!strcmp(name, "min_children")) min_children = atoi(val);
        if (!strcmp(name, "nthread")) nthread = atoi(val);
        if (!strcmp(name, "step")) step = static_cast<double> (atof(val));
        if (!strcmp(name, "bootstrap")) bootstrap = static_cast<double> (atof(val));
    }   
 
    void buildForest(vector<DFeature> &features, bool is_show) {        
                
        if (max_feature < 0) max_feature = int(sqrt(features[0].f.size()) + 1);

        // load train data
        cur_vals = vector<double>(features.size());
        ori_vals = vector<double>(features.size());
        for (int i = 0; i < features.size(); i++)
            ori_vals[i] = features[i].y;
        
        // load validation data
        vector<double> val_vals;
        vector<double> pred_vals;
        if (val_features_ptr != NULL) {
            vector<DFeature> &val_features = *val_features_ptr;
            pred_vals = vector<double>(val_features.size());
            val_vals = vector<double>(val_features.size());
            for (int i = 0; i < val_features.size(); i++)
                val_vals[i] = val_features[i].y;
        }        
        
        double train_time = 0.0;
        double start_time = get_time();
        // for each tree
        for (int i = 0; i < num_tree; i++)
        {
			double train_rmse = -1, test_rmse = -1;
			double train_auc = -1, test_auc = -1;

            double iter_start_time = get_time();
            // update residual for each instance
            for (int j = 0; j < features.size(); j++)
                features[j].y = ori_vals[j] - cur_vals[j];
            // create a decision tree, the only different is features.y(residual)
            DecisionTree *dt = new DecisionTree(features, depth, max_feature, max_pos, min_children, bootstrap, nthread);
            trees.push_back(dt);
			
			for (int j = 0; j < features.size(); j++) {
				cur_vals[j] += dt->predictTree(features[j].f) * step; // pred of this instance sum(leaf node value of all trees)
			}

			train_rmse = cal_rmse(cur_vals, ori_vals);
			train_auc = cal_r2_score(cur_vals, ori_vals);
            
            train_time += get_time() - iter_start_time;
            
            if (val_features_ptr != NULL) {                
                vector<DFeature> &val_features = *val_features_ptr;
                for (int j = 0; j < val_features.size(); j++) {
                    pred_vals[j] += dt->predictTree(val_features[j].f) * step;                    
                }
                test_rmse = cal_rmse(pred_vals, val_vals);
                test_auc = cal_r2_score(pred_vals, val_vals);
            }
            
            steps.push_back(step);
            
			if (is_show || i == num_tree - 1) {
				printf("Iteration: %d, Train-Rmse: %.6lf, tree_size: %d\n", i + 1, train_rmse, dt->tree.size());
				printf("Train_R2_Score: %.6lf\n", train_auc);
				printf("%.3f seconds passed, %.3f seconds in training\n", get_time() - start_time, train_time);
				printf("--------------------------------------------------\n");
			}

        }
                
        for (int j = 0; j < features.size(); j++)
            features[j].y = ori_vals[j];
    }

    void addTree(vector<DFeature> &features) {        
        addTree(features, 1);
    }

    void addTree(vector<DFeature> &features, int treecnt) {     
		cur_vals = vector<double>(features.size());
		ori_vals = vector<double>(features.size());

        for (int j = 0; j < features.size(); j++) {
            ori_vals[j] = features[j].y;            
        }
        while (treecnt--) {
			double train_time = 0.0;
			double start_time = get_time();

            for (int j = 0; j < features.size(); j++) {                
                features[j].y = ori_vals[j] - cur_vals[j];
            }
            DecisionTree *dt = new DecisionTree(features, depth, max_feature, max_pos, min_children, bootstrap, nthread);
            trees.push_back(dt);                          
            for (int j = 0; j < features.size(); j++) {
                cur_vals[j] += dt->predictTree(features[j].f) * step;
            }
            steps.push_back(step);

			// evaluation
			double train_rmse = cal_rmse(cur_vals, ori_vals);
			double train_auc = cal_r2_score(cur_vals, ori_vals);

			printf("Iteration: %d, Train-Rmse: %.6lf, tree_size: %d\n", treecnt + 1, train_rmse, dt->tree.size());
			printf("Train_R2_Score: %.6lf\n", train_auc);
			printf("%.3f seconds passed, %.3f seconds in training\n", get_time() - start_time, train_time);
			printf("--------------------------------------------------\n");

        }
        for (int j = 0; j < features.size(); j++) {            
            features[j].y = ori_vals[j];
        }
    }
    
    void set_step(double step_) {
        step = step_;
    }
    
    double predictForest(vector<double> &f) {
        double ret = 0; 
        for (int j = 0; j < trees.size(); j++) {
            ret += trees[j]->predictTree(f) * steps[j];
			
        }        
        return ret;
    }
};
