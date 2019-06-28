#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include "Utils.hpp"

// instance info
struct DFeature {
	vector<double> f; // features of this instance
	double y; // residuals for this instance
	DFeature(vector<double>& f, double& y) {
		this->f = f;
		this->y = y;
	}
	DFeature(){}
};

// tree node info
struct TreeNode
{
	double value;
	double splitval; // split threshold
	int ind; //which feature be the split point
	int ch[2];
	double sum_y;
	double sum_sqr_y;
};

// Leaf Node
struct LeafNode {
	int nid;
	int cnt;
	double err;

	LeafNode() {
		nid = cnt = 0;
	}

	LeafNode(const int &nid_, const int &cnt_) {
		nid = nid_;
		cnt = cnt_;
		err = 0.0f;
	}

	LeafNode(const int &nid_, const int &cnt_, const double &err_) {
		nid = nid_;
		cnt = cnt_;
		err = err_;
	}
};

struct SplitInfo {
	int bind; // Feature ID
	double bsplit; // Split Threshold
	int cnt[2]; // Num of instances for right and left
	double sum_y[2], sum_sqr_y[2];
	double err;

	void update(const SplitInfo &sinfo) {
		bind = sinfo.bind;
		bsplit = sinfo.bsplit;
		cnt[0] = sinfo.cnt[0]; 
		cnt[1] = sinfo.cnt[1];
		sum_y[0] = sinfo.sum_y[0]; 
		sum_y[1] = sinfo.sum_y[1];
		sum_sqr_y[0] = sinfo.sum_sqr_y[0]; 
		sum_sqr_y[1] = sinfo.sum_sqr_y[1];
		err = sinfo.err;
	}
};

struct ThreadInfo {
	int cnt0; // Right num of instances for split
	double sum0, ss0;
	double last_val;
	SplitInfo spinfo;
};

class DecisionTree {
private:
    vector<LeafNode> q; // All leaf nodes for one level

    vector<SplitInfo> split_infos; // Split info for each LeafNode in q
    
    double *y_list, *sqr_y_list;
    int *positions;
    
    vector<DFeature> *features_ptr;
    
    public:
    vector<TreeNode> tree;
    
    int min_children;
    int max_depth;
    
    int n; // Number of instances
    int m; // Number of features
    int nthread; // Number of threads    
    
    
private:
    void init_data() {
        q.reserve(256);
        q.resize(0);
        split_infos.reserve(256);
        split_infos.resize(0);
        tree.reserve(256);
        tree.resize(0);
        
        omp_set_num_threads(nthread);
        
        #pragma omp parallel
        {
            this->nthread = omp_get_num_threads();
        }
        printf("Parallel Decision Tree Building, Number of thread: %d\n", this->nthread);
    }       
    
    void update_queue() {
        vector<DFeature> &features = *features_ptr;
        vector<LeafNode> new_q; // all leaf node
        TreeNode new_node;
        vector< pair<int, int> > children_q_pos(q.size());
        for (int i = 0; i < q.size(); i++) {
            //printf("nid: %d, left: %d, right: %d\n", q[i].nid, q[i].left, q[i].right);
			/*
            printf("nid: %d, cnt %d\n", q[i].nid, q[i].cnt);
            printf("bind: %d, bsplit: %f, cnt0: %d, cnt1: %d\n", split_infos[i].bind, 
                split_infos[i].bsplit, split_infos[i].cnt[0], split_infos[i].cnt[1]);
            printf("sum0: %f, sum1: %f, v0: %f, v1: %f\n", split_infos[i].sum_y[0], 
                split_infos[i].sum_y[1], split_infos[i].sum_y[0] / max(1, split_infos[i].cnt[0]),
                split_infos[i].sum_y[1] / max(1, split_infos[i].cnt[1]));
            printf("\n");
			*/
			
			
            if (split_infos[i].bind >= 0) {
                // put the final split info to tree
                int ii = q[i].nid; // ith of LeafNode is iith of tree
                tree[ii].ind = split_infos[i].bind;                        
                tree[ii].splitval = split_infos[i].bsplit;                
                tree[ii].ch[0] = tree.size();
                tree[ii].ch[1] = tree.size() + 1;
                children_q_pos[i].first = new_q.size();
                children_q_pos[i].second = new_q.size() + 1;
                
                //new_q.push_back(LeafNode(tree.size(), split_infos[i].cnt[0], split_infos));
                //new_q.push_back(LeafNode(tree.size() + 1, split_infos[i].cnt[1]));
                                                
                // create two leaf node of tree and two LeafNode with info
                for (int c = 0; c < 2; c++) {
                    new_node.ind = -1;
                    new_node.value = split_infos[i].sum_y[c] / split_infos[i].cnt[c]; // mean of sum_y
                    new_node.sum_y = split_infos[i].sum_y[c];
                    new_node.sum_sqr_y = split_infos[i].sum_sqr_y[c];
                    double err = new_node.sum_sqr_y - new_node.sum_y*new_node.sum_y/split_infos[i].cnt[c];
                    new_q.push_back(LeafNode(tree.size(), split_infos[i].cnt[c], err));
                    tree.push_back(new_node);
                }
            }
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            int &pos = positions[i];
            if (pos >= 0 && split_infos[pos].bind >= 0) {
                if (features[i].f[split_infos[pos].bind] <=  split_infos[pos].bsplit) {
                    pos = children_q_pos[pos].first;
                } else {
                    pos = children_q_pos[pos].second;
                }                
            } else pos = -1;
        }
        
        q = new_q;
    }
       
    // set initial value and sort the column feature list
    void initial_column_feature_list(vector< vector< pair<double, int> > > &col_fea_list, vector<int> &id_list) {
        
        vector<DFeature> &features = *features_ptr; 
        
        col_fea_list.resize(m); // m features
                
        for (int i = 0; i < m; i++) {
            col_fea_list[i].resize(id_list.size()); // each feature has id_list.size() instances
        }
        
        #pragma omp parallel for schedule(static)    
        for (int i = 0; i < id_list.size(); i++) {
            int ins_id = id_list[i];
            for (int j = 0; j < m; j++) {
                col_fea_list[j][i].first = features[ins_id].f[j];
                col_fea_list[j][i].second = ins_id; 
            }
        }
        
        #pragma omp parallel for schedule(dynamic,1)
        for (int i = 0; i < m; i++) {
            sort(col_fea_list[i].begin(), col_fea_list[i].end()); // sort value of all instance for each feature
        }        
    }       
    
    // linear search for best split
    void find_split(int fid, vector< pair<double, int> > &fea_list, vector<ThreadInfo> &tinfo_list) {
        
        // each q node
        for (int i = 0; i < tinfo_list.size(); i++) {
            tinfo_list[i].cnt0 = 0;        
            tinfo_list[i].sum0 = 0.0f;
            tinfo_list[i].ss0 = 0.0f;
        }
        
        double ss1, sum1, err;
        int top = 0;
        // each instances
        for (int i = 0; i < fea_list.size(); i++) {
            int iid = fea_list[i].second; // instance id
            int pos = positions[iid]; // which node of q belong to
            if (pos < 0) 
            	continue;
            
            fea_list[top++] = fea_list[i];
            int nid = q[pos].nid; // node id of whole tree
            ThreadInfo &tinfo = tinfo_list[pos];

            if (tinfo.cnt0 >= min_children && q[pos].cnt - tinfo.cnt0 >= min_children 
					&& sign(fea_list[i].first - tinfo.last_val) != 0) {
                double &sum0 = tinfo.sum0;
                double &ss0 = tinfo.ss0;
                sum1 = tree[nid].sum_y - sum0;
                ss1 = tree[nid].sum_sqr_y - ss0;

                err = ss0 + ss1 - sum0 * sum0 / tinfo.cnt0 - sum1 * sum1 / (q[pos].cnt-tinfo.cnt0);
                // save new split info for this node
                if (sign(err - tinfo.spinfo.err) < 0) {
                    SplitInfo &tbest = tinfo.spinfo;
                    tbest.err = err;
                    tbest.bind = fid;
                    tbest.bsplit = (fea_list[i].first + tinfo.last_val) / 2; // split threshold
                    tbest.sum_y[0] = sum0; tbest.sum_y[1] = sum1;
                    tbest.sum_sqr_y[0] = ss0; tbest.sum_sqr_y[1] = ss1;
                    tbest.cnt[0] = tinfo.cnt0; tbest.cnt[1] = q[pos].cnt - tinfo.cnt0;
                }
            }
            tinfo.cnt0 += 1;
            tinfo.sum0 += y_list[iid];
            tinfo.ss0 += sqr_y_list[iid];                                               
            tinfo.last_val = fea_list[i].first;            
        }
        fea_list.resize(top);
    }
        
public:
    DecisionTree(vector<DFeature> &features, int max_depth, int max_feature, int max_pos, 
        int min_children, double bootstrap, int nthread) {
        
        this->n = features.size();        
        this->m = features.size() > 0 ? features[0].f.size() : 0;
        this->min_children = max(min_children, 1);
        this->max_depth = max_depth;
        this->nthread = nthread ? nthread : 1;
        this->features_ptr = &features;
        
        init_data();
        
        vector<int> id_list; // Instance num. of train data
        id_list.reserve(n);
        double sum_y = 0.0;
        double sum_sqr_y = 0.0;
        int tcnt = 0;
        
        y_list = new double[n];
        sqr_y_list = new double[n];
        positions = new int[n]; // Whether this instance is used
            
        // Process bootstrap
        for (int i = 0; i < n; i++) {
            if ((double)get_rand() / RAND_MAX >= bootstrap) {
                id_list.push_back(i); // Choose ith instance as train data
                y_list[i] = features[i].y;
                sqr_y_list[i] = sqr(features[i].y);
                sum_y += y_list[i];
                sum_sqr_y += sqr_y_list[i];                
                positions[i] = 0;
            } else {
                positions[i] = -1;
            }
        }        
        
		/*
		* Start Building Decision Tree 
		*/

        // Add the root node        
        TreeNode node;
        node.ind = -1;
        node.value = sum_y / (id_list.size() ? id_list.size() : 1);
        node.sum_y = sum_y;
        node.sum_sqr_y = sum_sqr_y;        
        tree.push_back(node);
        
        if (id_list.size() == 0) 
			return;
        q.push_back(LeafNode(0, id_list.size(), sum_sqr_y - sum_y * sum_y / id_list.size()));  
        
        // Set initial value and sort the column feature list
        vector< vector< pair<double, int> > > col_fea_list;
        initial_column_feature_list(col_fea_list, id_list);     
        
        vector< vector<ThreadInfo> > tinfos(nthread);
        
        // Build a decision tree         
        for (int dep = 0; dep < max_depth; dep++) {
            if (q.size() == 0) break;
            
            int nq = q.size();
            split_infos.resize(q.size());
            
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nthread; i++) {
                tinfos[i].resize(q.size());                
                for (int j = 0; j < q.size(); j++) {                    
                    tinfos[i][j].spinfo.bind = -1;
                    tinfos[i][j].spinfo.err = q[j].err;
                }
            }
            
            #pragma omp parallel for schedule(dynamic,1)
            for (int fid = 0; fid < m; fid++) {         
                const int tid = omp_get_thread_num();
                find_split(fid, col_fea_list[fid], tinfos[tid]);
            }
            
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nq; i++) {
                SplitInfo &spinfo = split_infos[i];
                spinfo.bind = -1;
                for (int j = 0; j < nthread; j++)
                    if (tinfos[j][i].spinfo.bind >= 0 && (spinfo.bind < 0 || spinfo.err > tinfos[j][i].spinfo.err))
                        spinfo.update(tinfos[j][i].spinfo);
            }
            
           /* update tree nodes
           (assign split info and create new leaf nodes) 
           and q nodes(new leaf nodes) */
            update_queue();
        }    
                   
        delete[] y_list;
        delete[] sqr_y_list;
        delete[] positions;
#ifdef cpp11        
        tree.shrink_to_fit();
#endif     
    }
        
    double predictTree(vector<double> &f) {
        int n = 0;
        while (tree[n].ind >= 0) // whether tree[n] is leaf node
        {
            if (f[ tree[n].ind ] <= tree[n].splitval)
                n = tree[n].ch[0];
            else
                n = tree[n].ch[1];
        }
        return tree[n].value;
    }           
};

#endif