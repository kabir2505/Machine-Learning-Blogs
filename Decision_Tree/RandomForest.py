
from Decision_Tree import Decision_Tree
import numpy as np

class RandomForest:
    def __init__(self,n_trees=10,max_depth=10,min_samples_split=2,n_features=None,):
        self.n_trees=n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_features
        self.trees=[]
    
    def fit(self,X,y):

        for _ in range(self.n_trees):
            tree=Decision_Tree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,n_features=self.n_features)
            X_sample,y_sample=self.bootstrap(X,y)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)

    def bootstrap(self,X,y):
        n_samples=X.shape[0]
        idxs=np.random.choice(n_samples,n_samples,replace=True)
        return X[idxs],y[idxs]

    def most_common_label(self,y):
        y_=np.ravel(y)
        unique_vals,counts=np.unique(y_,return_counts=True)
        max_count_index=np.argmax(counts)
        return unique_vals[max_count_index]
  
    def predict(self,X):
        predictions=np.array([tree.predict(X) for tree in self.trees])
        tree_preds=np.swapaxes(predictions,0,1)
        final_predictions=np.array([self.most_common_label(pred) for pred in tree_preds])
        return final_predictions
