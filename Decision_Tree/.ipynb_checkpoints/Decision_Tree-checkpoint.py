import numpy as np

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

    def is_leaf(self):
        return self.value is not None
    

class Decision_Tree:
    def __init__(self,min_samples_split=2,max_depth=100,n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def fit(self,X,y):
        if not self.n_features:
            self.n_features=X.shape[1]
        else:
             self.n_features=min(X.shape[1],self.n_features)
        
        self.root=self.grow_tree(X,y)


    def grow_tree(self,X,y,depth=0):
        n_samples,n_feats=X.shape
        n_labels=len(np.unique(y))
            #if n_labels==1, node is pure
        #(A) check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value=self.most_common_label(y)
            return Node(value=leaf_value)
        #(B)find the best split
        feat_idxs=np.random.choice(n_feats,self.n_features,replace=False)
        best_feature,best_thresh=self.best_split(X,y,feat_idxs)
        #(C)create child nodes
        left_idxs,right_idxs=self.split(X[:,best_feature],best_thresh)
        left=self.grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right=self.grow_tree(X[right_idxs,:],y[right_idxs],depth+1)

        return Node(best_feature,best_thresh,left,right)

    def most_common_label(self,y):
        y_=np.ravel(y)
        unique_vals,counts=np.unique(y_,return_counts=True)
        max_count_index=np.argmax(counts)
        return unique_vals[max_count_index]
    def best_split(self,X,y,feat_idxs):
        best_gain=-1
        split_idx,split_threshold=None,None

        for feat_idx in feat_idxs:
            X_column=X[:,feat_idx]
            thresholds=np.unique(X_column)

            for threshold in thresholds:
                #information gain
                gain=self.information_gain(y,X_column,threshold)


                if gain>best_gain:
                    best_gain=gain
                    split_idx=feat_idx
                    split_threshold=threshold

        return split_idx,split_threshold

    def information_gain(self,y,X_column,threshold):
        #Entropy(D)
        parent_entropy=self.entropy(y)
        #create children
        left_idx,right_idx=self.split(X_column,threshold)

        if len(left_idx)==0 or len(right_idx)==0:
            return 0

        #Entropy(D_yes) and Entropy(D_no)
        n=len(y)
        n_l,n_r=len(left_idx),len(right_idx)
        e_l,e_r=self.entropy(y[left_idx]),self.entropy(y[right_idx])

        child_entropy=(n_l/n)*e_l + (n_r/n)*e_r


        #calculate Information gain
        information_gain=parent_entropy - child_entropy
        return information_gain

    def split(self,X_column,split_thresh):
        left_idx=np.argwhere(X_column<=split_thresh).flatten()
        right_idx=np.argwhere(X_column>split_thresh).flatten()
        return left_idx,right_idx
    def entropy(self,y):
        
        #p=Proportion of 1s
        hist=np.bincount(y)
        ps=hist/len(y)
        return -np.sum([p*np.log(p) for p in ps if p>0])


    def traverse_tree(self,x,node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature]<=node.threshold:
            return self.traverse_tree(x,node.left)
        return self.traverse_tree(x,node.right)


    def predict(self,X):
        return np.array([self.traverse_tree(x,self.root) for x in X])       