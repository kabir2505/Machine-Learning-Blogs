What needs to be decided on?
- Split feature
- Split point
- When to stop splitting

Steps
TRAINING
Given the whole dataset
- Calulcate Information Gain with each possible split
- Divide set with feature and value that gives the most IG
- Divide tree and do the same for all created branches
- ... until a stopping criteria is reached 

TESTING
Given a test point
- Follow the tree until you reach a leaf node
- Return the most common class label

Terms:
- Information gain 
    E(D)- weighted(E(D_yes)+E(D_no))
- Entropy
- Stopping criteria: maximum depth, minimum number of samples, min impurity decrease