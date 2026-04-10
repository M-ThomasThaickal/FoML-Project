# Feedback from TAs

### Project Proposal

1. For the training/testing/validation split, you could include ablations: e.g., every class occurs in all sets/zero-shot settings where some classes in the testing set have never appeared in the training set.
2. Ablations on different configurations of the SVM/CNN would be interesting!

### Project Updates

1. Image feature extraction: I would like to ask if adding more feature extraction methods/ablating the existing method wrt parameters could be helpful in increasing the accuracy. At the same time, if we could visualize the extracted HOGs paired with the original features we might be able to do some sanity checks, or getting some intuitive ideas for improving the performance.
2. SVM models ablation. I would like to ask if it would be helpful to have some further comparisons that uses SVM on the raw data, to see if it is the HOG extraction that is decreasing the model performance.
