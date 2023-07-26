# CIFAR-10-Bayesian-Classifiers
The CIFAR-10 Bayesian Classifier is an open-source project that aims to implement and explore the effectiveness of Bayesian classifiers on the CIFAR-10 dataset. The project comprises two main components: Naive Bayes classification and Multivariate Bayesian classification with different image shapes.

The Naive Bayes classifier is trained on the CIFAR-10 dataset, where it learns class distributions based on color features. It computes the mean and standard deviation for each class and utilizes these parameters to classify new test samples. The project offers a function to evaluate the accuracy of the Naive Bayes classifier on the CIFAR-10 test set.

The Multivariate Bayesian classifier extends the classification approach to various image shapes, including 1x1, 2x2, 4x4, 8x8, and 16x16. It resizes the images accordingly and learns class distributions using multivariate normal distributions. The classifier predicts labels for the test set images and calculates accuracy for each image shape.

This project includes essential functions for data preprocessing, model training, and classification. Additionally, it offers visualization capabilities, enabling users to view randomly selected CIFAR-10 images and plot the accuracy of the Multivariate Bayesian classifier for different image shapes.

By providing a user-friendly implementation and exploration of Bayesian classifiers on the CIFAR-10 dataset, this project becomes a valuable resource for understanding and utilizing Bayesian classification techniques. Its adaptability in exploring diverse image shapes allows users to gain insights into the performance of Bayesian classifiers with varying image resolutions.
