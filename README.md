# Estimation-Of-Plant-Diseased-Area-Percentage-From-Photo-With-Machine-Learning

This code is a computer vision project for plant disease detection using machine learning. The project aims to classify plant images into five classes (usage, herbs, stem, soil and leaf).

The project reads a CSV file named 'data.csv' containing the RGB color values ​​for each class. Plots the RGB values ​​of each class in 3D space. It then trains a DecisionTreeClassifier model on the given data to predict the class label for a new image.

The project loads an image named "Image2.jpeg" and converts it to grayscale using OpenCV. It then applies the trained model to each pixel of the image and classifies the pixel into one of five classes. Finally, it shows the output image containing each pixel colored according to the classified class.

The program also prints the percentage of disease rate. This is calculated by dividing the number of pixels classified as infestation (infected class) by the total number of pixels classified as infestation or leaf (healthy class). The program calculates the percentage and prints it to the console.
