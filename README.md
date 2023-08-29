Assignment III - ML  Report


Part I

 		The MNIST Digit Dataset

We have chosen the following feature extraction methods for our MNIST handwritten digit classification task: pixel intensity feature, HOG (Histogram of Oriented Gradients) feature, and PCA (Principal Component Analysis) feature.
1, Pixel Intensity Feature:
We have chosen the pixel intensity feature because it provides a simple representation of the MNIST images. By normalizing the pixel values to a range between 0 and 1, we made the features more comparable and suitable for the softmax regression model. This feature extraction method allowed the model to learn patterns based on the intensity values of the individual pixels.
2, HOG feature:
We have opted for the HOG feature because it captures local image gradient information. By computing the HOG feature vector, we extracted information about the shape and structure of the digits, which can be beneficial for distinguishing between different classes.

3, PCA Feature:
We have utilized the PCA feature extraction method to reduce the dimensionality of the data. By applying PCA, we transformed the high-dimensional pixel intensity features into a lower-dimensional representation. This helped us in reducing the computational complexity of the model and to potentially enhance its performance by focusing on the most important features (principal components) that explain the majority of the variance in the dataset.

Overall, with these feature extraction methods, we aimed to provide the softmax regression model with a diverse set of features that capture different aspects of the MNIST images. BBC Text Classification




The Demo Weather Dataset

We have chosen the following feature extraction methods for our Weather demo classification task: Extract_features_basic, Extract_ features _numeric  ,Extract_features _categorical 

1, Extract_features_basic

 This feature extraction implementation excludes the 'RainToday' and 'RainTomorrow' columns from the dataset. These columns are excluded because 'RainTomorrow' is the target variable we want to predict, and 'RainToday' is closely related to the target variable. By excluding them, we focus on the remaining features to understand their independent influence on predicting rain tomorrow. This approach simplifies the dataset and reduces potential correlations between the target variable and the extracted features.
	
2, Extract_features_numeric
	
This implementation specifically handles numeric features in the dataset. It converts numeric columns to float values and handles missing values ('NA') by replacing them with 0.0. Additionally, it deals with columns containing categorical values, such as wind direction. By converting these categorical values into one-hot encoded numeric values, the model can treat each category independently. This allows the model to capture the impact of different categories and their relationship with predicting rain tomorrow. The implementation accounts for both numeric and categorical aspects of the dataset, providing a more comprehensive representation of the features.

3, Extract_features_categorical

This feature extraction implementation focuses solely on the categorical features in the dataset. It selects columns that represent categorical variables, such as wind direction. By isolating these features, it aims to understand the individual impact of categorical variables on predicting rain tomorrow. This approach can be useful when categorical variables play a significant role in determining the outcome and when we want to analyze the specific influence of these variables.

overall, Each feature extraction implementation serves a distinct purpose and is used based on the specific needs of the analysis or prediction task. The choice of which implementation to use depends on the dataset characteristics, the problem being addressed, and the research objectives. These implementations allow for a flexible and adaptable approach to feature extraction, enabling a deeper understanding of the relationship between the features and the target variable.




The BBC Text Classification

We have chosen the following feature extraction methods for our BBC datasets classification task: 

1, TF-IDF Method:

The TF-IDF method is chosen for feature extraction in the BBC text classification example because it effectively captures the importance of terms within individual documents while considering their relevance across the entire corpus. By assigning higher weights to terms that are specific to individual documents and less common in the corpus, TF-IDF helps identify distinguishing features that can be useful for classification. This method allows the classifier to focus on terms that are highly informative and discriminatory for each class, improving the accuracy of the classification results.

    		2, Bag-of-Words (BOW) Method:

The BOW method is utilized for feature extraction in the BBC text classification scenario due to its simplicity and efficiency. BOW represents each document as a vector by considering the presence or absence of words, thereby creating a straightforward representation of the documents. Although it disregards grammar and word order, BOW can still capture important keywords or features that contribute to the classification task. By employing BOW, the classifier can learn patterns based on the occurrence or frequency of specific words, enabling effective classification even in the absence of complex linguistic structures.

3, MY Feature:

The MY_Feature class is implemented for feature extraction in text classification using the BBC dataset. The method extracts the top k features from the input data based on their frequency of occurrence across different classes(obtained by dividing the max word frequency across the classes by the sum of the frequency of the word across the classes) . The reason for implementing this method is to identify the most informative features that can contribute significantly to the classification task. By focusing on the frequency of features within each class, the method aims to capture the discriminatory power of certain terms or words that are more prevalent in specific categories. This approach can help in reducing the dimensionality of the data and improving the efficiency and effectiveness of the classification process.








 


NAIVE BAYES
1) Naive Bayes Model Experiment On BBC Data set

Based on the experiments conducted using different feature extraction methods (Bag of Words, TF-IDF, and MY) with varying Laplace smoothing values, the following insights can be derived:

1.1 Bag of Words (BoW) Feature:

The model accuracy is highest (95%) when using a Laplace smoothing value of 0.1.
As the Laplace smoothing value increases, the accuracy gradually decreases, indicating that excessive smoothing can have a negative impact on the model's performance.
The accuracy drops significantly to 47% when the Laplace smoothing value is set to 100, indicating that overly aggressive smoothing can cause the model to underperform.
1.2 TF-IDF Feature:

The model achieves the highest accuracy (94%) with a Laplace smoothing value of 0.1, similar to the Bag of Words feature.
Increasing the Laplace smoothing value leads to a gradual decline in accuracy.
The accuracy decreases to 47% when the Laplace smoothing value is set to 100, similar to the Bag of Words feature.

1.3 MY Feature:

The highest accuracy (85%) is obtained with a Laplace smoothing value of 0.1.
Similar to the other feature extraction methods, increasing the Laplace smoothing value causes the accuracy to decrease gradually.
The accuracy decreases to 48% when the Laplace smoothing value is set to 100.




Table summarising the insights:

Feature Extraction Method
Optimal Laplace Smoothing Value
Corresponding Accuracy
Bag of Words
0.1
95%
TF-IDF
0.1
94%
MY
0.1
85%


Generalised Insights:

Based on the performance analysis of the Naive Bayes model with different feature extraction methods and varying Laplace smoothing values, the following observations can be made:

For all feature extraction methods (Bag of Words, TF-IDF, and MY), lower values of Laplace smoothing (0.1) tend to yield higher accuracy.
As the Laplace smoothing value increases, the model's accuracy gradually decreases.
Excessive smoothing (higher Laplace smoothing values) can lead to a significant drop in accuracy, indicating the importance of finding an optimal balance between smoothing and preserving the discriminative power of the features.

2)Naive Bayes Model Experiment On Demo Weather Data set

Based on the experiments conducted using different feature extraction methods (categorical, numerical, and basic) with varying laplace smoothing values, the following insights can be derived:

2.1 Categorical Feature:

The model achieves the highest accuracy of 85% when using a laplace smoothing value of 100.
Generally, increasing the laplace smoothing value leads to improved accuracy up to a certain point.
The accuracy remains relatively stable with values above 40, ranging from 83% to 85%.

2.2 Numerical Feature:

The model achieves the highest accuracy of 98% when using a laplace smoothing value of 0.1.
The accuracy rapidly decreases as the laplace smoothing value increases.
When the laplace smoothing value is 100, the accuracy drops significantly to 60%.

2.3 Basic Feature:

The model achieves the highest accuracy of 99% when using a laplace smoothing value of 0.1.
The accuracy gradually decreases as the laplace smoothing value increases.
Even with higher laplace smoothing values (80 and 100), the accuracy remains relatively high, ranging from 86% to 87%.

Table summarising the insights gained:


Feature Extraction Method
Best Laplace Smoothing Value
Maximum Accuracy
Categorical
0.1
85%
Numerical
0.1
98%
Basic
0.1
99%





      
Based on the insights, it can be concluded that the choice of feature extraction method significantly impacts the performance of the Naive Bayes model on the weather dataset. The categorical and basic features outperform the numerical feature in terms of accuracy. Additionally, a higher laplace smoothing value does not always guarantee better accuracy, as seen in the numerical feature where the accuracy drops with increased smoothing. Overall, the basic feature extraction method with a laplace smoothing value of 0.1 exhibits the highest accuracy among all the experiments conducted.

3)Naive Bayes Experiment On MNIST Data set

Based on the experiments conducted with the Naive Bayes model using different feature extraction methods (Pixel Intensity, HOG, and PCA) on the MNIST digit dataset, the following insights can be drawn:

3.1 Pixel Intensity Feature:

The model accuracy gradually decreases as the Laplace smoothing value increases. This suggests that a smaller smoothing value (0.1) performs better than larger values for this feature extraction method.
The initial accuracy is relatively high (75%) with a low smoothing value (0.1), indicating that the model captures the pixel intensity information effectively. However, as the smoothing value increases, the accuracy drops significantly.

3.2 HOG Feature:

Similar to the Pixel Intensity feature, the accuracy decreases with an increase in the Laplace smoothing value.
The initial accuracy (74%) is reasonably high with a smoothing value of 0.1, indicating that the HOG feature extraction method captures meaningful information about the shape and structure of the digits. However, compared to the Pixel Intensity feature, the overall accuracy for HOG feature is lower across all smoothing values, suggesting that this method may not be as effective for this particular dataset.

3.3 PCA Feature:

The model accuracy remains relatively low across all Laplace smoothing values, indicating that the PCA feature extraction method alone may not be suitable for accurately classifying the MNIST digit dataset using the Naive Bayes model.
The accuracy values remain fairly consistent (around 12-14%) regardless of the smoothing value used. This suggests that the principal components derived from PCA may not capture sufficient discriminatory information for the Naive Bayes model to achieve high accuracy.



Table summarising the insights gained:


Feature Extraction Method
Best Laplace Smoothing Value
Maximum Accuracy
Pixel Intensity
0.1
75%
HOG
0.1
74%
PCA
100
14%



Generalised Insights:

The Pixel Intensity feature exhibits the highest initial accuracy, but it drops significantly with increasing smoothing values.
The HOG feature has lower overall accuracy compared to Pixel Intensity, indicating that it may not capture the digit characteristics as effectively.
The PCA feature, when used alone, results in consistently low accuracy across different smoothing values, suggesting that it may not provide sufficient discriminatory information for the Naive Bayes model.

LOGISTIC REGRESSION


1) Logistic Regression Experiment On MNIST Data set

Based on the experiments on the logistic regression model with different feature extraction methods (Pixel Intensity, HOG, and PCA) for the MNIST digit dataset, we can draw the following insights:

1.1) Pixel Intensity Feature:

As the Laplace smoothing hyperparameter increases from 0.0001 to 1.5, the model accuracy improves consistently.
The model accuracy starts at a relatively low value of 12% for a Laplace smoothing value of 0.0001 and gradually increases to 88% for a Laplace smoothing value of 1. However, there is a slight drop in accuracy to 84% when the Laplace smoothing value is set to 1.5.

1.2) HOG Feature:

The HOG feature extraction method performs better than Pixel Intensity in terms of model accuracy.
The model accuracy increases significantly with increasing Laplace smoothing hyperparameter.
Starting from 4.5% accuracy for a Laplace smoothing value of 0.0001, the accuracy rises to 95% for a Laplace smoothing value of 1.
There is a slight decrease in accuracy to 94.5% for a Laplace smoothing value of 1.5.

1.3 PCA Feature:

The PCA feature extraction method shows moderate performance compared to Pixel Intensity and HOG.
The model accuracy improves as the Laplace smoothing hyperparameter increases.
Starting from 15.5% accuracy for a Laplace smoothing value of 0.0001, the accuracy reaches 87.5% for a Laplace smoothing value of 1.5.
The highest accuracy achieved is 87.5% for a Laplace smoothing value of 1.5.




Table summarising the insights:


Feature Extraction Method
Best Laplace Smoothing Value
Maximum Accuracy
Pixel Intensity
1
88%
HOG
1
95%
PCA
1.5
87.5%



In summary, the HOG feature extraction method achieves the highest accuracy among the three methods, with a maximum accuracy of 95% for a Laplace smoothing value of 1. The Pixel Intensity method shows a gradual increase in accuracy up to 88% for a Laplace smoothing value of 1, while the PCA method reaches a maximum accuracy of 87.5% for a Laplace smoothing value of 1.5.

2) Logistic Regression Experiment On BBC Data set

Based on the experiments with different feature extraction methods and Laplace smoothing hyperparameter values for the logistic regression model on the BBC dataset, the following insights can be observed:
2.1 Bag of Words Feature:

The accuracy of the model improves as the Laplace smoothing hyperparameter increases. At lower values (e.g., 0.0001 and 0.001), the accuracy is quite low, indicating that the model struggles to handle unseen words. However, as the Laplace smoothing value increases (e.g., 0.01, 0.1, 1, 1.5), the accuracy improves significantly and reaches a plateau at around 93.48%. This suggests that the Bag of Words feature benefits from Laplace smoothing to handle out-of-vocabulary words and enhances the model's performance.

2.2 MY Feature:

The accuracy of the model increases with increasing Laplace smoothing hyperparameter values, but the overall performance is lower compared to the Bag of Words feature. At smaller Laplace smoothing values (e.g., 0.0001 and 0.001), the accuracy is quite low, indicating difficulties in handling unseen words. However, as the Laplace smoothing value increases (e.g., 0.01, 0.1, 1, 1.5), the accuracy improves gradually, reaching 45.39% at 1.5. The MY feature seems to benefit from Laplace smoothing but achieves lower accuracy compared to the Bag of Words feature.

2.3 TF-IDF Feature:

The TF-IDF feature extraction method performs differently compared to the previous two methods. The accuracy starts relatively low and increases significantly with increasing Laplace smoothing values. At lower Laplace smoothing values (e.g., 0.0001 and 0.001), the accuracy is quite low. However, as the Laplace smoothing value increases (e.g., 0.01, 0.1, 1, 1.5), the accuracy improves significantly, reaching 55.28% at 1.5. This suggests that the TF-IDF feature benefits greatly from Laplace smoothing and achieves the highest accuracy among the three feature extraction methods.


Table summarising the insights:


Feature Extraction Method
Best Laplace Smoothing Value
Maximum Accuracy
Bag Of Words
0.1
93.48%
MY
1.5
45.39%
TF-IDF
1.5
55.28%



From the table, we can observe that the Bag of Words feature achieves the highest accuracy overall, followed by the TF-IDF feature. The MY feature extraction method lags behind in terms of accuracy. Moreover, all three feature extraction methods benefit from increasing Laplace smoothing values to handle unseen words, although the extent of improvement varies.

3) Logistic Regression Experiment On BBC Data set

Based on the experiments conducted using the logistic regression model with different feature extraction methods (numerical, categorical, and basic), the following insights can be derived from the performance evaluation:

3.1 Numerical Feature Extraction:

The accuracy of the model varies with different learning rates. With a low learning rate (0.0001), the model achieves the highest accuracy of 85%. 
As the learning rate increases, the accuracy gradually decreases. This indicates that a larger learning rate may cause the model to overshoot the optimal solution.
The accuracy drops to 48% when the learning rate is set to 1.5, indicating a significant decline in model performance.

3.2 Categorical Feature Extraction:

The model exhibits higher accuracy compared to the other feature extraction methods across different learning rates.
The accuracy remains consistently high, ranging from 83% to 85% across various learning rates. This suggests that the categorical features are informative and contribute significantly to the model's ability to classify the weather data accurately.
The performance is relatively stable, indicating that the model is less sensitive to changes in the learning rate.

3.3 Basic Feature Extraction:

The basic feature extraction method yields the lowest accuracy compared to the other methods.
The highest accuracy achieved is 82% with a learning rate of 0.0001.
Increasing the learning rate leads to a gradual decline in accuracy.
The accuracy drops to 63% when the learning rate is set to 1.5, indicating a substantial decrease in model performance.


Summary of Performance Insights:

To summarise the insights gained from the experiments, we can generalise the performance of the logistic regression model with different feature extraction methods as follows:


Feature Extraction Method
Optimal Learning Rate
Maximum Accuracy
Numerical
0.0001
85%
Categorical
1.5
85%
Basic
0.0001
82%



From the table and the graph, we can see that the categorical feature extraction method consistently outperforms the other methods, achieving the highest accuracy of 85% across different learning rates. The numerical feature extraction method also demonstrates reasonably good performance, with an accuracy of 85% at a learning rate of 0.0001. However, the basic feature extraction method yields the lowest accuracy, peaking at 82% with a learning rate of 0.0001.
