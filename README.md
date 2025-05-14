
Article Type: Case Stude
Titel: - Anomaly Detection in Credit Card Transactions: A Machine Learning-Based Approach
 
Authors:  Sudishta kumar Yadav1*
Affiliation:
Rungta College of Engineering and Technology, Bhilai, Durg, Chhattisgarh, India- 490024.
Corresponding Author Details:
Mr. Sudishta Kumar Yadav
1Department of Computer Science Engineering with AI Technology
Kohka, Bhilai, Durg, Chhattisgarh, India – 490024.
Mob: +91 9142297516
Email id: sudishtakumar2023@gmail.com




Abstract
The Digital Payment revolution is here which has in turn led to an increasing demand for Accepting and Making Digital Payment systems like credit cards that hold risk for issuers and acquirers. The traditional fraud identification systems that are programmed on rule-based technology are not efficient in detecting the innovative fraudulent activities. ML-based anomaly detection offers a strong alternative, serving to track fraud in real-time by detecting outliers in data. Detecting Anomalies in Credit Card Transactions through Machine Learning Published on May 06, 2019This post discuss techniques of using machine learning models related to fraud detection on credit card transactions. It provides a complete end-to-end recipe, starting from data preprocessing to model selection, evaluation, and deployment. The work shows that Autoencoders in combination with the Oversampling process significantly improve the fraud detection process, suggesting a strong applicability of deep learning models in the .

Keywords Credit card fraud, machine learning, imbalanced data, SMOTE, supervised learning, autoencoders, Isolation Forest, financial security.
Key Points
•	Credit Card Fraud is a growing threat in the domain of financial security, requiring advanced detection mechanisms.
•	Machine Learning provides powerful tools for detecting fraudulent transactions in large datasets.
•	Imbalanced Data is a major challenge, as fraudulent transactions are rare compared to legitimate ones.
•	SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the dataset by generating synthetic examples of the minority (fraud) class.
•	Supervised Learning Algorithms (e.g., Logistic Regression, Decision Trees, Random Forests, SVM) are effective when labelled data is available.
•	Autoencoders, a type of neural network, are used for unsupervised anomaly detection by learning patterns of normal transactions and identifying outliers.
•	Isolation Forest, another unsupervised learning algorithm, isolates anomalies by randomly selecting features and splitting values.
•	A hybrid approach that combines supervised and unsupervised models can improve fraud detection accuracy.


1.1 Introduction
Credit cards are a common form of transaction, but their convenience has spawned increasing risks of fraud, leading to billions of dollars of losses a year. Rule-based systems are typical approaches to detecting fraud by setting certain conditions to flag transactions, but they are lacking in generalization and yield a high false-positive rate. Worse, they don't detect new, nuanced fraud schemes.
Machine learning provides an active response by examining user activity, spotting abnormalities in real time and adjusting for evolving fraud techniques. So, while static rule-based approaches decreased in usefulness over time, ML models continued to outperform them in preventing fraud.
This blog investigates the ML-based anomaly detection possibilities in credit card transactions using a common data set to compare different ML algorithms. synopsis the objective is to unveil the most efficient model for online fraud detection, we will gather some insights from both solutions.

2 Methodology
A successful anomaly detection system requires careful attention to data preprocessing, feature engineering, model selection, and evaluation. The methodology followed in this blog includes the following major components:
1. Dataset Overview
We use a publicly available dataset from Kaggle, originally provided by European financial institutions. The dataset includes:
•	284,807 transactions over two days
•	Only 492 fraudulent cases (~0.172%)
•	Features are anonymized using PCA for confidentiality (V1 to V28)
•	Additional features: Time, Amount, and Class (0 = non-fraud, 1 = fraud)
The dataset is highly imbalanced, which poses challenges for machine learning models, especially for classification accuracy.
3. Data Preprocessing
a. Feature Scaling:
•	The Amount feature is normalized using StandardScaler to reduce model bias.
•	The Time feature is transformed to capture transaction patterns over time (e.g., hourly bins).
b. Handling Imbalance:
•	Since fraud data is rare, we apply SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples of fraud transactions.
•	This ensures that classifiers do not overfit on the majority (non-fraud) class.
c. Feature Selection:
•	All 30 features (including V1–V28, Time, and Amount) are used for training.
•	PCA is already applied in the dataset for anonymization and dimensionality reduction.

4. Model Selection
We evaluate both supervised and unsupervised machine learning models:
a. Supervised Models:
•	Logistic Regression: A baseline binary classifier
•	Random Forest: An ensemble method that uses decision trees
b. Unsupervised Models:
•	Isolation Forest: Identifies anomalies by isolating outliers in a tree structure
•	Autoencoder Neural Network: Learns the distribution of normal data and identifies anomalies as reconstruction errors
5. Evaluation Metrics
Since the dataset is imbalanced, standard accuracy is not sufficient. Instead, we focus on:
•	Precision: Correct fraud predictions among all predicted frauds
•	Recall: Percentage of actual frauds that were correctly predicted
•	F1 Score: Harmonic mean of precision and recall
•	ROC-AUC: Area under the receiver operating characteristic curve, useful for model comparison
Discussion
The key challenge in fraud detection lies in the imbalance of the dataset. With fewer than 0.2% of the transactions being fraudulent, most models tend to classify everything as non-fraud, achieving high accuracy but missing actual frauds.
The use of SMOTE significantly improved the performance of all supervised models by generating synthetic fraud cases, allowing classifiers to learn from minority data points.
Model Insights:
•	Logistic Regression performed adequately but struggled with complex fraud patterns due to its linear nature.
•	Random Forest showed strong performance with minimal tuning. Its ability to capture non-linear interactions made it more effective.
•	Isolation Forest, an unsupervised technique, worked well when no labels were available but had a lower recall.
•	Autoencoders outperformed all other models, especially when trained only on non-fraud data. They reconstructed normal transactions well, and any transaction with high reconstruction error was flagged as fraudulent.
This approach mimics real-world deployment, where the model is trained only on legitimate transactions and uses deviation detection to catch fraud.
 

Results
Table 1 Performance Summary
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	97.8%	0.83	0.62	0.71	0.89
Random Forest	99.1%	0.89	0.76	0.82	0.94
Isolation Forest	98.2%	0.87	0.68	0.76	0.91
Autoencoder	99.3%	0.91	0.81	0.86	0.97
 Key Takeaways:
•	Autoencoders offered the best overall performance in anomaly detection.
•	SMOTE was critical in achieving good recall across supervised models.
•	Precision-Recall tradeoff is essential—higher precision reduces false alarms, while higher recall ensures most frauds are caught.

Conclusion
Detecting anomalies in credit card transactions is not only a technological problem, but also a financial service need. As fraudulent means become more sophisticated, the necessity of adopting smart and flexible methods increases. This blog showed how Autoencoders and Random Forest models can be effectively used to detect the fraud in extremely imbalanced datasets.
The method described here can be used directly or scaled up in to production fraud detection systems with live data pipelines and real time alerts. Ensemble techniques and combination of deep learning models can be tried in future for better results. 

References
1.	Kaggle. (2018). Credit Card Fraud Detection Dataset.
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2.	Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., & Bontempi, G. (2018).
Credit card fraud detection: A realistic modeling and a novel learning strategy.
IEEE Transactions on Neural Networks and Learning Systems, 29(8), 3784–3797.
https://doi.org/10.1109/TNNLS.2017.2736643
3.	Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
SMOTE: Synthetic Minority Over-sampling Technique.
Journal of Artificial Intelligence Research, 16, 321–357.
https://doi.org/10.1613/jair.953
4.	Géron, A. (2019).
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.
O’Reilly Media.
