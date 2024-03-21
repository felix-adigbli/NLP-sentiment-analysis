# NLP-sentiment-analysis

Sentiment Analysis for Depression Classification using Tweeter data.




		
Abstract
In recent years, the prevalence of mental health issues, including depression, has escalated, prompting the need for innovative detection and intervention strategies. Social media platforms, particularly Twitter, have become rich sources of data for sentiment analysis, offering insights into public mood and individual mental health states. This report explores the application of sentiment analysis to Twitter data to classify signs of depression, highlighting the importance of early detection and the potential for supporting mental health initiatives.
Sentiment analysis is a natural language processing (NLP) technique used to determine the emotional tone behind words. It is widely used in social media monitoring, customer feedback analysis, and more recently, in mental health studies, particularly for depression classification.
In this report, we compare traditional machine learning models with deep learning models; Visualize the distribution of sentiments; Identify potential sources of bias in the dataset and discuss methods to mitigate them.


1	Introduction

1.1	Objective
The primary aims of this project are twofold:
1.	Sentiment Classification: To execute sentiment analysis on Twitter data with the specific objective of identifying and categorizing depressive sentiments. This involves meticulously analyzing textual data to discern the underlying emotional states conveyed by users, focusing on the presence and extent of depressive moods and indicators.
2.	Model Performance Comparison: To conduct a comprehensive comparison between traditional machine learning techniques and advanced deep learning approaches in the task of sentiment classification. This comparison seeks to assess the relative effectiveness of these methodologies in accurately classifying and differentiating between nuanced sentiment expressions within social media content.


1.2 Relevant methods for Sentiment analysis.
Sentiment analysis is a subfield of natural language processing (NLP) that involves classifying the polarity of a given text as positive, negative, or neutral. Over the years, various methods have been developed to tackle this task. Here is a survey of some of the relevant methods for sentiment analysis:
Lexicon-Based Methods
1.	Dictionary-Based Approaches: These methods rely on a pre-defined list of words with associated sentiment scores. The overall sentiment of a text is determined by aggregating the scores of the individual words.
2.	Corpus-Based Approaches: These methods use statistical techniques to infer the sentiment orientation of words based on their co-occurrence patterns in a large corpus.
Classical Machine Learning Methods
1.	Supervised Learning: This approach involves training a classifier on a labeled dataset where the sentiment of each text is known. Common algorithms include Naive Bayes, Support Vector Machines (SVM), Decision Trees, and Random Forests.
2.	Unsupervised Learning: In this approach, the algorithm tries to discover patterns in the data without using labeled examples. Clustering techniques like K-means or hierarchical clustering can be used to group texts with similar sentiments.
Deep Learning Methods
1.	Recurrent Neural Networks (RNN): RNNs, especially Long Short-Term Memory (LSTM) networks, are effective for sentiment analysis as they can capture the sequential nature of text data.
2.	Convolutional Neural Networks (CNN): CNNs are used for extracting spatial features from text data and have been successfully applied to sentiment analysis tasks.
3.	Transformer Models: Models like BERT (Bidirectional Encoder Representations from Transformers) and its variants (RoBERTa, DistilBERT, etc.) have achieved state-of-the-art performance in sentiment analysis by leveraging self-attention mechanisms.

These methods represent a range of approaches to sentiment analysis, from simple lexicon-based techniques to sophisticated deep learning models. The choice of method depends on a range of factors, including the data's nature, the availability of labeled datasets, and the computational resources at hand. For this report, I will comparing classical machine learning methods and Deep learning methods.


2	Implementation
2.1	Dataset 
The dataset comprises tweets extracted from Kaggle's "Sentimental Analysis for Tweets" dataset, available at https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets/data. The dataset has 10314 unique records. The two relevant fields in the data set are: label (depression result) which specifies if a person has depression? 0 stands for NO and 1 for YES; message to examine which message the Sentimental Analysis needs to be performed on.

2.2	Exploratory Analysis

2.2.1 Distribution of Sentiments in the Dataset
An initial exploratory analysis of the dataset was conducted to understand the distribution of sentiments within the collected tweets. The figure below illustrates the sentiment distribution:


 
Fig: Distribution of Sentiment

This visualization presents a clear disparity in the volume of tweets across different sentiment categories, serving as a foundational understanding of the dataset's composition before delving into more complex analysis.

2.2.2 Kernel Density 
The KD plot indicates that the dataset, primarily contains short tweets with most entries having between 10 to 15 words. There is a right skew, suggesting a presence of some longer texts. These characteristics imply that while the dataset is suitable for analyzing brief expressions of sentiment, the variability in text lengths could pose challenges for sentiment analysis models, especially when dealing with the nuanced sentiments of longer texts. Outliers with significantly higher word counts may require special consideration during preprocessing to ensure the sentiment analysis is accurate across the dataset.

 
Fig: Kernel distribution of words in the dataset


2.2.3 Word Cloud Analysis
In sentiment analysis, word clouds serve as an intuitive method to visually represent the frequency of word occurrence. The size of each word in the word cloud corresponds to its frequency: the larger the word, the more frequently it appears in the dataset.

Depressive Sentiment Word Cloud (Sentiment = 1)
The word cloud for positive sentiments (Fig: The word cloud generated from the dataset with sentiment = 1) vividly illustrates the linguistic footprint of tweets with a positive connotation (depressed). Dominant words in this category are those that express sadness, discontent, or negativity. The prominence of such words provides a quick, perceptible measure of the common vocabulary used in positively charged tweets.
 
Fig: The word cloud generated from the dataset with sentiment = 1(depressed)

Non-Depressed Sentiment Word Cloud (Sentiment = 0)
Conversely, the word cloud for negative sentiments (non-depressed) (Fig: The word cloud generated from the dataset where sentiment = 0) reveals the most common terms associated with negative emotions or experiences in terms of depression. Dominant words in this category are those that convey happiness, approval, and optimism.  The visualization underscores the lexical choices that are recurrently used in the context of negative sentiment, offering insight into the thematic concerns expressed in such tweets.


 
Fig: The word cloud generated from the dataset where sentiment = 0 (not depressed)
2.1 Methodology
2.1.1 Data cleaning Process
Before model induction, an imperative step in sentiment analysis is the thorough cleaning of text data. Our dataset underwent a meticulous cleaning process to ensure the quality and uniformity of the input data for both deep learning and traditional machine learning models. The data cleaning process encompassed the following steps:
•	Contraction Expansion: Contractions were expanded to their full form to maintain uniformity and clarity in the text.
•	HTML Tag Removal: We used BeautifulSoup to strip away any HTML tags, ensuring that only textual content was retained for analysis.
•	URL Removal: All hyperlinks were removed from the tweets to eliminate irrelevant web addresses.
•	Mentions and Hashtag Removal: User mentions starting with '@', and hashtags prefixed with '#' were stripped from the tweets to reduce noise.
•	Retweet Symbol Removal: The 'RT' symbol, indicative of retweets, was removed to maintain the originality of the content.
•	Non-Alphanumeric Character and Punctuation Removal: All characters that were not letters or numbers, as well as punctuation marks, were removed to simplify the text.
•	Digit Removal: Numbers were eliminated from the tweets as they hold no sentiment value.
•	Case Normalization: The entire text corpus was converted to lowercase to ensure that word capitalization did not affect the analysis.
•	Accented Character Normalization: Accented characters were normalized to their ASCII counterparts to maintain consistency in character encoding.
•	Emoji Conversion: Emojis were translated into corresponding text to capture the additional sentiment they convey.
•	Chat Words Translation: Internet slangs and chat words were converted to their full-word equivalents to avoid misinterpretation by the models.
•	Tokenization: Tweets were broken down into individual words or tokens, preparing them for further processing steps.
•	Stop word Removal: Commonly used words that offer no sentiment value were filtered out to focus on words with sentiment potential.
•	Negation Handling: Phrases indicating negation were carefully handled to preserve their impact on the sentiment of the text.
Each of these cleaning steps was applied sequentially, transforming the raw social media text into a refined form ready for the subsequent lemmatization and vectorization processes, thereby laying a solid foundation for the sentiment classification models.


2.1.2 Lemmatization and Vectorization
Lemmatization
The preprocessing phase included a crucial step where tweets were tokenized into constituent words, and then each word was reduced to its canonical form through lemmatization. For this purpose, we utilized the Spacy library, which effectively transforms words to their dictionary form, thereby normalizing the textual data and ensuring consistency across different inflections.

Vectorization
After lemmatization, the text data was transformed into numerical representations—a process essential for the machine learning algorithms to interpret and analyze the text. We employed two primary vectorization techniques:
•	Word Embeddings: Using the Word2Vec algorithm, we generated word embeddings that capture semantic meanings and relationships between words in a high-dimensional space.
•	TF-IDF Vectorization: The TfidfVectorizer was leveraged to convert text into a matrix of TF-IDF features, reflecting the importance of words within the tweets relative to the entire dataset.
Both vectorization methods are pivotal in translating text into a structured form that can serve as input for various machine learning models, facilitating the subsequent sentiment analysis.


2.1.3 Model Implementation
Traditional Machine Learning Models
In our sentiment analysis project, we deployed a range of traditional machine learning algorithms, each tested with two distinct types of vectorization techniques to analyze their performance with distinct feature representations:
•	Naive Bayes, Support Vector Machine (SVM), Random Forest, Logistic Regression, K-Nearest Neighbors (KNN), and XGBOOST: These models were each implemented once with word embeddings derived from Word2Vec and once with features extracted using Term Frequency-Inverse Document Frequency (TF-IDF). This dual approach allowed us to evaluate and compare the efficacy of the models when utilizing raw word embeddings versus statistically derived feature weights.

Deep Learning Models
A variety of deep learning architectures were applied to the task of sentiment classification to capture the complex patterns and dependencies in textual data:
•	Recurrent Neural Networks (RNNs): This category included Simple RNN, Long Short-Term Memory (LSTM) networks, Gated Recurrent Unit (GRU) networks, and Bidirectional LSTMs (BiLSTMs), leveraging their temporal dynamic behavior for sequence data processing.
•	Convolutional Neural Networks (CNNs): Utilized for their ability to capture local features and learn spatial hierarchies, CNNs were employed to assess their performance in text classification tasks.
•	Deep Neural Networks (DNNs): Comprising multiple layers, DNNs were tested for their capacity to learn non-linear relationships in high-dimensional data.
•	Pre-trained Language Models: The project utilized state-of-the-art NLP models, including Bidirectional Encoder Representations from Transformers (BERT) and Robustly Optimized BERT Approach (RoBERTa). For BERT, we employed a pre-trained tokenizer, whereas for RoBERTa, we took a more hands-on approach by training the tokenizer from scratch using the transformers library, aiming to tailor it more closely to the linguistic nuances present in our dataset.
The implementation of these models provided a comprehensive understanding of the trade-offs and benefits of different machine learning paradigms and feature representations in the domain of sentiment analysis.


2.1.3 Evaluation Metrics
Models were assessed based on accuracy, precision, recall, F1-score, and ROC to ensure a comprehensive evaluation of their performance on sentiment classification.
3	Results
3.1 Model Performance Analysis
A range of sentiment analysis models, spanning traditional machine learning to state-of-the-art deep learning and pretrained transformer models, were evaluated on a dataset. The performance was measured based on Accuracy, Precision, Recall, and F1 Score. This report provides an analysis of their performance, outlining key observations and insights. The table show the performance of each of the models.


Model	Accuracy	Precision	Recall	F1 Score:
Naive Bayes with TF-IDF	0.9278	0.9319	0.9278	0.9227
Naïve Bayes with Word2vec	0.8434	0.8329	0.8434	0.8341
Logistic Regression with TF_IDF	0.9695	0.9706	0.9695	0.9686
Logistic Regression with Word2Vec	0.9031	0.9079	0.9031	0.8939
KNN with TF-IDF	0.8076	0.8350	0.8076	0.7438
KNN with Word2vec	0.9418	0.9417	0.9418	0.9418
SVM with TD-IDF	0.9796	0.9801	0.9796	0.9793
SVM with Word2vec	0.9336	0.9363	0.9336	0.9297
XGBOOST with TF-IDF	0.9811	0.9815	0.9811	0.9808
XGBOOST with Word2vec	0.9666	0.9666	0.9666	0.9659
Random Forest with TF-IDF	0.9855	0.9856	0.9855	0.9853
Random Forest with Word2vector	0.9583	0.9593	0.9583	0.9570
Simple RNN	0.9540	0.9537	0.9540	0.9529
LSTM	0.9021	0.9047	0.9021	0.8936
GRU	0.9840	0.9840	0.9840	0.9839
BiLSTMs	0.9845	0.9846	0.9845	0.9843
CNN	0.9845	0.9845	0.9845	0.9843
DNN	0.9796	0.9796	0.9796	0.9795
BERT	0.9855	0.9855	0.9855	0.9853
RoBERTa	0.9811	0.9813	0.9811	0.9808
Table: Accuracy, Precision, Recall, and F1 Scores of the models
The diagram below represents the graphical representation of the scores from the various models.
 
Fig: Line graph of model performance

3.1.1 Traditional vs. Deep Learning Models
The traditional machine learning models with TF-IDF vectorization, particularly Logistic XGBOOST and Random Forest, exhibit impressive performance, rivalling that of deep learning models. Notably, Random Forest with TF-IDF achieved an impressive F1 Score of approximately 0.9853, suggesting a well-balanced precision and recall. Similarly, XGBOOST with TF-IDF displayed a robust F1 Score of 0.9808, marking it as one of the top performers. However, traditional models with Word2Vec did not fare as well as their TF-IDF counterparts, indicating that Word2Vec vectorization may not capture the nuances of sentiment as effectively in this context.
Deep learning models, such as GRU, BiLSTMs, and CNN, demonstrate a slight edge over most traditional models. The GRU, CNN, and BiLSTMs models achieved F1 Scores above 0.983, suggesting excellent balance and generalization capabilities.

3.1.2 Deep Learning vs. Pretrained Models
Among deep learning models, the GRU and BiLSTMs displayed exceptional performance, which is on par with, and in some metrics even surpassing, the more complex pretrained models of RoBERTa. Pretrained models, BERT and RoBERTa, showed formidable performance, with BERT achieving an F1 Score around 0.985 and RoBERTa approximately 0.981. Their superior performance can be attributed to the extensive pretraining on diverse corpora, enabling them to capture deeper linguistic and semantic representations.

3.1.3 Key Observations
•	TF-IDF remains a robust vectorization technique for traditional models, outperforming Word2Vec in this analysis.
•	Pretrained models achieve remarkable scores across all metrics, indicating their advanced capability to understand and classify sentiment in text.
•	Deep learning models that capture sequential information and context, such as GRU and BiLSTMs, show a slight advantage over models that do not consider sequence order, such as DNN.
•	The most advanced models do not always lead in every metric, emphasizing the importance of model selection based on the specific requirements and trade-offs of the task at hand.

3.2 ROC Curve Analysis.
The Receiver Operating Characteristic (ROC) curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. The curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.

 
Fig: ROC curve diagram of the models

1.	The Chance line represents an AUC of 0.5, which is the baseline where the model has no discrimination capacity between positive and negative classes. Any model significantly above this line is considered to have a good measure of separability.
2.	XGBOOST and Random Forest with TF-IDF: Both models exhibit high AUC scores (0.96 and 0.97, respectively), which is indicative of their superior predictive capabilities and confirms that ensemble methods are particularly effective for this task.
3.	The KNN with TF-IDF has the lowest AUC (0.56) among all the models. This could be due to the algorithm's sensitivity to the high-dimensional space that TF-IDF creates, which does not pair well with the KNN's instance-based nature.
4.	Models that can capture complex patterns and sequences in the text data, such as GRU, BiLSTMs, and transformer-based models like BERT and RoBERTa, tend to have higher AUC scores. This reflects the importance of contextual and sequential information in sentiment analysis tasks.
5.	Deep Learning Models (RNNs): GRU, and BiLSTMs all share a high AUC of 0.97, highlighting their effectiveness at capturing sequential information and making accurate class predictions. LSTM is slightly behind with an AUC of 0.79.
6.	Pretrained Transformer Models (BERT and RoBERTa): Both these models have shown remarkably high AUC scores of 0.97 and 0.96, respectively. This indicates their robustness and effectiveness in handling sentiment classification tasks due to their deep contextual understanding.
7.	TF-IDF Vectorization: Models using TF-IDF vectorization show superior ROC performance compared to those using Word2Vec. This could suggest that TF-IDF, which considers not only the frequency of words but also how unique they are to the document, is more effective in this dataset for discriminating between sentiments.
8.	Word2Vec Vectorization: While Naive Bayes and Random Forest models with Word2Vec vectorization have lower AUC scores compared to their TF-IDF counterparts, XGBOOST with Word2Vec performs remarkably well, with an AUC of 0.93. This might indicate that XGBOOST can better capitalize on the dense word embeddings provided by Word2Vec.
9.	Deep Learning Models: The deep learning models, especially the GRU, BiLSTMs, and the transformer-based BERT and RoBERTa, show excellent performance, with AUC scores at or above 0.96. This highlights the strength of deep learning in capturing complex patterns and relationships in the data.
10.	Potential Overfitting: While high AUC scores are desirable, they should also be considered in conjunction with other metrics and the complexity of the model to guard against overfitting.




Challenges Faced

1. Data Quality and Preprocessing
One of the most significant challenges faced was ensuring data quality. Tweets often contain noise—such as URLs, hashtags, user mentions, and emojis—that can obfuscate the sentiment signal. Extensive preprocessing was required to clean and standardize the data, which included:
•	Removing irrelevant characters and symbols.
•	Expanding contractions for consistency.
•	Converting emojis to text to capture sentiment conveyed by these symbols.
•	Translating internet slangs to standard English.
•	Handling negations which are crucial in sentiment analysis.

2. Data Imbalance
The dataset contained an uneven distribution of sentiments, with negative (0) sentiments being more common than positive (1) ones. Such class imbalance can lead to biased model performance, where models over-predict the majority class. Techniques like SMOTE may be employed to address this, but they also introduced synthetic data, which may not always perfectly replicate the complexity of natural language.

3. Vectorization Choices
The choice between Word2Vec and TF-IDF vectorization methods significantly affected model performance. While TF-IDF performed better with traditional models, there was a trade-off between leveraging semantic richness (Word2Vec) and emphasizing keyword frequency (TF-IDF).
4. Model Complexity and Interpretability
Deep learning models, particularly LSTM, GRU, and transformer-based models like BERT and RoBERTa, showed excellent performance but at the cost of increased complexity and reduced interpretability. Such models are often regarded as 'black boxes,' making it difficult to understand the basis of their predictions.
5. Computational Demands
The training and fine-tuning of deep learning models required substantial computational resources. Models like BERT and RoBERTa, while yielding high accuracy, demanded significant memory and processing power, which can be a barrier for deployment in resource-constrained environments.
6. Performance Metric Selection
Determining the appropriate performance metric posed a challenge. While accuracy is the most intuitive metric, it can be misleading in the presence of class imbalance. Therefore, a balance between various metrics, including precision, recall, F1 score, and AUC, was necessary to get a comprehensive view of model performance.
7. Overfitting Concerns
High performance on the test set, especially with complex models, raised concerns about overfitting. Ensuring the models' generalizability to unseen data remains a critical issue, necessitating techniques like cross-validation and regularization.



4.	Potential sources of bias in the dataset and methods to mitigate them.

When conducting sentiment analysis on Twitter data for depression classification, it is important to be aware of potential biases that can affect both the analysis and the outcome of the model. Here are some potential sources of bias in the dataset: 
Language and Expression Bias
Issue: People express sentiments in diverse ways, influenced by cultural, regional, and individual differences. Certain expressions or slang may not be uniformly distributed across the dataset.
Mitigation: Ensure the dataset includes tweets from various demographics and use natural language processing tools capable of understanding context and colloquialisms. Employing multilingual models can help capture a broader range of expressions.
Sampling Bias
Issue: The dataset might not be representative of the general Twitter population, or of those experiencing depression, if it is collected based on specific hashtags or keywords.
Mitigation: Use stratified sampling methods to gather tweets from a wide array of users and topics. Consider including tweets across various times and events to avoid time-related biases.
Labeling Bias
Issue: If the dataset is manually labeled, annotator's biases can influence the sentiment classification. Different annotators may interpret the sentiment of tweets differently.
Mitigation: Develop a clear annotation guideline and conduct training sessions for annotators. Use multiple annotators for each tweet and employ inter-annotator agreement measures to ensure reliability.
Class Imbalance Bias
Issue: If there are significantly more tweets for one sentiment than another, the model might be biased towards the majority class.
Mitigation: Balance the dataset either by oversampling the minority class or under sampling the majority class. Alternatively, adjust the class weights during model training.
Exclusion Bias
Issue: Excluding certain groups, either intentionally or unintentionally, can lead to biased results. For example, not including non-English tweets might ignore non-English speaking users' sentiments.
Mitigation: Include and analyze tweets in different languages and from various user groups. Employ translation services and multilingual models to ensure inclusivity.
Socioeconomic Bias
Issue: Users from different socioeconomic backgrounds might use Twitter differently, affecting the dataset composition.
Mitigation: Ensure the data collection process does not favor certain user groups over others. This might involve analyzing user metadata to assess diversity in the dataset.


 
5.	Conclusions
In our sentiment analysis project, we navigated the intricate process of classifying depressive sentiments from Twitter data, involving data preparation, model training, and evaluation, our evaluation of traditional machine learning models and advanced deep learning models, including pretrained transformers like BERT and RoBERTa, revealed diverse levels of effectiveness. The deep learning models, especially RNNs and pretrained language models, were notably proficient in identifying sentiment subtleties, delivering consistently high performance across metrics like AUC, signifying strong discriminative abilities.

Traditional models using TF-IDF vectorization, such as XGBOOST and Random Forest, also performed admirably. These models proved that a straightforward approach could sometimes match the sophistication of more complex models, offering advantages in interpretability and computational efficiency.

For real-time prediction tasks, simpler models like Random Forest with TF-IDF could be preferable due to their balance of performance and speed. In contrast, deep learning models, capable of modeling the sequence and context of words, are more suited for intricate tasks where such factors heavily influence sentiment. BERT and RoBERTa excel in tasks demanding nuanced language comprehension, though their computational demands could pose challenges for deployment in limited-resource settings.

Overall, the ROC curve analysis underscores the high quality of the dataset and indicates that both traditional and deep learning models are effective for sentiment analysis, with deep learning models having a marginal advantage. The consistently high AUC scores across various models affirm the dataset's integrity and the distinctiveness of sentiment expressions within it.
