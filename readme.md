Code of Graph-based Method for Predicting Individual Bus Trips
# Abstract

  Predicting future trips in the public transportation system for a given user is of great significance for intelligent transportation systems. In this article, we focus on the problem of predicting a user's bus trips in a few future weeks given their past bus trip sequence over a period of time. Existing methods for bus trip prediction mostly treat the trip sequence as time series data, and use methods for time series data processing, such as Markov Model, Hidden Markov Model and Recurrent Neural Network for prediction. However, the association between two bus trips is not always increase as the time difference between them decrease and this character may violate the induction bias of these models, so these prediction methods may have difficulty capturing the inherent patterns of a user's travel behavior. In this article, we treat a user's bus travel behavior as data with a graph structure, rather than traditional time series data and then use mature classification algorithms for graph structure data to predict a user's future bus trips. We compared the prediction result of various prediction methods on our dataset, and found that our proposed method of classification on graph structure yields the best prediction performance. 
