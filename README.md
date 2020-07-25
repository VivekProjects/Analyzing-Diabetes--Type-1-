# Analyzing Diabetes (Type 1) in Women
The cells in our body require glucose for energy and the glucose is made when our stomach breaks down the food. However, in order for the glucose to enter the cell, insulin is required which is produced by our pancreas to bind with the cell and open the pathway for glucose to enter inside the cell. So, according to National Institutes of Health, the lack of energy in the cell due to low insulin is called diabetes. Based on a data sample from Kaggle on diabetes on different women. We used logistic regression to predict if the women in the future are potential victims of diabetes. The data obtained must be preprocessed for applying the algorithms on it. The schema definition and the type of the data must be defined. After these steps, the data must be passed into the Spark platform and converted into dataframe, which must be further converted into a dataset. Then, a suitable machine learning algorithm must be selected. In this case, the logistic regression will help us provide us the output in 0 (Not at Risk for Getting Diabetes) and 1 (At Risk for getting Diabetes).
<p align="center">
  <img src="https://github.com/VivekProjects/Media/blob/master/Diabetes%20Flow%20Chart.png"><br>
  <h1>Result</hi><br>
  <img src="https://github.com/VivekProjects/Media/blob/master/Confusion%20Metrix.png">
</p>
<h1>Result</hi><br>
<p align="center">
  <img src="https://github.com/VivekProjects/Media/blob/master/Confusion%20Metrix.png">
</p>
<p align="center"><b>Based on Area Under the Curve (AUC), our prediction data model was able to predict with 73.44% accuracy.
