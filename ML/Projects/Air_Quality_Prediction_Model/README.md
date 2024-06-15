
  

# **Air Quality Prediction**

  

## **AIM**

- Brief approach for this project from a basic level upto highest possible accuracy.


## **DATASET**

https://www.kaggle.com/rohanrao/air-quality-data-in-india

  

## **DESCRIPTION**

  

The main aim of the project is to use 3-4 algorithms to implement the models and compare all the algorithms to find out the best fitted algorithm for the model by checking the accuracy score.

  

## **WHAT I DID**

* Analyzed the data and found insights such as correlation, missing values and plotted different plots and compared them with pre and post covid times.
* Next trained model with algorithms:
>	* SVM
>	* Random Forest
>	* XGBoost
* In this XGBoost performed the best with 100% accuracy 
  

## **MODELS USED**

> 1. RBF SVM : SVM performs well on classification problems when size of dataset is not too large. Support Vector Machine can also be used as a regression method, maintaining all the main features that characterize the algorithm (maximal margin).
> 2. Random Forest : It **provides higher accuracy through cross validation**. Random forest regressor will handle the missing values and maintain the accuracy of a large proportion of data. If there are more trees, it won't allow over-fitting trees in the model.
> 3. XGBoost : XGBoost is **a library for developing fast and high performance gradient boosting tree models**. XGBoost achieves the best performance on a range of difficult machine learning tasks.

**LIBRARIES NEEDED**

* Numpy
* Pandas
* Matplotlib
* scikit-learn
* xgboost
* seaborn
* missingno
* chart_studio
* cufflinks
  
  

## **VISUAL PLOTS**


![](/ML/Projects/Air_Quality_Prediction_Model/IMAGES/most_polluted_cities_pre_covid.jpg)


![](/ML/Projects/Air_Quality_Prediction_Model/IMAGES/most_polluted_cities_post_covid.jpg)



### **Evaluation Result** 

| Model                      | Accuracy (%)         |
|----------------------------|----------------------|
|Support Vector Machine (SVM)| 96.12                |
| Random Forest              | 99.92                |
| XGBoost                    | 98.67                |



## **CONCLUSION**

  

We investigated the data, checking for data unbalancing, visualizing the features, and understanding the relationship between different features. We made a comparision of data between pre covid times (between 2015 to 2019) and Post covid times (after 2020). We then investigated three predictive models.
We saw we had imbalanced dataset. This will cause data imbalance problem. In order to overcome this problem we use the technique called SMOTE(Synthetic Minority Oversampling Technique). This approach solve this problem by oversample the examples in the minority class.

We predicted with models SVM, Random Forest and XGBoost and found accuracy of 96.15%, 99.89% and 100% on test dataset.

We also concluded that :
1. Vehicular pollution contents are more related to air quality index.
2. Delhi is the most polluted city in terms of vehicular pollution contents.
3. Ahmadabad is the most polluted city in terms of industrial pollution content.
4. After COVID19 pandemic there is gradual dicrease in vehicular pollution contents, industrial pollution content.
5. Extra Gradient Boost classifier 100% accurately classify the target variable.
  
### Contributor : 
 *Harsh Raj*

 *Avdhesh-Varshney* (Mentor)

### Connect with me:
[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harsh-raj-58921728b/) [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/HarshRaj29004)
