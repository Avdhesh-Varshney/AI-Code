
# 📜 Project Title:  Bulldozer-Price-Prediction-using-ML

## 🎯 AIM 
This project aims to predict the auction prices of bulldozers using machine learning techniques. The dataset used for this project comes from the Kaggle competition "Blue Book for Bulldozers," which provides historical data on bulldozer sales.

## 📊 DATASET LINK 
<!-- Attach the link of the Dataset. If no, Mention "NOT USED" -->
[Kaggle Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data)


## 📓 KAGGLE NOTEBOOK 
<!-- Attach both links Kaggle URL/ Embed URL public notebook link. -->
[Kaggle Notebook](https://www.kaggle.com/code/nandagopald2004/bulldozer-price-prediction-using-ml)


## ⚙️ TECH STACK 

| **Category**             | **Technologies**                            |
|--------------------------|---------------------------------------------|
| **Languages**            | Python                                      |
| **Libraries/Frameworks** | Scikit Learn,Numpy,Pandas,Matplotlib        |



--- 

## 📝 DESCRIPTION 

Requirement of the Project

The project aims to predict the price of used bulldozers based on various factors such as equipment type, usage hours, manufacturing year, and other relevant parameters. The goal is to develop an accurate pricing model using Machine Learning (ML) techniques.

Why is it Necessary?

The construction and heavy machinery industry heavily relies on the resale of used equipment. Incorrect pricing can lead to financial losses for sellers or overpriced purchases for buyers. A data-driven approach helps ensure fair pricing, improving efficiency in the marketplace.

How is it Beneficial and Used?

1. Helps businesses and individuals estimate bulldozer prices before buying or selling.

2. Assists construction companies in budgeting for equipment procurement.

3. Enables auction houses and dealerships to set competitive and data-backed prices.

4. Reduces reliance on manual estimation, making pricing more transparent and objective.

Approach to the Project

**Data Collection** – Gathered historical sales data of bulldozers, including features like sale date, equipment age, and location.

**Data Preprocessing** – Cleaned missing values, handled categorical variables, and transformed data for ML models.

**Exploratory Data Analysis (EDA)** – Identified key factors influencing bulldozer prices.

**Model Selection & Training** – Implemented and evaluated various ML models such as Random Forest, Gradient Boosting, and Linear Regression.

**Evaluation & Optimization** – Tuned hyperparameters and tested model performance using metrics like RMSE (Root Mean Squared Error).

**Deployment** – Integrated the trained model into a user-friendly interface for real-world use.

This project ensures a more systematic and accurate approach to bulldozer pricing, leveraging ML to enhance decision-making in the heavy equipment industry.


--- 

## 🔍 PROJECT EXPLANATION 

### 🧩 DATASET OVERVIEW & FEATURE DETAILS 


**SalesID**	  unique identifier of a particular sale of a machine at auction

**MachineID**	  identifier for a particular machine;  machines may have multiple sales

**ModelID**	  identifier for a unique machine model (i.e. fiModelDesc)

**datasource**	  source of the sale record;  some sources are more diligent about reporting attributes of the machine than others.  Note that a particular datasource may report on multiple auctioneerIDs.

**auctioneerID**	  identifier of a particular auctioneer, i.e. company that sold the machine at auction.  Not the same as datasource.

**YearMade**	  year of manufacturer of the Machine

**MachineHoursCurrentMeter**	  current usage of the machine in hours at time of sale (saledate);  null or 0 means no hours have been reported for that sale

**UsageBand**	  value (low, medium, high) calculated comparing this particular Machine-Sale hours to average usage for the fiBaseModel;  e.g. 'Low' means this machine has less hours given it's lifespan relative to average of fiBaseModel.

**Saledate**	  time of sale

**Saleprice**	  cost of sale in USD

**fiModelDesc**	  Description of a unique machine model (see ModelID); concatenation of fiBaseModel & fiSecondaryDesc & fiModelSeries & fiModelDescriptor

**fiBaseModel**	  disaggregation of fiModelDesc

**fiSecondaryDesc**	  disaggregation of fiModelDesc

**fiModelSeries**	  disaggregation of fiModelDesc

**fiModelDescriptor**	  disaggregation of fiModelDesc

**ProductSize**	  Don't know what this is 

**ProductClassDesc**	  description of 2nd level hierarchical grouping (below ProductGroup) of fiModelDesc

**State**	  US State in which sale occurred

**ProductGroup**	  identifier for top-level hierarchical grouping of fiModelDesc

**ProductGroupDesc**	  description of top-level hierarchical grouping of fiModelDesc

**Drive_System	machine configuration**;  typcially describes whether 2 or 4 wheel drive

**Enclosure	machine configuration** - does machine have an enclosed cab or not

**Forks	machine configuration** - attachment used for lifting

**Pad_Type	machine configuration** - type of treads a crawler machine uses

**Ride_Control	machine configuration** - optional feature on loaders to make the ride smoother

**Stick	machine configuration** - type of control 

**Transmission	machine configuration** - describes type of transmission;  typically automatic or manual

**Turbocharged	machine configuration** - engine naturally aspirated or turbocharged

**Blade_Extension	machine configuration** - extension of standard blade

**Blade_Width	machine configuration** - width of blade

**Enclosure_Type	machine configuration** - does machine have an enclosed cab or not

**Engine_Horsepower	machine configuration** - engine horsepower rating

**Hydraulics	machine configuration** - type of hydraulics

**Pushblock	machine configuration** - option

**Ripper	machine configuration** - implement attached to machine to till soil

**Scarifier	machine configuration** - implement attached to machine to condition soil

**Tip_control	machine configuration** - type of blade control

**Tire_Size	machine configuration** - size of primary tires

**Coupler	machine configuration** - type of implement interface

**Coupler_System	machine configuration** - type of implement interface

**Grouser_Tracks	machine configuration** - describes ground contact interface

**Hydraulics_Flow	machine configuration** - normal or high flow hydraulic system

**Track_Type	machine configuration** - type of treads a crawler machine uses

**Undercarriage_Pad_Width	machine configuration** - width of crawler treads

**Stick_Length	machine configuration** - length of machine digging implement

**Thumb	machine configuration** - attachment used for grabbing

**Pattern_Changer	machine configuration** - can adjust the operator control configuration to suit the user

**Grouser_Type	machine configuration** - type of treads a crawler machine uses

**Backhoe_Mounting	machine configuration** - optional interface used to add a backhoe attachment

**Blade_Type	machine configuration** - describes type of blade

**Travel_Controls	machine configuration** - describes operator control configuration

**Differential_Type	machine configuration** - differential type, typically locking or standard

**Steering_Controls	machine configuration** - describes operator control configuration

### 🛤 PROJECT WORKFLOW 
The following steps are followed in building the machine learning model:
1. **Data Preprocessing**
   - Handling missing values
   - Feature engineering
   - Encoding categorical variables
   
2. **Exploratory Data Analysis (EDA)**
   - Identifying trends and relationships
   - Visualizing key insights
   
3. **Model Selection and Training**
   - Random Forest Regressor
   - Hyperparameter tuning using RandomizedSearchCV

4. **Model Evaluation**
   - Root Mean Squared Log Error (RMSLE)
   - R² Score



## 📉 MODELS USED AND THEIR EVALUATION METRICS 
<!-- Summarize the models used and their evaluation metrics in a table. -->

|    Model   | Accuracy |  MSE  | R2 Score |
|------------|----------|-------|----------|
| RandomForestRegressor |    95%   | 0.022 |   0.832588403039663   |

--- 

## ✅ CONCLUSION 

The Bulldozer Price Prediction using ML project successfully demonstrates the power of machine learning in estimating the resale price of used bulldozers. By leveraging historical sales data and applying predictive modeling techniques, the project provides a data-driven approach to price estimation, reducing uncertainty and improving decision-making in the heavy equipment market. The final model helps sellers, buyers, and auction houses determine fair market prices, making the process more transparent and efficient.

## Key Learnings

1. **Data Quality Matters** – Handling missing values, feature engineering, and proper data preprocessing significantly impact model performance.

2. **Feature Importance** – Certain factors, such as equipment age, sale date, and operational hours, play a crucial role in price prediction.

3. **Model Selection & Tuning** – Experimenting with different machine learning models (Random Forest, Gradient Boosting, etc.) and optimizing hyperparameters enhances prediction accuracy.

4. **Evaluation Metrics** – Understanding and applying RMSE and other performance metrics helps assess and improve model reliability.

5. **Real-World Deployment** – Preparing a model for deployment requires considering scalability, usability, and integration with business applications.
--- 
