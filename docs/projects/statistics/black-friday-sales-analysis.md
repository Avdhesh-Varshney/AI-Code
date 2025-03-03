# 📜 Exploratory Data Analysis

<div align="center">
    <img src="https://images.app.goo.gl/mf9Dw8GbrQdCsGNq6" alt='' />
</div>

## 🎯 AIM 

To analyze the Black Friday sales dataset, understand customer purchasing behavior, identify trends, and generate insights through data visualization and statistical analysis.

## 📊 DATASET LINK 

[https://www.kaggle.com/datasets/rajeshrampure/black-friday-sale/data](https://www.kaggle.com/datasets/rajeshrampure/black-friday-sale/data)

## 📓 KAGGLE NOTEBOOK 

[https://www.kaggle.com/code/kashishkhurana1204/exploratory-data-analysis-eda](https://www.kaggle.com/code/kashishkhurana1204/exploratory-data-analysis-eda)

??? Abstract "Kaggle Notebook"

    <iframe 
        src="https://www.kaggle.com/code/kashishkhurana1204/exploratory-data-analysis-eda" 
        height="600" 
        style="margin: 0 auto; width: 100%; max-width: 950px;" 
        frameborder="0" 
        scrolling="auto" 
        title="bpl-analysis">
    </iframe>

## ⚙️ TECH STACK 

| **Category**             | **Technologies**                            |
|--------------------------|---------------------------------------------|
| **Languages**            | Python                                      |
| **Libraries/Frameworks** | Matplotlib, Pandas, Seaborn, Numpy          |
| **Tools**                | Github, Jupyter, VS Code, Kaggle            |

--- 

## 📝 DESCRIPTION 

!!! info "What is the requirement of the project?"
    - Understanding customer purchasing behavior during Black Friday Sales.
    - Identifying trends in product sales and demographics.
    - Performing statistical analysis and data visualization.

??? info "How is it beneficial and used?"
    - Helps businesses in decision-making for better marketing strategies.
    - Identifies key customer demographics for targeted advertising.
    - Provides insights into which products perform well in sales.

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - I was thinking about a project that helps businesses in decision-making for better marketing strategies.
    - I searched for relevant datasets on Kaggle that fulfill my project requirements.
    - I found the Black Friday Sales dataset which is a perfect fit for my project.
    - I started by understanding the dataset and its features.
	- Data Cleaning: Handled missing values and corrected data types.
    - Data Exploration: Analyzed purchasing patterns by customer demographics.
    - Statistical Analysis: Derived insights using Pandas and Seaborn.
    - Data Visualization: Created visual graphs for better understanding.

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    - [https://www.kaggle.com/datasets/rajeshrampure/black-friday-sale/data](https://www.kaggle.com/datasets/rajeshrampure/black-friday-sale/data)
    

--- 

## 🔍 PROJECT EXPLANATION 

### 🧩 DATASET OVERVIEW & FEATURE DETAILS 

??? example "📂 BlackFriday.csv"

    - The dataset contains transaction records of Black Friday Sales.

| Feature Name               | Description                              | Datatype   |
|----------------------------|------------------------------------------|------------|
| User_ID                    | Unique identifier for customers          | int64      |
| Product_ID                 | Unique identifier for products           | object     |
| Gender                     | Gender of customer                       | object     |
| Age                        | Age group of customer                    | object     |
| Occupation                 | Occupation category                      | int64      |
| City_Category              | City category (A, B, C)                  | object     |
| Stay_In_Current_City_Years | Duration of stay in the city             | object     |
| Marital_Status             | Marital status of customer               | int64      |
| Purchase                   | Amount spent by the customer             | int64      |
  

--- 

### 🛤 PROJECT WORKFLOW 

!!! success "Project workflow"

    ``` mermaid
    graph LR
    A[Data Collection] --> B[Data Cleaning]
    B --> C[Exploratory Data Analysis]
    C --> D[Data Visualization]
    D --> E[Conclusion & Insights]
    ```

=== "Step 1"
    **Data Loading and Preprocessing**

        - Importing the dataset using Pandas and checking the initial structure.

        - Converting data types and renaming columns for consistency.

=== "Step 2"
    **Handling Missing Values and Outliers**

        - Identifying and filling/removing missing values using appropriate techniques.

        - Detecting and treating outliers using boxplots and statistical methods.

=== "Step 3"
    **Exploratory Data Analysis (EDA) with Pandas and Seaborn**

        - Understanding the distribution of key features through summary statistics.

        - Using groupby functions to analyze purchasing behavior based on demographics.

=== "Step 4"
    **Creating Visualizations for Insights**

        - Using Seaborn and Matplotlib to generate bar charts, histograms, and scatter plots.

        - Creating correlation heatmaps to identify relationships between variables.

=== "Step 5"
    **Identifying Trends and Patterns**

        - Analyzing seasonal variations in sales data.

        - Understanding the impact of age, gender, and occupation on purchase amounts.

=== "Step 6"
    **Conclusion and Final Report**

        - Summarizing the key findings from EDA.

        - Presenting actionable insights for business decision-making.

--- 

### 🖥 CODE EXPLANATION 

=== "plotgraph() function"

    ```py
    gender_sales = df.groupby('Gender')['Purchase'].sum()

    plt.figure(figsize=(6, 6))
    plt.pie(gender_sales, labels=gender_sales.index, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
    plt.title('Sales by Gender', fontsize=16)

    plt.show()

    age_gender_sales = df.groupby(['Age', 'Gender'])['Purchase'].sum().unstack()

    age_gender_sales.plot(kind='bar', figsize=(12, 6))
    plt.title('Sales by Age Group and Gender')
    plt.xlabel('Age Group')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.legend(title='Gender')
    plt.show()
    ```

    - It displays the visualization graph of sales by age group and gender.

--- 

### ⚖️ PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade Off 1"
    - **Trade-off:** High computational time due to large dataset.
    - **Solution:** Used optimized Pandas functions to enhance performance.

=== "Trade Off 2"
    - **Trade-off:** Data Imbalance due to customer distribution.
    - **Solution:** Applied statistical techniques to handle biases.

--- 

## 🖼 SCREENSHOTS 

!!! tip "Visualizations and EDA of different features"

    === "Sales by Age Group and Gender"
        ![sales_by_age_group_and_gender](https://github.com/Kashishkh/-Exploratory-Data-Analysis-/blob/main/Screenshot%202025-02-28%20182656.png)

    === "Sales by City Category"
        ![sales_by_city_category](https://github.com/Kashishkh/-Exploratory-Data-Analysis-/blob/main/Screenshot%202025-02-28%20182735.png)

    === "Sales by Occupation"
        ![sales_by_occupation](https://github.com/Kashishkh/-Exploratory-Data-Analysis-/blob/main/Screenshot%202025-02-28%20182720.png)

    === "Purchase Behavior via Marital Status"
        ![Purchase_behavior_via_marital_status](https://github.com/Kashishkh/-Exploratory-Data-Analysis-/blob/main/Screenshot%202025-02-28%20182621.png)

    === "Sales by Age Group"
        ![sales_by_age _group](https://github.com/Kashishkh/-Exploratory-Data-Analysis-/blob/main/Screenshot%202025-02-28%20182744.png)

    === "Sales by Gender"
        ![sales_by_gender](https://github.com/Kashishkh/-Exploratory-Data-Analysis-/blob/main/Screenshot%202025-02-28%20182706.png)

--- 

## ✅ CONCLUSION 

### 🔑 KEY LEARNINGS 

!!! tip "Insights gained from the data"
    - Majority of purchases were made by young customers.

    - Men made more purchases compared to women.

    - Electronic items and clothing were the top-selling categories.

--- 

### 🌍 USE CASES 

=== "Application 1"
	**Retail Analytics**
     - Helps businesses understand customer behavior and target promotions accordingly.

=== "Application 2"
	**Sales Forecasting**
     - Provides insights into seasonal trends and helps in inventory management.

### 🔗 USEFUL LINKS 

=== "GitHub Repository"
    - [https://github.com/Kashishkh/-Exploratory-Data-Analysis-](https://github.com/Kashishkh/-Exploratory-Data-Analysis-)