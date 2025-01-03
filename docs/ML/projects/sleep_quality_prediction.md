<h1>Sleep Quality Prediction</h1>

<h2>AIM</h2>
<p>To predict sleep quality based on lifestyle and health factors.</p>

---

<h2>DATASET LINK</h2>
<p>
    <a href="https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset">
        Sleep Health and Lifestyle Dataset
    </a>
</p>

---

<h2>DESCRIPTION</h2>

<h3>What is the requirement of the project?</h3>
<ul>
    <li>This project aims to predict the quality of sleep using various health and lifestyle metrics. Predicting sleep quality helps individuals and healthcare professionals address potential sleep-related health issues early.</li>
</ul>

<h3>Why is it necessary?</h3>
<ul>
    <li>Sleep quality significantly impacts physical and mental health. Early predictions can prevent chronic conditions linked to poor sleep, such as obesity, heart disease, and cognitive impairment.</li>
</ul>

<h3>How is it beneficial and used?</h3>
<ul>
    <li><strong>Individuals:</strong> Assess their sleep health and make lifestyle changes to improve sleep quality.</li>
    <li><strong>Healthcare Professionals:</strong> Use the model as an auxiliary diagnostic tool to recommend personalized interventions.</li>
</ul>

<h3>How did you start approaching this project? (Initial thoughts and planning)</h3>
<ul>
    <li>Researching sleep health factors and existing literature.</li>
    <li>Exploring and analyzing the dataset to understand feature distributions.</li>
    <li>Preprocessing data for effective feature representation.</li>
    <li>Iterating over machine learning models to find the optimal balance between accuracy and interpretability.</li>
</ul>

<h3>Mention any additional resources used</h3>
<ul>
    <li><strong>Research Paper:</strong> <a href="https://www.researchgate.net/publication/355188118_Long_Short_Term_Memory-based_Models_for_Sleep_Quality_Prediction_from_Wearable_Device_Time_Series_Data">Analyzing Sleep Patterns Using AI</a></li>
    <li><strong>Public Notebook:</strong> <a href="https://www.kaggle.com/code/alexandermk04/sleep-quality-prediction-with-96-accuracy/notebook">Sleep Quality Prediction with 96% Accuracy</a></li>
</ul>

---

<h2>LIBRARIES USED</h2>
<ul>
    <li>pandas</li>
    <li>numpy</li>
    <li>scikit-learn</li>
    <li>matplotlib</li>
    <li>seaborn</li>
    <li>joblib</li>
    <li>flask</li>
</ul>

---

<h2>EXPLANATION</h2>

<h3>DETAILS OF THE DIFFERENT FEATURES</h3>

<table>
    <thead>
        <tr>
            <th>Feature Name</th>
            <th>Description</th>
            <th>Type</th>
            <th>Values/Range</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Gender</td>
            <td>Respondent's gender</td>
            <td>Categorical</td>
            <td>[Male, Female]</td>
        </tr>
        <tr>
            <td>Age</td>
            <td>Respondent's age</td>
            <td>Numerical</td>
            <td>Measured in years</td>
        </tr>
        <tr>
            <td>Sleep Duration (hours)</td>
            <td>Hours of sleep per day</td>
            <td>Numerical</td>
            <td>Measured in hours</td>
        </tr>
        <tr>
            <td>Physical Activity Level</td>
            <td>Daily physical activity in minutes</td>
            <td>Numerical</td>
            <td>Measured in minutes</td>
        </tr>
        <tr>
            <td>Stress Level</td>
            <td>Stress level on a scale</td>
            <td>Numerical</td>
            <td>1 to 5 (low to high)</td>
        </tr>
        <tr>
            <td>BMI Category</td>
            <td>Body Mass Index category</td>
            <td>Categorical</td>
            <td>[Underweight, Normal, Overweight, Obese]</td>
        </tr>
        <tr>
            <td>Systolic Blood Pressure</td>
            <td>Systolic blood pressure</td>
            <td>Numerical</td>
            <td>Measured in mmHg</td>
        </tr>
        <tr>
            <td>Diastolic Blood Pressure</td>
            <td>Diastolic blood pressure</td>
            <td>Numerical</td>
            <td>Measured in mmHg</td>
        </tr>
        <tr>
            <td>Heart Rate (bpm)</td>
            <td>Resting heart rate</td>
            <td>Numerical</td>
            <td>Beats per minute</td>
        </tr>
        <tr>
            <td>Daily Steps</td>
            <td>Average number of steps per day</td>
            <td>Numerical</td>
            <td>Measured in steps</td>
        </tr>
        <tr>
            <td>Sleep Disorder</td>
            <td>Reported sleep disorder</td>
            <td>Categorical</td>
            <td>[Yes, No]</td>
        </tr>
    </tbody>
</table>

---

<h2>WHAT I HAVE DONE</h2>

<h4>Step 1: Exploratory Data Analysis</h4>
<ul>
    <li>Summary statistics</li>
    <li>Data visualization for numerical feature distributions</li>
    <li>Target splits for categorical features</li>
</ul>

<h4>Step 2: Data Cleaning and Preprocessing</h4>
<ul>
    <li>Handling missing values</li>
    <li>Label encoding categorical features</li>
    <li>Standardizing numerical features</li>
</ul>

<h4>Step 3: Feature Engineering and Selection</h4>
<ul>
    <li>Merging features based on domain knowledge</li>
    <li>Creating derived features such as "Activity-to-Sleep Ratio"</li>
</ul>

<h4>Step 4: Modeling</h4>
<ul>
    <li>Model trained: Decision Tree</li>
    <li>Class imbalance handled using SMOTE</li>
    <li>Metric for optimization: F1-score</li>
</ul>

<h4>Step 5: Result Analysis</h4>
<ul>
    <li>Visualized results using confusion matrices and classification reports</li>
    <li>Interpreted feature importance for tree-based models</li>
</ul>

---

<h2>MODELS USED AND THEIR ACCURACIES</h2>

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Accuracy (%)</th>
            <th>F1-Score (%)</th>
            <th>Precision (%)</th>
            <th>Recall (%)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Decision Tree</td>
            <td>74.50</td>
            <td>75.20</td>
            <td>73.00</td>
            <td>77.50</td>
        </tr>
    </tbody>
</table>

---


<h2>CONCLUSION</h2>

<h3>WHAT YOU HAVE LEARNED</h3>

<div>
    <h4>Insights gained from the data</h4>
    <ul>
        <li>Sleep Duration, Stress Level, and Physical Activity are the most indicative features for predicting sleep quality.</li>
    </ul>
</div>

<div>
    <h4>Improvements in understanding machine learning concepts</h4>
    <ul>
        <li>Learned and implemented preprocessing techniques like encoding categorical variables and handling imbalanced datasets.</li>
        <li>Gained insights into deploying a machine learning model using Flask for real-world use cases.</li>
    </ul>
</div>

<div>
    <h4>Challenges faced and how they were overcome</h4>
    <ul>
        <li>Managing imbalanced classes: Overcame this by using SMOTE for oversampling the minority class.</li>
        <li>Choosing a simple yet effective model: Selected Decision Tree for its interpretability and ease of deployment.</li>
    </ul>
</div>

---

<h3>USE CASES OF THIS MODEL</h3>

<div>
    <h4>Application 1</h4>
    <p>
        A health tracker app can integrate this model to assess and suggest improvements in sleep quality based on user inputs.
    </p>
</div>

<div>
    <h4>Application 2</h4>
    <p>
        Healthcare providers can use this tool to make preliminary assessments of patients' sleep health, enabling timely interventions.
    </p>
</div>

---

<h3>FEATURES PLANNED BUT NOT IMPLEMENTED</h3>

<div>
    <h4>Feature 1</h4>
    <p>
        Advanced models such as Random Forest, AdaBoost, and Gradient Boosting were not implemented due to the project's focus on simplicity and interpretability.
    </p>
</div>

<div>
    <h4>Feature 2</h4>
    <p>
        Integration with wearable device data for real-time predictions was not explored but remains a potential enhancement for future work.
    </p>
</div>



