# Music Genre Classification Model 

## AIM
<!-- In 1 single line -->
To be able to classify musical data based on the genre they are from.

## DATASET LINK
<!-- Attach the link of the Dataset -->
[GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)
## MY NOTEBOOK LINK
<!-- Mention your notebook link where you have solve it either on Kaggle or Collab Link or Drive Link. -->
[Music Genre Classification Model](https://colab.research.google.com/drive/1j8RZccP2ee5XlWEFSkTyJ98lFyNrezHS?usp=sharing)


## DESCRIPTION
<!-- Properly describe the project. Provide the answer of all the questions,
what is the requirement of the project?, 
why is it necessary?, 
how is it beneficial and used?, 
how did you start approaching this project?, 
Any additional resources used like blogs reading, books reading (mention the name of book along with the pages you have read)?
etc. -->
- What is the requirement of the project?
    - The objective of this research is to develop a precise and effective music genre classification model using Convolutional Neural Networks (CNN), Support Vector Machines (SVM), Random Forest and XGBoost algorithms for the Kaggle GTZAN Dataset Music Genre Classification.

- Why is it necessary?
    - Music genre classification has several real-world applications, including music recommendation, content-based music retrieval, and personalized music services. However, the task of music genre classification is challenging due to the subjective nature of music and the complexity of audio signals.

- How is it beneficial and used?
  - **For User :** Provides more personalised music
  - **For Developers:** A recommendation system for songs that are of interest to the user 
  - **For Business:** Able to charge premium for the more personalised and recommendation services provided


- How did you start approaching this project? (Initial thoughts and planning)
  - Initially how the different sounds are structured. 
  - Learned how to represent sound signal in 2D format on graphs using the librosa library.
  - Came to know about the various features of sound like 
    * Mel-frequency cepstral coefficients (MFCC)
    * Chromagram
    * Spectral Centroid
    * Zero-crossing rate
    * BPM - Beats Per Minute

- Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.).
  - https://scholarworks.calstate.edu/downloads/73666b68n
  - https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data
  - https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8


## EXPLANATION

### DETAILS OF THE DIFFERENT FEATURES
<!-- Elaborate the features as mentioned in the issues, perfoming any googling to learn about the features -->
There are 3 different types of the datasets.

- genres_original
- images_original
- features_3_sec.csv
- feature_30_sec.csv

The features in `genres_original`
['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
Each and every genre has 100 WAV files

The features in `genres_original`
['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
Each and every genre has 100 PNG files

There are 60 features in `features_3_sec.csv`

There are 60 features in `features_30_sec.csv`


### WHAT I HAVE DONE
<!-- Write this in steps not in too much long paragraphs. -->
* Created data visual reprsentation of the data to help understand the data
* Found strong relationships between independent features and dependent feature using correlation.
* Performed Exploratory Data Analysis on data.
* Used different Classification techniques like SVM, Random Forest, 
* Compared various models and used best performance model to make predictions.
* Used Mean Squared Error and R2 Score for evaluating model's performance.
* Visualized best model's performance using matplotlib and seaborn library.

### PROJECT TRADE-OFFS AND SOLUTIONS
<!-- Explain the trade-offs encountered during the project and how you addressed them -->

1. **Trade-off 1**: How do you visualize audio signal
   - **Solution**: 
     - **_librosa_**: It is the mother of all audio file libraries
     - **Plotting Graphs**: As I have the necessary libraries to visualize the data. I started plotting the audio signals
     - **Spectogram**:A spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies with time. When applied to an audio signal, spectrograms are sometimes called sonographs, voiceprints, or voicegrams. Here we convert the frequency axis to a logarithmic one.

2. **Trade-off 2**: Features that help classify the data
   - **Solution**:
     - **Feature Engineering**: What are the features present in audio signals
     - **Spectral Centroid**: Indicates where the ”centre of mass” for a sound is located and is calculated as the weighted mean of the frequencies present in the sound.
     - **Mel-Frequency Cepstral Coefficients**: The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope. It models the characteristics of the human voice.
     - **Chroma Frequencies**: Chroma features are an interesting and powerful representation for music audio in which the entire spectrum is projected onto 12 bins representing the 12 distinct semitones (or chroma) of the musical octave.

3. **Trade-off 3**: Performing EDA on the CSV files
   - **Solution**:
     - **Tool Selection**: Used the correlation matrix on the features_30_sec.csv dataset to extract most related datasets
     - **Visualization Best Practices**: Followed best practices such as using appropriate chart types (e.g., box plots for BPM data, PCA plots for correlations), adding labels and titles, and ensuring readability.
     - **Iterative Refinement**: Iteratively refined visualizations based on feedback and self-review to enhance clarity and informativeness.

4. **Trade-off 4**: Implementing Machine Learning Models
   - **Solution**:
     - **Cross-validation**: Used cross-validation techniques to ensure the reliability and accuracy of the analysis results.
     - **Collaboration with Experts**: Engaged with Music experts and enthusiasts to validate the findings and gain additional perspectives.
     - **Contextual Understanding**: Interpreted results within the context of the music, considering factors such as mood of the users, surrounding, and specific events to provide meaningful and actionable insights.

### LIBRARIES NEEDED
<!-- Mention it in bullet points either in numbering or simple dots -->
- librosa
- matplotlib
- pandas
- sklearn
- seaborn
- numpy
- scipy
- xgboost


### SCREENSHOTS
<!-- Attach the screenshots and images of tree diagram of project approaching,
Visualization and EDA of different features, etc. -->
![img](./assets/Top-5-Player-of-Match.png)
![img](./assets/Top-5-Batsman-Runners.png)
![img](./assets/Top-5-Four-Runners.png)
![img](./assets/Top-5-Overs.png)
![img](./assets/Top-5-Runs.png)
![img](./assets/Top-5-Umpires.png)
![img](./assets/Top-5-Wickets.png)
![img](./assets/Toss-Winners.png)

### MODELS USED AND THEIR ACCURACIES
<!-- If you have used any model, Provide this data in tabular form with,
Accuracy, MSE, R2 Score -->
| Model                        | Accuracy   | 
|------------------------------|------------|
| KNN                          |0.80581     |
| Random Forest                |0.81415     |
| Cross Gradient Booster       |0.90123     |
| SVM                          |0.75409     |

### MODELS COMPARISON GRAPHS 
<!-- Attach the images and screenshots of models accuracy and losses graphs. -->

![img](./assets/RF_Regression_Plot.png)

## CONCLUSION

* Here we can see that Accuracy plots of the different models
* Here, XGB Classifier can predict most accurate results for predicting the Genre of the music

### WHAT YOU HAVE LEARNED

- **Insights gained from the data**:
  - Discovered a new library that help visualize audio signal
  - Discovered new features related to audio like STFT, MFCC, Spectral Centroid, Spectral Rolloff
  - Gained a deeper understanding of the features of different genres of music

- **Improvements in understanding machine learning concepts**:
  - Enhanced knowledge of data cleaning and preprocessing techniques to handle real-world datasets.
  - Improved skills in exploratory data analysis (EDA) to extract meaningful insights from raw data.
  - Learned how to use visualization tools to effectively communicate data-driven findings.

### USE CASES OF THIS MODEL

1. **Application 1: User Personalisation**:
   - **Explanation**: Can be used to provide more personalised music recommendation for users based on their taste in music or the various genres they listen to. This personalisation experience can be used to develop 'Premium' based business models

2. **Application 2: Compatability Between Users**:
   - **Explanation**: Based on the musical taste and the genres they listen we can identify the user behaviour and pattern come with similar users who can be friends with. This increases social interaction within the app.

### HOW TO INTEGRATE THIS MODEL IN REAL WORLD

1. Use API to collect user information
2. Deploy the model using appropriate tools (e.g., Flask, Docker)
3. Monitor and maintain the model in production

### FEATURES PLANNED BUT NOT IMPLEMENTED

- **Feature 1: Real-time Compatability Tracking**:
  - **Description**: Implementing a real-time tracking system to view compatability between users
  - **Reason it couldn't be implemented**: Lack of access to live data streams and the complexity of integrating real-time data processing.

- **Feature 2: Predictive Analytics**:
  - **Description**: Using advanced machine learning algorithms to predict the next song the users is likely to listen to.
  - **Reason it couldn't be implemented**: Constraints in computational resources and the need for more sophisticated modeling techniques that were beyond the current scope of the project.

### YOUR NAME
*Filbert Shawn*

[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/filbert-shawn-1a694a256/)


#### Happy Coding 🧑‍💻
### Show some &nbsp;❤️&nbsp; by &nbsp;🌟&nbsp; this repository!

