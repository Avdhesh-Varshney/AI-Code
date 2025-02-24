# Music Genre Classification Model 

### AIM 

To develop a precise and effective music genre classification model using Convolutional Neural Networks (CNN), Support Vector Machines (SVM), Random Forest and XGBoost Classifier algorithms for the Kaggle GTZAN Dataset Music Genre Classification. 

### DATASET LINK 

[https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)

### MY NOTEBOOK LINK 

[https://colab.research.google.com/drive/1j8RZccP2ee5XlWEFSkTyJ98lFyNrezHS?usp=sharing](https://colab.research.google.com/drive/1j8RZccP2ee5XlWEFSkTyJ98lFyNrezHS?usp=sharing)

### LIBRARIES NEEDED 

??? quote "LIBRARIES USED"

    - librosa
    - matplotlib
    - pandas
    - sklearn
    - seaborn
    - numpy
    - scipy
    - xgboost

--- 

### DESCRIPTION 

!!! info "What is the requirement of the project?"
    - The objective of this research is to develop a precise and effective music genre classification model using Convolutional Neural Networks (CNN), Support Vector Machines (SVM), Random Forest and XGBoost algorithms for the Kaggle GTZAN Dataset Music Genre Classification.

??? info "Why is it necessary?"
    - Music genre classification has several real-world applications, including music recommendation, content-based music retrieval, and personalized music services. However, the task of music genre classification is challenging due to the subjective nature of music and the complexity of audio signals.

??? info "How is it beneficial and used?"
    - **For User:** Provides more personalised music
	- **For Developers:** A recommendation system for songs that are of interest to the user 
	- **For Business:** Able to charge premium for the more personalised and recommendation services provided

??? info "How did you start approaching this project? (Initial thoughts and planning)"
	- Initially how the different sounds are structured. 
	- Learned how to represent sound signal in 2D format on graphs using the librosa library.
	- Came to know about the various features of sound like 
    	- Mel-frequency cepstral coefficients (MFCC)
    	- Chromagram
    	- Spectral Centroid
    	- Zero-crossing rate
    	- BPM - Beats Per Minute

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
	- [https://scholarworks.calstate.edu/downloads/73666b68n](https://scholarworks.calstate.edu/downloads/73666b68n)
	- [https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)
	- [https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8](https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8)

--- 

### EXPLANATION 

#### DETAILS OF THE DIFFERENT FEATURES 

  	There are 3 different types of the datasets.

	- genres_original
	- images_original
	- features_3_sec.csv
	- feature_30_sec.csv

- The features in `genres_original`

    ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    Each and every genre has 100 WAV files

- The features in `genres_original`

    ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    Each and every genre has 100 PNG files

- There are 60 features in `features_3_sec.csv`

- There are 60 features in `features_30_sec.csv`

--- 

#### WHAT I HAVE DONE 

=== "Step 1"

    - Created data visual reprsentation of the data to help understand the data

=== "Step 2"

    - Found strong relationships between independent features and dependent feature using correlation.

=== "Step 3"

    - Performed Exploratory Data Analysis on data.

=== "Step 4"

    - Used different Classification techniques like SVM, Random Forest, 

=== "Step 5"

    - Compared various models and used best performance model to make predictions.

=== "Step 6"

    - Used Mean Squared Error and R2 Score for evaluating model's performance.

=== "Step 7"

    - Visualized best model's performance using matplotlib and seaborn library.

--- 

#### PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade Off 1"

    How do you visualize audio signal

     - **Solution**: 

      - **_librosa_**: It is the mother of all audio file libraries
      - **Plotting Graphs**: As I have the necessary libraries to visualize the data. I started plotting the audio signals
      - **Spectogram**:A spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies with time. When applied to an audio signal, spectrograms are sometimes called sonographs, voiceprints, or voicegrams. Here we convert the frequency axis to a logarithmic one.

=== "Trade Off 2"

    Features that help classify the data

     - **Solution**:

      - **Feature Engineering**: What are the features present in audio signals
      - **Spectral Centroid**: Indicates where the ”centre of mass” for a sound is located and is calculated as the weighted mean of the frequencies present in the sound.
      - **Mel-Frequency Cepstral Coefficients**: The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope. It models the characteristics of the human voice.
      - **Chroma Frequencies**: Chroma features are an interesting and powerful representation for music audio in which the entire spectrum is projected onto 12 bins representing the 12 distinct semitones (or chroma) of the musical octave.

=== "Trade Off 3"

    Performing EDA on the CSV files

     - **Solution**:

      - **Tool Selection**: Used the correlation matrix on the features_30_sec.csv dataset to extract most related datasets
      - **Visualization Best Practices**: Followed best practices such as using appropriate chart types (e.g., box plots for BPM data, PCA plots for correlations), adding labels and titles, and ensuring readability.
      - **Iterative Refinement**: Iteratively refined visualizations based on feedback and self-review to enhance clarity and informativeness.

=== "Trade Off 4"

    Implementing Machine Learning Models

     - **Solution**:

      - **Cross-validation**: Used cross-validation techniques to ensure the reliability and accuracy of the analysis results.
      - **Collaboration with Experts**: Engaged with Music experts and enthusiasts to validate the findings and gain additional perspectives.
      - **Contextual Understanding**: Interpreted results within the context of the music, considering factors such as mood of the users, surrounding, and specific events to provide meaningful and actionable insights.

--- 

### SCREENSHOTS 

!!! success "Project workflow"

    ``` mermaid
      graph LR
        A[Start] --> B{Error?};
        B -->|Yes| C[Hmm...];
        C --> D[Debug];
        D --> B;
        B ---->|No| E[Yay!];
    ```

??? tip "Visualizations and EDA of different features"

    === "Harm Perc"
        ![harm _perc](https://github.com/user-attachments/assets/1faab657-812b-453f-bbea-f0ba9a7cfb44)

    === "Sound Wave"
        ![sound _wave](https://github.com/user-attachments/assets/f83fd865-567f-4943-9776-fc4e1223caa3)

    === "STFT"
        ![stft](https://github.com/user-attachments/assets/22d288bf-9063-4593-a3d2-7225b2550807)

    === "Pop Mel-Spec"
        ![pop _mel-_spec](https://github.com/user-attachments/assets/1bee61ef-3fd3-4d21-becc-bf14621824a1)

    === "Blues Mel-Spec"
        ![blues _mel-_spec](https://github.com/user-attachments/assets/fb31fcc4-c40f-4767-bec3-95c520c031ca)

    === "Spec Cent"
        ![spec _cent](https://github.com/user-attachments/assets/b203347d-cd37-42c8-b9e1-f8da59d235db)

    === "Spec Rolloff"
        ![spec _rolloff](https://github.com/user-attachments/assets/e7a468d3-f6e2-4877-b3a7-687a14d8566b)

    === "MFCC"
        ![m_f_c_c](https://github.com/user-attachments/assets/f1eaf291-ecc3-4710-bbbb-286e89b348b3)

    === "Chromogram"
        ![chromogram](https://github.com/user-attachments/assets/fffc9b5c-5466-45f0-a552-b369b44e197c)

    === "Corr Heatmap"
        ![corr _heatmap](https://github.com/user-attachments/assets/6f2afb34-e6c0-4319-a474-d1cbf8631c92)

    === "BPM Boxplot"
        ![b_p_m _boxplot](https://github.com/user-attachments/assets/b8a9be0f-d686-4c12-8157-f5dcf06fcb06)

    === "PCA Scatter Plot"
        ![p_c_a _scattert](https://github.com/user-attachments/assets/66fe7232-1166-4af0-932f-f6ba197fa042)

    === "Confusion Matrix"
        ![conf matrix](https://github.com/user-attachments/assets/f53b1aa8-34e4-4839-bd39-9f6837805b01)

--- 

### MODELS USED AND THEIR ACCURACIES 

| Model                        | Accuracy   | 
|------------------------------|------------|
| KNN                          |0.80581     |
| Random Forest                |0.81415     |
| Cross Gradient Booster       |0.90123     |
| SVM                          |0.75409     |

--- 

#### MODELS COMPARISON GRAPHS 

!!! tip "Models Comparison Graphs"

    === "ACC Plot"
        ![accplot](https://github.com/user-attachments/assets/4d4f3eff-c8f3-4163-9dc8-6cdb3c96ab28)

--- 

### CONCLUSION 

    We can see that Accuracy plots of the different models.
    XGB Classifier can predict most accurate results for predicting the Genre of the music.

#### WHAT YOU HAVE LEARNED 

!!! tip "Insights gained from the data"
    - Discovered a new library that help visualize audio signal
    - Discovered new features related to audio like STFT, MFCC, Spectral Centroid, Spectral Rolloff
    - Gained a deeper understanding of the features of different genres of music

??? tip "Improvements in understanding machine learning concepts"
    - Enhanced knowledge of data cleaning and preprocessing techniques to handle real-world datasets.
    - Improved skills in exploratory data analysis (EDA) to extract meaningful insights from raw data.
    - Learned how to use visualization tools to effectively communicate data-driven findings.

--- 

#### USE CASES OF THIS MODEL 

=== "Application 1"

	**User Personalisation**

     - It can be used to provide more personalised music recommendation for users based on their taste in music or the various genres they listen to. This personalisation experience can be used to develop 'Premium' based business models.

=== "Application 2"

	**Compatability Between Users**

     - Based on the musical taste and the genres they listen we can identify the user behaviour and pattern come with similar users who can be friends with. This increases social interaction within the app.

--- 

#### FEATURES PLANNED BUT NOT IMPLEMENTED 

=== "Feature 1"

    - **Real-time Compatability Tracking**

    - Implementing a real-time tracking system to view compatability between users.

=== "Feature 1"

    - **Predictive Analytics**

    - Using advanced machine learning algorithms to predict the next song the users is likely to listen to.

