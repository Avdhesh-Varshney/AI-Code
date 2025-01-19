# Counting Bicep Reps


### AIM 
To track and count bicep curls in real time using computer vision techniques with OpenCV and Mediapipe's Pose module.

### DATASET LINK 
This project does not use a specific dataset as it works with real-time video from a webcam.


### NOTEBOOK LINK 
[https://drive.google.com/file/d/13Omm8Zy0lmtjmdHgfQbraBu3NJf3wknw/view?usp=sharing](https://drive.google.com/file/d/13Omm8Zy0lmtjmdHgfQbraBu3NJf3wknw/view?usp=sharing)


### LIBRARIES NEEDED

??? quote "LIBRARIES USED"

    - OpenCV
    - Mediapipe
    - NumPy

--- 

### DESCRIPTION 

!!! info "What is the requirement of the project?"
    - The project aims to provide a computer vision-based solution for tracking fitness exercises like bicep curls without the need for wearable devices or sensors. 

??? info "Why is it necessary?"
    - Helps fitness enthusiasts monitor their workouts in real time.
    - Provides an affordable and accessible alternative to wearable fitness trackers.

??? info "How is it beneficial and used?"
    - Real-time feedback on workout form and repetition count.
    - Can be extended to other fitness exercises and integrated into fitness apps

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - Explored Mediapipe's Pose module for pose landmark detection.
    - Integrated OpenCV for video frame processing and real-time feedback.
    - Planned the logic for detecting curls based on elbow angle thresholds.

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    - Mediapipe official documentation.
    - OpenCV tutorials on video processing. 


--- 

### EXPLANATION 

#### DETAILS OF THE DIFFERENT FEATURES 
    - Pose Estimation: Utilized Mediapipe's Pose module to detect key landmarks on the human body.
    - Angle Calculation: Calculated angles at the elbow joints to determine curl movement.
    - Rep Tracking: Incremented rep count when alternating between full curl and relaxed positions.
    - Real-Time Feedback: Displayed the remaining curl count on the video feed.


--- 

#### PROJECT WORKFLOW 

=== "Step 1"
    Initial setup:
- Installed OpenCV and Mediapipe.
- Set up a webcam feed for video capture.  


=== "Step 2"
    Pose detection:
- Used Mediapipe's Pose module to identify body landmarks.


=== "Step 3"
    Angle calculation:
- Implemented a function to calculate the angle between shoulder, elbow, and wrist.


=== "Step 4"
    Rep detection:
- Monitored elbow angles to track upward and downward movements.


=== "Step 5"
    Real-time feedback:
- Displayed the remaining number of curls on the video feed using OpenCV.


=== "Step 6"
    Completion:
- Stopped the program when the target reps were completed or on manual exit.


--- 

#### PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade Off 1"
    - Accuracy vs. Simplicity:
        - Using elbow angles alone may not handle all body postures but ensures simplicity.
        - Solution: Fine-tuned angle thresholds and added tracking for alternating arms.

=== "Trade Off 2"
    - Real-Time Performance vs. Model Complexity:
        - Mediapipe's lightweight solution ensured smooth processing over heavier models.

--- 

### SCREENSHOTS 
!!! success "Project workflow"

    ```mermaid  
    graph LR  
    A[Webcam Feed] --> F[Enter No of Biceps Reps]
    F --> B[Mediapipe Pose Detection]  
    B --> C[Elbow Angle Calculation]  
    C --> D[Rep Count Decrement]  
    D --> E[Real-Time Update on Frsame]  
    ```  

--- 

### CONCLUSION 

#### KEY LEARNINGS 

!!! tip "Insights gained from the data"
    - Real-time video processing using OpenCV.
    - Pose detection and landmark analysis with Mediapipe.

??? tip "Improvements in understanding machine learning concepts"
    - Understanding geometric computations in pose analysis.
    - Effective use of pre-trained models like Mediapipe Pose.

??? tip "Challenges faced and how they were overcome"
    - Challenge: Handling incorrect postures.
        - Solution: Fine-tuning angle thresholds.

--- 

#### USE CASES
=== "Application 1"

    **Personal Fitness Tracker**  
    - Helps users track their workouts without additional equipment.  

=== "Application 2"

    **Fitness App Integration**  
    - Can be integrated into fitness apps for real-time exercise tracking.  
