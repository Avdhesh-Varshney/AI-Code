# 📜 Brightness control  
<div align="center">
    <img src="https://user-images.githubusercontent.com/59369441/116009572-1542a280-a638-11eb-9d94-2a2d38b856a5.PNG" />
</div>

## 🎯 AIM 
To develop a real-time brightness control system using hand gestures, leveraging OpenCV and MediaPipe for hand detection and brightness adjustment.


## 📊 DATASET LINK 
No dataset used

## 📓 NOTEBOOK LINK 
<!-- Provide the link to the notebook where you solved the project. It must be only Kaggle public notebook link. -->
[https://drive.google.com/file/d/1q7kraajGykfc2Kb6-84dCOjkrDGhIQcy/view?usp=sharing](https://www.google.chttps://drive.google.com/file/d/1q7kraajGykfc2Kb6-84dCOjkrDGhIQcy/view?usp=sharingom)


## ⚙️ TECH STACK

| **Category**             | **Technologies**                            |
|--------------------------|---------------------------------------------|
| **Languages**            | Python                          |
| **Libraries/Frameworks** | OpenCV, NumPy, MediaPipe, cvzone                    |
| **Tools**                | Jupyter Notebook, Local Python IDE               |


--- 

## 📝 DESCRIPTION 
!!! info "What is the requirement of the project?"
    - The project requires a webcam to capture real-time video and detect hand gestures for brightness control.

??? info "How is it beneficial and used?"
     - Allows users to control screen brightness without physical touch, making it useful for touchless interfaces. 
     - Ideal for applications in smart home systems and assistive technologies. 

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - Identified the need for a touchless brightness control system. 
    - Selected OpenCV for video processing and MediaPipe for efficient hand tracking. 
    - Developed a prototype to calculate brightness based on hand distance. 

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    - OpenCV documentation for video processing. 
    - MediaPipe's official guide for hand tracking. 


--- 

## 🔍 EXPLANATION 

### 🧩 DETAILS OF THE DIFFERENT FEATURES 

#### 🛠 Developed Features 

| Feature Name | Description | Reason   |
|--------------|-------------|----------|
| Hand Detection  | Detects hand gestures in real-time   | To control brightness with gestures |
| Distance Calculation    | Calculates distance between fingers   | To adjust brightness dynamically |
| Brightness Mapping    | Maps hand distance to brightness levels   | Ensures smooth adjustment of brightness |


--- 

### 🛤 PROJECT WORKFLOW 

!!! success "Project workflow"

    ``` mermaid
      graph LR
    A[Start] --> B[Initialize Webcam];
    B --> C[Detect Hand Gestures];
    C --> D[Calculate Distance];
    D --> E[Adjust Brightness];
    E --> F[Display Output];
    ```

=== "Step 1"
- Initialize the webcam using OpenCV.


=== "Step 2"
- Use MediaPipe to detect hands in the video feed.

=== "Step 3"
- Calculate the distance between two fingers (e.g., thumb and index).

=== "Step 4"
- Map the distance to a brightness range.

=== "Step 5"
- Display the adjusted brightness on the video feed.

--- 

### 🖥 CODE EXPLANATION 

=== "Section 1: Import Libraries" 
- Import essential libraries like OpenCV, NumPy, and cvzone's HandDetector for hand tracking.

=== "Section 2: Webcam Initialization" 
- Set up the webcam with predefined width and height.

=== "Section 3: Hand Detection" 
- Use MediaPipe via cvzone's HandDetector to detect hands and calculate distances.


--- 

### ⚖️ PROJECT TRADE-OFFS AND SOLUTIONS 


=== "Trade Off 1" 
    - Real-time processing vs. computational efficiency: Optimized hand detection by limiting the maximum number of detectable hands to 1.


=== "Trade Off 2" 
    - Precision in brightness control vs. usability: Adjusted mapping function to ensure smooth transitions.

--- 

## 🖼 SCREENSHOTS 
??? tip "Working of the model"

    === "Image Topic"
<img width="485" alt="image" src="https://github.com/user-attachments/assets/4b7ea465-a916-4592-9026-9bfa41711763" />



--- 

## ✅ CONCLUSION 

### 🔑 KEY LEARNINGS 
!!! tip "Insights gained from the data" 
- Improved understanding of real-time video processing. 
- Learned to integrate gesture detection with hardware functionalities.

??? tip "Improvements in understanding machine learning concepts" 
- Gained insights into MediaPipe's efficient hand detection algorithms.

--- 

### 🌍 USE CASES
=== "Smart Homes" 
- Touchless brightness control for smart home displays.

=== "Assistive Technologies" 
- Brightness adjustment for users with limited mobility.
