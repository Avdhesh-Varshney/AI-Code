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
[https://drive.google.com/file/d/1q7kraajGykfc2Kb6-84dCOjkrDGhIQcy/view?usp=sharing](https://drive.google.com/file/d/1q7kraajGykfc2Kb6-84dCOjkrDGhIQcy/view?usp=sharing)


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

=== "Section 1: Webcam Initialization" 
- The program begins by setting up the webcam to capture frames with a resolution of 640x480 pixels. This ensures consistent processing and visualization of the video stream.

    ```python
    cap = cv2.VideoCapture(0)  
    cap.set(3, 640)  # Set width  
    cap.set(4, 480)  # Set height 
    ```

=== "Section 2: Hand Detection and Brightness Control" 
- Using the `HandDetector` from `cvzone`, the program tracks one hand (maxHands=1). The brightness of the video frame is dynamically adjusted based on the distance between the thumb and index finger.

    ```python
    detector = HandDetector(detectionCon=0.8, maxHands=1)  
    brightness = 1.0  # Default brightness level  
    ```

- The HandDetector detects hand landmarks in each frame with a confidence threshold of 0.8. The initial brightness is set to 1.0 (normal).

    ```python
    hands, img = detector.findHands(frame, flipType=False)  

    if hands:  
        hand = hands[0]  
        lm_list = hand['lmList']  
        if len(lm_list) > 8:  
            thumb_tip = lm_list[4]  
            index_tip = lm_list[8]  
            distance = int(((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5)  
            brightness = np.interp(distance, [20, 200], [0, 1])  
    ```

- The program calculates the distance between the thumb tip (`lmList[4]`) and index finger tip (`lmList[8]`). This distance is mapped to a brightness range of 0 to 1 using np.interp.

=== "Section 3: Brightness Adjustment and Display " 

- The captured frame's brightness is modified by scaling the value (V) channel in the HSV color space according to the calculated brightness level.

    ```python
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness, 0, 255).astype(np.uint8)  
    frame_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  
    cv2.imshow("Brightness Controller", frame_bright)  
    ```

- This technique ensures smooth, real-time brightness adjustments based on the user's hand gestures. The output frame is displayed with the adjusted brightness level.


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
