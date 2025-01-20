# Black and White Image Colorizer  

### AIM 
Colorization of Black and White Images using OpenCV and pre-trained caffe models.

### PRE-TRAINED MODELS
[colorization_deploy_v2.prototxt](https://github.com/richzhang/colorization/blob/caffe/models/colorization_deploy_v2.prototxt) - 
[colorization_release_v2.caffemodel](https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1) - 
[pts_in_hull.npy](https://github.com/richzhang/colorization/blob/caffe/resources/pts_in_hull.npy)

### NOTEBOOK LINK 

[Colab Notebook](https://colab.research.google.com/drive/1w5GbYEIsX41Uh8i_5q7c8Nh0y5UOpBGb)

### LIBRARIES NEEDED 

??? quote "LIBRARIES USED"

    - numpy
    - cv2

--- 

### DESCRIPTION 

!!! info "What is the requirement of the project?"

    - The project aims to perform colorization of black and white images.
    - It involves in showcase the capabilities of OpenCV's DNN module and caffe models.
    - It is done by processing given image using openCV and use Lab Color space model to hallucinate an approximation of how colorized version of the image "might look like".

??? info "Why is it necessary?"

    - It helps preserving historical black-and-white photos. 
    - It can be used adding color to grayscale images for creative industries.  
    - It acts an advancing computer vision applications in artistic and research fields.

??? info "How is it beneficial and used?"

    - **Personal use :** It helps in restoring old family photographs.  
    - **Cultural and Political :** it also enhances grayscale photographs of important historic events for modern displays. 
    - **Creativity and Art  :** it improves AI-based creative tools for artists and designers.  

??? info "How did you start approaching this project? (Initial thoughts and planning)"

    - **Initial approach** : reading various research papers and analyze different approaches on how to deal with this project.
    - Identified Richzhang research paper on the title : Colorful Image colorization.
    - Did some research on pre-trained models for image colorization.  
    - Understood OpenCV's DNN module and its implementation.  
    - Experimented with sample images to test model outputs. 

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    
    - [Richzhang's Colorful Image Colorization](https://richzhang.github.io/colorization)
    - [Lab Color space](https://www.xrite.com/blog/lab-color-space)
    - [openCV Documentation ](https://pypi.org/project/opencv-python/)

--- 

### EXPLANATION

#### WHAT I HAVE DONE 

=== "Step 1"

    Initial data exploration and understanding:
    
      - Load the grayscale input image.
      - Load pre-trained caffe models using openCV dnn module.

=== "Step 2"

    Data cleaning and preprocessing:
    
      - Preprocess image to normalize and convert to LAB color space.
      - Resize image for the network.
      - Split L channel and perform mean subtraction.
      - Predict ab channel from the input of L channel.

=== "Step 3"

    Feature engineering and selection:
    
      - Resize predicted ab channel's volume to same dimension as our image.
      - Join L and predicted ab channel.
      - Convert image from Lab back to RGB.

=== "Step 4"

    Result : 
    
    - Resize and Show the Original and Colorized image.

--- 

#### PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade Off 1"

    - Computational efficiency vs. color accuracy.  
    - **Solution : **Used optimized preprocessing pipelines to reduce runtime. 

=== "Trade Off 2"

    - Pre-trained model generalization vs. custom training.  
    - **Solution : **Choose the pre-trained model for faster implementation and reliable results.  

--- 

### SCREENSHOTS 

!!! success "Project structure or tree diagram"

    ``` mermaid
      graph LR  
        A[Load Grayscale Image] --> B[Preprocess Image];  
        B --> C[Load Pre-trained Model];  
        C --> D[Predict A and B Channels];  
        D --> E[Combine with L Channel];  
        E --> F[Convert to RGB];  
        F --> G[Display/Save Colorized Image];
    ```

??? tip "Visualizations of results"

    === "Original Image"
        ![Original Image](https://github.com/user-attachments/assets/98a68022-eb8a-4e2e-b87a-edf5b8a392fa)
    
    === "Colorized Image"
        ![Colorized Image](https://github.com/user-attachments/assets/181f585e-a2de-4bf5-aead-1c3a56ac7f8e)

    === "Result"
        ![result](https://github.com/user-attachments/assets/6bc14754-e097-4093-95df-4826cd0bae85)

--- 

### CONCLUSION 

#### KEY LEARNINGS 

!!! tip "Insights gained from the data"

    - **Color Space : **LAB color space facilitates colorization tasks.  
    - **Pre-trained Models :** Pre-trained models can generalize across various grayscale images.

??? tip "Improvements in understanding machine learning concepts"

    - **OpenCV : **Enhanced knowledge of OpenCV's DNN module.  
    - **Caffe Models : **Usage of pre-trained models.
    - **Image Dimensionality : **Understanding how Image can be manipulated.

??? tip "Challenges faced and how they were overcome"
    
    - **Color Space Conversion : **Initial difficulties with LAB to RGB conversion; resolved using OpenCV documentation. 

--- 

#### USE CASES 

=== "Application 1"

    **Image Restoration**
    
      - Restoring old family photographs to vivid colors.

=== "Application 2"

    **Creative Industries**
    
      - Colorizing artistic grayscale sketches for concept designs.
