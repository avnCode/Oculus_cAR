# Project : Oculus_cAR 

I have implemented the YOLOv4 algorithm to create custom trained weights for desired object detections(in our case Classes detected are: Car, Vehicle, Stop Signs, Person, Animals).
and sliding window technique is merged to detect lanes.

Steps Involved in Implementing Custom trained YOLOv4 Detection in Google CoLab!

1. Build Darknet
2. Perform Detections with Darknet and YOLOv4 on Pre-trained weights
3. Training a Custom YOLOv4 Object Detector in the Cloud
4. Gather and Label Custom Dataset
5. Train Custom Object Detector


Steps Involved in Lane detection!

1. Pre-processing of Image
2. Perspective transform
3. Applying Sliding Windows Technique.
4. Curve fitting
5. Inverse perspective transform
6. Plotting curve on  input image/frame.

**Input Image**
![street](https://user-images.githubusercontent.com/111170719/206979304-101d908e-83f3-4350-919d-fcf4911c17a4.jpg)


**Object Detection through Custom trained YOLOv4**
![Screenshot (362)](https://user-images.githubusercontent.com/111170719/206979259-4e7b8ed1-2519-435c-8311-bc0015284009.png)




**Testing our object detection model on IISc campus:**

Input:

![ezgif-3-1a1ff08eff](https://user-images.githubusercontent.com/111170719/206987964-20dc75e9-9a4e-4221-aa8e-0c7f6b2bfdd4.gif)

Output:

![ezgif-3-d6eb98d887](https://user-images.githubusercontent.com/111170719/206984800-e4be559e-bd91-49ce-8f28-6cfa392046c9.gif)

Link to full Video:
https://user-images.githubusercontent.com/111170719/206981021-ce4624af-9dc6-489a-b2ee-f48bda1bfbbc.mp4




**Testing our object detection model on Random Delhi Highway YouTube video:**

Input:

![test_gdrive_AdobeExpress (2)](https://user-images.githubusercontent.com/111170719/206983433-22e9c83c-df13-4b6a-99c2-568d7797ba1c.gif)

Output:

![test_gdrive_AdobeExpress (4)](https://user-images.githubusercontent.com/111170719/206983842-c600a555-a17f-473b-928a-68f1726dda23.gif)

Link to full Video:
https://user-images.githubusercontent.com/111170719/207027449-07eaaf4d-de20-4d43-bb5b-9dd183e48bdc.mp4




**Testing our Lane detection and object detection model on Random sample video from YouTube:**

Link to full Video:
https://user-images.githubusercontent.com/111170719/206981713-f24840b8-fff4-4e2c-a71f-89946065f8de.mp4



Quantitative Evaluation:


Plot of Loss Curve with Iterations-

![Screenshot 2022-12-12 131058](https://user-images.githubusercontent.com/111170719/206989000-481c5405-fefc-40eb-b5a3-4679bab9d500.jpg)

***NOTE: Due to limitation on GPU usage for training on Google colab the loss curve is discontinous in between***

FUTURE PLANS:
THE MODEL WILL BE TRAINED ON MOTORCYCLE AND TRAFFIC SIGNAL DATASETS ALSO.
