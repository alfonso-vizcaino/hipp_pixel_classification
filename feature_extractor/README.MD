# Pixel feature extraction code
## Source images 
   Source images are loaded from "./images/hippocampal"

---
## Pixel Labels 
## Hippocampus pixel labels are read from files located in the "./labels" folder
### Files with ".txt" extension only contain hippocampus pixels coordinates
#### file data structure:
   * x, y
#### meaning:
   * x : integer. The x coordinate of the pixel
   * y : integer. The y coordinate of the pixel

### Files with ".cvs" extension contain hippocampus label data
####   file data structure:
   * ImageFileName, x, y, HippArea

####   meaning:
   * ImageFileName : string. The name of the file that owns the pixel data
   * x : integer. The x coordinate pixel label of bounding box (ROI) image 
   * y : integer. The y coordinate pixel label of bounding box (ROI) image 
   * HipArea: string. Label that identifies if it's a hippocampus pixel. 
		 Possible values: CA1, CA2, CA3, DG, other. 
		 Note that CA1, CA2, CA3, DG labels are not 100% accurate. They are all treated as hippocampus pixel in code

---
## Feature extraction
   Extracted features are saved in folder "./extracted_features" with name "[image_file_name]_Features.csv"
#### file data structure:
   * ImageFileName, x, y, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, HippArea

#### meaning:
   * ImageFileName : string. The name of the file that owns the pixel data
   * x : integer. The x coordinate pixel label of bounding box (ROI) image 
   * y : integer. The y coordinate pixel label of bounding box (ROI) image
   * f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14 : float. 
		 Numeric pixel characterization
   * HipArea: string. Label that identifies if it's a hippocampus pixel. 
		 Possible values: CA1, CA2, CA3, DG, other. 
		 Note that CA1, CA2, CA3, DG labels are not 100% accurate. Theyare all treated as hippocampus pixel in code

---
## Usage
   compile and make code with the compiler of your choice and run executable. 
### Suggestion:
	$ g++ -o featureExtractor featureExtractor.cpp `pkg-config opencv --cflags --libs`
	$ ./featureExtractor

   Once the program is running:
   * Use keyboard arrow keys (left, right) to navigate through the images
   * Press 's' to save pixel features to a file
   * Press 'r' to realod image
   * Press 'q' to quit program

---
## Dependencies and compiler
	OpenCV version : 3.4.8
	g++ (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2)
	GNU Make 4.2.1

---
## This code was generated in
	Fedora release 31 (Thirty One)
	kernel version 5.6.16-200.fc31.x86_64

