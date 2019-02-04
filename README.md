# Sensing-and-Data-Analytics (Biosensor Data Fusion)
We looked at the contributions of spatial frequency and color focusing on rapid categorization of affective versus neutral natural scenes with brief (~33 ms) image presentations that were immediately backward masked. 

Between-subjects, participants viewed unpleasant (bodily mutilation) and neutral natural scenes in either (1) true color, (2) achromatic, or (3) false color {red-green inverted} viewing conditions.

Images were bandpassed at four spatial frequency (SF) ranges, (i) 2-4 {most blurry}, (ii) 4-8, (iii) 8-16, or (iv) 16-32 {least blurry} cycles per image. All subjects saw an equal number of mutilation and neutral images (Image Type) at each SF range.

After each image, subjects categorized the image as either unpleasant or neutral.

The data is contained in SpaceCat(#####_#)xxxx.app#.mat files. The ##### is an individual subject’s ID number and _# is color condition (1=true, 2=achromatic, 3=false). The xxxx is filtering information (which you can ignore), and the # in .app# is 1 to 8 and is indicator of spatial frequency and image type. This 1-8 value is:

.app1: Mutilation 16-32 cpi
.app2: Mutilation 8-16 cpi
.app3 Mutilation 4-8 cpi
.app4: Mutilation 2-4 cpi

.app5: Neutral 16-32 cpi
.app6: Neutral 8-16 cpi
.app7: Neutral 4-8 cpi
.app8: Neutral 2-4 cpi

Each .mat file has three dimensions:

•	Dimension 1: 129 rows, 1 one for each EEG channel. (row 1 = channel 1, etc.)
•	Dimension 2: 1501 data points (500 ms before image onset to 1000 ms after image onset). In other words, point 501 is the time of image onset.
•	Dimension 3: N trials for the given condition

Each subject has 8 .app#.mat files. 

We’re looking at an electrophysiological component called the late positive potential (LPP), in which arousing images typically are associated with higher amplitudes than neutral images. The LPP is typically found at about 500 to 900 ms after image onset. One might start by looking at parietal and occipital channels 52, 53, 54, 59, 60, 61, 62, 66, 67, 71, 72, 75, 76, 77, 78, 79, 84, 85, 86, 91, & 92. The file Net.tiff highlights the location of these sensors. This might serve as some guide for classification, but feel free to look at other regions.


