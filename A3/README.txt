Required Libraries:
numpy
os
math
scipy
random
cv2
skimage
matplotlib

Note: In order to run the code, please unzip the Q3_optical_flow.zip and Q4.zip in the current folder.
i.e. The folder that contains this README.txt 

Command to Run:
Question 1: 
python ./code/Q1.py


Question 2:
python ./code/Q2(a).py
python ./code/Q2(b).py

Notice that the result of these might be different since I use random to pick
samples to calculate the affine matrix.

In addition, the version in this code, I assume there are at least half of inliers
in the Q2(a).py and there are 0.3 percentage of inliers in the Q2(b).py since it is 
easier for us to observe the impact of autoupdate on the number of iterations, which 
means, I use 40 iterations in Q2(a).py and 194 iterations in Q2(b).py. Since the
percentage of inliers is lower than 0.5, the probability of the correctness of the 
model should be less than 0.995 in Q2(a).py


Question 3:
python ./code/Q3.py

In this question, I implement the Lucus-Kanade algorithm with iterative refinement
and using this algorithm to detect the motion for each image sequence. If you run the 
code above, it will pick a representative frame for each image sequences and display the 
result for the selected frame, which means 11 images will be shown in total.


Question 4:
python ./code/Q4.py

In this part, if you run the command above, then the HoG result of three images will 
show directly and three text files (1.txt, 2.txt, 3.txt) will be created in the current
folder which contains the result of decriptors. For each file, there will be (M-1)x(N-1)x24
numbers in total.
