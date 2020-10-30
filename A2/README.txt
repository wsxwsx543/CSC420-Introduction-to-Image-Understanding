Required Library:
sys
argparse
matplotlib
math
numpy 
numba (I use this to accelerate my code or it will take extremely long time to run, 
       but this library is not on the teaching lab)
cv2 (I use cv2 to calculate the convolution of the image and grad image which mentioned in Piazza that 
     we are allowed to use external library to do the simple calculation. This can decrease the running 
     time of my program significantly. This library is also not on the teaching lab)

Assignment Structure
A2-|---ex1.jpg
   |---ex2.jpg
   |---ex3.jpg
   |---uc1.jpg
   |---uc2.jpg
   |---README.txt
   |---code-|---resize.py
            |---CornerDetection.py

Command use to run:
1. Seam Carving:
python3 resize.py PATH/TO/IMAGE desiredRow desiredColumn
please use "python3 resize.py -h" (without quote) to see details of arguments.

It might takes you minutes to run the code.

To reproduce my result:
python3 resize.py ../ex1.jpg 968 957
python3 resize.py ../ex2.jpg 861 1200
python3 resize.py ../ex3.jpg 870 1200

2. Corner Detection
python3 CornerDetection.py PATH/TO/IMAGE 
please use "python3 CornerDetection.py -h" (without quote) to see details of other arguments.

It might takes you 2 minutes to run the code.

To reproduce my results:
python3 CornerDetection.py ../uc1.jpg --windowSize 5 --windowSigma 1 --threshold 7000
python3 CornerDetection.py ../uc2.jpg --windowSize 5 --windowSigma 1 --threshold 8000
python3 CornerDetection.py ../uc1.jpg --windowSize 5 --windowSigma 15 --threshold 7000
python3 CornerDetection.py ../uc2.jpg --windowSize 5 --windowSigma 15 --threshold 8000