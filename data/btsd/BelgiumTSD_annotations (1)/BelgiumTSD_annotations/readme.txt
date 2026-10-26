
 BelgiumTS for Detection 
=========================

This folder contains annotations for BelgiumTS dataset for Detection.
We maintain the same train/test split of BelgiumTS and BelgiumTSC.

Thanks to Alexander Chigorin we have a few douzens extra annotations for circular traffic signs in the testing split (these are not in the default set of annotations of BelgiumTS).

Superclasses
============
We ease the task for the practitioners and provide these annotations with a structure similar to the German Traffic Sign Detection Benchmark (GTSDB)
by defining the following superclasses (based on shape and color):

-1.   undefined traffic sign (TS) class type; %-1
0.    other defined TS = [all the other defined TS ids besides the following 11]; %0
1.    triangles = [2 3 4 7 8 9 10 12 13 15 17 18 22 26 27 28 29 34 35];   %1 (corresponds to Danger superclass in GTSDB)
2.    redcircles  = [36 43 48 50 55 56 57 58 59 61 65]; %2 (corresponds to Prohibitory superclass in GTSDB)
3.    bluecircles = [72 75 76 78 79 80 81];    %3 (corresponds to Mandatory superclass in GTSDB)
4.    redbluecircles = [82 84 85 86];    %4
5.    diamonds = [32 41]; %5
6.    revtriangle = [31]; %6
7.    stop = [39]; %7
8.    forbidden = [42];%8
9.    squares = [118 151 155 181]; %9
10.    rectanglesup  = [37,87,90,94,95,96,97,149,150,163];%10
11.    rectanglesdown= [111,112]; %11

Training dataset
================
Corresponds to the BelgiumTS and BelgiumTSC training data.

The training annotations are found in:
  BTSD_training_GT.txt   
  BTSD_training_GTclear.txt  -- short format

The training annotations are divided as follows:
[annotations]([superclass])
        2040 (-1)
        1705 (0)
         765 (1)
         891 (2)
        1026 (3)
         455 (4)
         291 (5)
         252 (6)
          43 (7)
         375 (8)
         414 (9)
         540 (10)
          54 (11)
Total 8851 training annotations in 5905 images

Supplementary are 16045 background images (the same from BelgiumTS).

Testing dataset
===============
Corresponds to the BelgiumTS and BelgiumTSC testing data.

The testing annotations are found in:
  BTSD_testing_GT.txt
  BTSD_testing_GTclear.txt   -- short format

The testing annotations are divided as follows:
[annotations]([superclass])
         990 (-1)
         871 (0)
         580 (1)
         795 (2)
         570 (3)
          78 (4)
         116 (5)
         184 (6)
          45 (7)
          61 (8)
         115 (9)
         221 (10)
           3 (11)
Total 4629 testing annotations in 3101 images

Supplementary are 583 background images to assess false positives (the same from BelgiumTS).

Annotation format
=================
The format of the annotation files is as follows:

[camera]/[image];[x1];[y1];[x2];[y2];[class id];[superclass id];[pole id];[number on pole];[camera number];[frame number];[class label]

Example:
"01/image.000945.jp2;1066.23;383.12;1109.31;429.08;80;3;4;1;1;945;D7;"

camera: 01
image: image.000945.jp2
x1,y1,x2,y2: 1066.23;383.12;1109.31;429.08
class id: 80
superclass: 3
pole id: 4
number on pole: 1
camera number: 1
frame number: 945
class label: D7

The *clear.txt have only the following information:
[camera]/[image];[x1];[y1];[x2];[y2];[class id];[superclass id];

Example:
"01/image.000945.jp2;1066.23;383.12;1109.31;429.08;80;3;"

camera: 01
image: image.000945.jp2
x1,y1,x2,y2: 1066.23;383.12;1109.31;429.08
class id: 80
superclass: 3

Reference
=========

Please include a reference to:
[1] Radu Timofte, Karel Zimmermann, and Luc Van Gool, "Multi-view traffic sign detection, recognition, and 3D localisation", Journal of Machine Vision and Applications, December 2011

[2] Radu Timofte, Markus Mathias, Rodrigo Benenson, and Luc Van Gool, "Traffic sign recognition - How far are we from the solution?", Proc. of IJCNN, August 2013


Updates
=======
April 25, 2013 - created by Radu Timofte
