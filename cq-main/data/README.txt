DIRECTORIES

'data' directory contains all of the measurement data:

RF:

has sets of Vb2 (y axis) vs Vb1 (x axis). numbers of subdirectories are middle points of the x axis, e.g. 0.25 contains value of Vb1 (x axis) from 0.2 to 0.3, etc. Vb2 (y axis) has range 0-0.4 V for all the sets. This was used to create a big map. 


DC: 

is a different measurement, using DC rather than RF. It does NOT correspond to anything in RF. I just included it as another sample. It contains one set of readouts of current as a function of Vb1 (y axis) and Vb2 (x axis).



DATA FORMAT

refer to the 'data_read.py' for minimum code sample for data loading. 

to load the data, use numpy package. every directory has the raw data in the npz form, it's simply called data.
 
z axis for the RF measurements is either channel 1 or channel 2. Both represent the same physical measurement, however usually channel 1 gives a better result. You can think of channel 1 and channel two as real and imaginary components of some complex number describing signal intensity. All ch1, ch2, magnitude and phase describe the same physical result.

z axis for the DC measurement is just the reading of the multimeter.


