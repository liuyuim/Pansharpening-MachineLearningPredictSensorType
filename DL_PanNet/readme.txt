This is a re-implementation of our paper [1] and for non-commercial use only. You need to install Python (with Tensorflow-GPUand OpenCV packages) to run this code.



Usage:


1. Since the data is relative large to upload, the users can use Matlab to generate their own training data. We give a simple example named "train.mat" in the "/training_data".

2. In the "train.mat", "gt" is ground truth, "ms" is low resolution MS image, "lms" is up-sampled MS image, "pan" is PAN image.

3. The size of "gt", "ms" and "lms" in the "train.mat" are 4-D, indicates batch size, patch height, patch width and channels, respectively.  "pan" is 3-D since it only contains one channel.

4. After generating training data, run 
"train.py" for training and trained models should be generated at "/models".

5. After training, run 
"test.py" to test new data, the result should be generated at "/result".



Some images can be downloaded at:  http://www.digitalglobe.com/product-samples


If this code helps your research, please cite our related paper:

[1] J. Yang, X. Fu, Y. Hu, Y. Huang, X. Ding, J. Paisley. "PanNet: A deep network architecture for pan-sharpening", ICCV, 2017.




Welcome to our homepage: http://smartdsp.xmu.edu.cn/



