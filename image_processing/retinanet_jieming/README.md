# Image Extraction using Retinanet



Keras implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár. Revised from keras-retinanet and specified for USC Project Minion's image extraction.

## Installation

1) Clone this repository.
2) Ensure numpy is installed using `pip install numpy --user`
3) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
4) Alternatively, you can run the code directly from the cloned  repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.\

## Data preparation

The data is inside data/ folder. Images are grabbed from google street view images and labeled by labelme.

Json2csv.py can transform the json files from labelme into csv file as the input of the model. By specifying the number of training split and validation split, the script can generate two csv file which contains the path to the image file and its bounding box with the label.

class.csv defines the classes name and its related label.



## Testing

The testing script is by running eval.sh. If you are using your own testing or validation images, you need to change the csv file and class file. 

--score-threshold here means that the output will show the image larger than this threshold.

--iou-threshold means the model will do non-max-suppression using this value.

You can specify the output path using --save-path.

My model trained on only "sign" class can be downloaded here: https://drive.google.com/file/d/1uWsZqPjpdSnZH5LvtH60Hjgr838HasCa/view?usp=sharing



## Training

Training configure can be specified under train.sh.

--weights is the pretrained weight for resnet101, the weight can be downloaded here https://drive.google.com/file/d/1RmT4fsTWYaDxBCJaqXIAbDtuLBIqvq4M/view?usp=sharing. 

Train.csv, val.csv and class.csv should also be changed for different input datat.

The script will automatically save the model into /snapshot folder and this can also be changed.













