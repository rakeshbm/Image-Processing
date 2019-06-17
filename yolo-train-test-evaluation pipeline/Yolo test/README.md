****Download the model weights and configuration files****

##  For Linux
Run the getModels.sh file from command line to download the needed model files

	sudo chmod a+x getModels.sh
	./getModels.sh


##  For Windows
Download the files from the links given below

https://pjreddie.com/media/files/yolov3.weights
https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
https://github.com/pjreddie/darknet/blob/master/data/coco.names


******Running the code******

Python:
Commandline usage for object detection using YOLOv3
a single image:
	python3 object_detection_yolo.py --image=bird.jpg
a video file:
	python3 object_detection_yolo.py --video=run.mp4

C++:
a single image:
    ./object_detection_yolo.out --image=bird.jpg
a video file:
    ./object_detection_yolo.out --video=run.mp4
