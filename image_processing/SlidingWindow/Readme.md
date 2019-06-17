1. SlidingWindow.py has the sliding window algorithm implemented which uses the vgg-weight1.h5 to load the model.
2. vggagain.py has the algorithm I have followed to train my model on the classes - door, window, traffic lights and building.
3. The vgg-weight1.h5 contains the weights of the model I trained.
4. To run the file- Python SlidingWindow.py -i <imagename.jpg>

Output:-
1. The file with the object detected is saved as "Window0.jpg".
2. The json file is created as "<imagename_jpg>.json".
3. The detected objects are cropped and saved as "img{0...n}.jpg".
