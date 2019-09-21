import sys
import argparse

from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))

from keras_yolo3.yolo import YOLO
from keras_yolo3.yolo_video import call_yolo
from SlidingWindow.SlidingWindow import call_sliding_window
from image_database import populate_database

# main argument parser
argparser = argparse.ArgumentParser(description="Image Processing Model Wrapper")

# create a subparser for the model type
argsubparsers = argparser.add_subparsers(title="Model", dest="model")
argsubparsers.required = True

# add subparsers for YOLO, Sliding Window and Retinanet models
yolo_parser = argsubparsers.add_parser("yolo", help="Invoke Yolo object detection model.")
sliding_window_parser = argsubparsers.add_parser("sliding-window", help="Invoke Sliding Window object detection model.")
retinanet_parser = argsubparsers.add_parser("retinanet", help="Invoke Retinanet object detection model.")


# define arguments for yolo subparser
yolo_parser.add_argument(
             '--model', type=str,
             help='path to model weight file, default ' + YOLO.get_defaults("model_path")
        )
yolo_parser.add_argument(
             '--anchors', type=str,
             help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
        )
yolo_parser.add_argument(
             '--classes', type=str,
             help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
        )
yolo_parser.add_argument(
             '--gpu_num', type=int,
             help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
        )
yolo_parser.add_argument(
             '--image', default=False, action="store_true",
             help='Image detection mode, will ignore all positional arguments'
        ) 
yolo_parser.add_argument(
            "--input", nargs='?', type=str,required=False,default='./path2your_video',
            help = "Video input path"
        )
yolo_parser.add_argument(
            "--output", nargs='?', type=str, default="",
            help = "[Optional] Video output path"
        )


# define arguments for sliding window subparser
sliding_window_parser.add_argument("--image", type=str, help="Path to the image", required=True)
sliding_window_parser.add_argument("--model", type=str, help="Path to the model weights")


# define arguments for retinanet subparser        
retinanet_subparsers = retinanet_parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
retinanet_subparsers.required = True

coco_parser = retinanet_subparsers.add_parser('coco')
coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

pascal_parser = retinanet_subparsers.add_parser('pascal')
pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

csv_parser = retinanet_subparsers.add_parser('csv')
csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

retinanet_parser.add_argument('model',              help='Path to RetinaNet model.')
retinanet_parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
retinanet_parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
retinanet_parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
retinanet_parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
retinanet_parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
retinanet_parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
retinanet_parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
retinanet_parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
retinanet_parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
retinanet_parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')


# class for "dotted dictionary" in python
class DotDict(dict):
    def __getattr__(self, name):
        return self[name]
    
def yolo_model():
    # parse YOLO arguments
    FLAGS = yolo_parser.parse_args(sys.argv[2:])
    FLAGS = DotDict(**vars(FLAGS))
    print("Calling YOLO...")
    # invoke YOLO model for objects detection
    return call_yolo(FLAGS)
    
def sliding_window_model():
    # parse Sliding Window arguments
    FLAGS = sliding_window_parser.parse_args(sys.argv[2:])
    FLAGS = DotDict(**vars(FLAGS)) 
    print("Calling Sliding Window...")
    # invoke Sliding Window model for objects detection
    call_sliding_window(FLAGS)
    
def retinanet_model():
    # parse Retinanet arguments
    FLAGS = retinanet_parser.parse_args(sys.argv[2:])
    FLAGS = DotDict(**vars(FLAGS)) 
    print("Calling RetinaNet...")
    # invoke Retinanet model for objects detection
    return call_retinanet(FLAGS)
    

def main(model, argv=sys.argv[1:]):
    args = argparser.parse_args(argv)
    
    if model == "yolo":
        pass
        # processed_image_info = yolo_model()
    elif model == "sliding-window":
        processed_image_info = sliding_window_model()
        # add processed image information to database
        insert_into_database(processed_image_info)
    elif model == "retinanet":
        pass
        # processed_image_info = retinanet_model()
    
    