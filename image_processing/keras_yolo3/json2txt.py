# This script would transform the annotation file from labelme into json format
# Source files are in the format '%04d.jpg' 
# Output format is img_file,x1,y1,x2,y2,class_name

import json
import glob

class_idx=['sign positive','window positive','door positive']

class_name={'sign positive':'0','window positive':'1','door positive':'2'}

csv_out = open('testdata.txt', 'w')
json_path='testdata/annotation/*.json'
for jsonfile in glob.glob(json_path):
    print(jsonfile)
    data = json.load(open(jsonfile,'r'))

    # img_file,x1,y1,x2,y2,class_name
    name = data['annotation']['filename']
    csv_out.write('enhanced/train_images/' + name + ' ')
    if isinstance(data['annotation']['object'],list):
        for objects in data['annotation']['object']:
            print(objects)
            if objects['name'] in class_idx:
                box=objects['bndbox'] 
                x1, y1, x2, y2 = box['xmin'],box['ymin'],box['xmax'],box['ymax']
                csv_out.write(str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',')
                csv_out.write(class_name[objects['name']] + ' ')
    else:
        objects=data['annotation']['object']
        if objects['name'] in class_idx:
            box=objects['bndbox']
            x1, y1, x2, y2 = box['xmin'],box['ymin'],box['xmax'],box['ymax']
            csv_out.write(str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',')
            csv_out.write(class_name[objects['name']] + ' ')
    csv_out.write('\n')


csv_out.close()