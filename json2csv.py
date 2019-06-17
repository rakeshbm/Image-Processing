# This script would transform the annotation file from labelme into json format
# Source files are in the format '%04d.jpg' 
# Output format is img_file,x1,y1,x2,y2,class_name

import json

image_total = 60
csv_out = open('label.csv', 'w')

for i in range(1, image_total + 1):
    data = json.load(open('%04d.json' % i))
    # print(data)

    # img_file,x1,y1,x2,y2,class_name
    name = data['imagePath']
    

    for shape in data['shapes']:
        csv_out.write('images/' + name + ',')
        [x1, y1], [x2, y2] = shape['points']
        x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        csv_out.write(str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',')
        csv_out.write(shape['label'] + '\n')

csv_out.close()