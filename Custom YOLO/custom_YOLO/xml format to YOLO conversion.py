# damaged apple dataset 다운로드 URL : https://medium.com/deepquestai/ai-in-agriculture-detecting-defects-in-apples-b246799b329c

from xml.dom import minidom
import os

current_path = os.path.abspath(os.curdir)
print("Current path is {}".format(current_path))
YOLO_FORMAT_PATH = current_path + '/apple_detection_dataset/apple_dataset/train/images'
XML_FORMAT_PATH = current_path + '/apple_detection_dataset/apple_dataset/train/annotations'
file_count = 0
classes={}
classes["apple"] = 0               # XML 파일에서 object의 name(apple) 과 동일한 이름으로 key를 지정해 줘야 함
classes["damaged_apple"] = 1  # XML 파일에서 object의 name(damaged_apple) 과 동일한 이름으로 key를 지정해 줘야 함

# XML cordinates 를 YOLO cordinates로 변환해 주는 함수
def getYoloCordinates(size, box):
    width_ratio = 1.0/size[0]
    height_ratio = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1]-box[0]
    h = box[3]-box[2]
    x = x * width_ratio
    w = w * width_ratio
    y = y * height_ratio
    h = h * height_ratio
    return (x,y,w,h)

os.chdir(XML_FORMAT_PATH)

# classes.txt 파일 생성
with open(YOLO_FORMAT_PATH + '/' + 'classes.txt', 'w') as txt:
    for item in classes:
        txt.write(item + '\n')
        print ("[%s] is added in classes.txt" % item)
    
# 각 이미지에 대한 annotations 파일(.txt) 파일 생성 
for current_dir, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.xml'):
            xmldoc = minidom.parse(file)
            yolo_format = (YOLO_FORMAT_PATH+'/'+file[:-4]+'.txt')

            with open(yolo_format, "w") as f:
    
                objects = xmldoc.getElementsByTagName('object')
                size = xmldoc.getElementsByTagName('size')[0]
                width = int((size.getElementsByTagName('width')[0]).firstChild.data)
                height = int((size.getElementsByTagName('height')[0]).firstChild.data)
    
                for item in objects:
                    name =  (item.getElementsByTagName('name')[0]).firstChild.data
                    if name in classes:
                        class_name = str(classes[name])
                    else:
                        class_name = "-1"
                        print ("[Warning]Class name ['%s'] is not in classes" % name)
    
                    # get bbox coordinates
                    xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                    ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                    xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                    ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                    xml_cordinates = (float(xmin), float(xmax), float(ymin), float(ymax))
                    yolo_cordinates = getYoloCordinates((width,height), xml_cordinates)
    
                    f.write(class_name + " " + " ".join([("%.6f" % a) for a in yolo_cordinates]) + '\n')
                    
            file_count += 1        
            print ("{}. [{}] is created".format(file_count, yolo_format))
                    
