import os

current_path = os.path.abspath(os.curdir)

COLAB_DARKNET_PATH = '/content/gdrive/MyDrive/darknet'

YOLO_FORMAT_TRAIN_PATH = current_path + '/apple_detection_dataset/apple_dataset/train/images'
YOLO_IMAGE_TRAIN_PATH = current_path + '/apple_detection_dataset/apple_dataset/train/images'
YOLO_FORMAT_TEST_PATH = current_path + '/apple_detection_dataset/apple_dataset/validation/images'
YOLO_IMAGE_TEST_PATH = current_path + '/apple_detection_dataset/apple_dataset/validation/images'

class_count = 0
paths = []
paths_test = []

# classes. names 파일 생성
with open(YOLO_FORMAT_TRAIN_PATH + '/' + 'classes.names', 'w') as names, \
     open(YOLO_FORMAT_TRAIN_PATH + '/' + 'classes.txt', 'r') as txt:
    for line in txt:
        names.write(line)  
        class_count += 1
    print ("[classes.names] is created")

# custom_data.data 파일 생성
with open(YOLO_FORMAT_TRAIN_PATH + '/' + 'custom_data.data', 'w') as data:
    data.write('classes = ' + str(class_count) + '\n')
    data.write('train = ' + COLAB_DARKNET_PATH + '/apple_detection_dataset/apple_dataset/train/images/' + 'train.txt' + '\n')
    data.write('valid = ' + COLAB_DARKNET_PATH + '/apple_detection_dataset/apple_dataset/validation/images/' + 'test.txt' + '\n')
    data.write('names = ' + COLAB_DARKNET_PATH + '/apple_detection_dataset/apple_dataset/train/images/' + 'classes.names' + '\n')
    data.write('backup = backup')
    print ("[custom_data.data] is created")

os.chdir(YOLO_IMAGE_TRAIN_PATH)
for current_dir, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.jpg'):
            train_image_path = COLAB_DARKNET_PATH + '/apple_detection_dataset/apple_dataset/train/images/' + f
            paths.append(train_image_path + '\n')

os.chdir(YOLO_IMAGE_TEST_PATH)
for current_dir, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.jpg'):
            test_image_path = COLAB_DARKNET_PATH + '/apple_detection_dataset/apple_dataset/validation/images/' + f
            paths_test.append(test_image_path + '\n')

# train.txt 파일 생성
with open(YOLO_FORMAT_TRAIN_PATH + '/' + 'train.txt', 'w') as train_txt:
    for path in paths:
        train_txt.write(path)
    print ("[train.txt] is created")

# test.txt 파일 생성
with open(YOLO_FORMAT_TEST_PATH + '/' + 'test.txt', 'w') as test_txt:
    for path in paths_test:
        test_txt.write(path)
    print ("[test.txt] is created")

