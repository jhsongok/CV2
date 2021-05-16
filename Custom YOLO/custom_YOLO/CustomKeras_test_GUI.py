import numpy as np
import cv2
import tensorflow as tf
from tkinter import *
from tkinter import filedialog
import tkinter.scrolledtext as tkst

min_confidence = 0.5
width = 1024
height = 0
show_ratio = 1.0
file_name = "./custom/fruit01.jpg"
classes_name = "./custom/classes.txt"
weight_name = "./custom_model/fruit_custom.h5"
title_name = 'Tenserflow Keras Custom Data Prediction'
classes = []
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
read_image = None
CW = 32
CH = 32
CD = 3

# Load TF model, classes
model = tf.keras.models.load_model(weight_name)
with open(classes_name, 'r') as txt:
    for line in txt:
        name = line.replace("\n", "")
        classes.append(name)

def selectWeightFile():
    global weight_name
    global model
    weight_name =  filedialog.askopenfilename(initialdir = "./custom_model",title = "Select Model file",filetypes = (("keras model files","*.h5"),("all files","*.*")))
    weight_path['text'] = weight_name
    model = tf.keras.models.load_model(weight_name)

def selectClassesFile():
    global classes_name
    global classes
    classes_name =  filedialog.askopenfilename(initialdir = "./custom",title = "Select Classes file",filetypes = (("text files","*.txt"),("all files","*.*")))
    classes_path['text'] = classes_name
    classes = []
    with open(classes_name, 'r') as txt:
        for line in txt:
            name = line.replace("\n", "")
            classes.append(name)
            
def selectFile():
    global read_image
    file_name =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    read_image = cv2.imread(file_name)
    file_path['text'] = file_name
    detectAndDisplay()

def detectAndDisplay():
    global read_image
    global classes
    test_images = []
    
    h, w = read_image.shape[:2]
    height = int(h * width / w)
    img = cv2.resize(read_image, (width, height))
    
    box = cv2.selectROI("Select Resign Of interest and Press Enter or Space key", img, fromCenter=False,
            showCrosshair=True)
    startX = int(box[0])
    startY = int(box[1])
    endX = int(box[0]+box[2])
    endY = int(box[1]+box[3])
    image = cv2.resize(img[startY:endY, startX:endX]
                       , (CW,CH), interpolation = cv2.INTER_AREA)
    test_images.append(image)
    # convert the data and labels to NumPy arrays
    test_images = np.array(test_images)
    # scale data to the range of [0, 1]
    test_images = test_images.astype("float32") / 255.0
    
    result = model.predict(test_images)
    result_number = np.argmax(result[0])
    print(result, result_number)
    print("%s : %.2f %2s" % (classes[result_number], result[0][result_number]*100, '%'))
        
    text = "{}: {}%".format(classes[result_number], round(result[0][result_number]*100,2))
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(img, (startX, startY), (endX, endY),
        colors[result_number], 2)
    cv2.putText(img, text, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[result_number], 2)
    cv2.imshow("Prediction Output", img)
    cv2.waitKey(0)

        
main = Tk()
main.title(title_name)
main.geometry()

label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)

weight_title = Label(main, text='Weight')
weight_title.grid(row=1,column=0,columnspan=1)
weight_path = Label(main, text=weight_name)
weight_path.grid(row=1,column=1,columnspan=2)
Button(main,text="Select", height=1,command=lambda:selectWeightFile()).grid(row=1, column=3, columnspan=1, sticky=(N, S, W, E))

classes_title = Label(main, text='Classes')
classes_title.grid(row=2,column=0,columnspan=1)
classes_path = Label(main, text=classes_name)
classes_path.grid(row=2,column=1,columnspan=2)
Button(main,text="Select", height=1,command=lambda:selectClassesFile()).grid(row=2, column=3, columnspan=1, sticky=(N, S, W, E))

file_title = Label(main, text='Image')
file_title.grid(row=3,column=0,columnspan=1)
file_path = Label(main, text=file_name)
file_path.grid(row=3,column=1,columnspan=2)
Button(main,text="Select", height=1,command=lambda:selectFile()).grid(row=3, column=3, columnspan=1, sticky=(N, S, W, E))

log_ScrolledText = tkst.ScrolledText(main, height=20)
log_ScrolledText.grid(row=4,column=0,columnspan=4, sticky=(N, S, W, E))

log_ScrolledText.configure(font='TkFixedFont')

log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14))
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center')

log_ScrolledText.insert(END, '\n\n1. Please select an Image\n2. Drag an Object\n3. Press Enter or Space key\n', 'TITLE')

main.mainloop()
