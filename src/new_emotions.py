import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk
import os
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
outgoing_arg = ""

def start_process():
    running_state.set(value=True)
    label2.destroy()
    authenticate()

def action_process():
    running_state.set(False)
    text_area.delete("1.0","end")
    command = ['python', r'C:\Users\risha\OneDrive\Desktop\Emotion-detection\src\player.py', outgoing_arg]
    subprocess.call(command)

def stop_process():
    running_state.set(False)
    text_area.delete("1.0","end")

def quit_process():
    AU_mainwindow.destroy()


# emotions will be displayed on your face from the webcam feed
def authenticate():
    model.load_weights(r'C:\Users\risha\OneDrive\Desktop\Emotion-detection\model.h5')
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
    # start the webcam feed
    while running_state.get():
        global outgoing_arg
        cap = cv2.VideoCapture(0)

        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier(r'C:\Users\risha\OneDrive\Desktop\Emotion-detection\src\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        text_area.delete("1.0","end")
        text_area.insert("1.0","Emotions Detected\n")
        str_var = ""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            str_var = emotion_dict[maxindex]
        img= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        label1['image'] = photo
        outgoing_arg = str_var
        text_area.insert("2.0",str_var)
        AU_mainwindow.update()
                
AU_mainwindow = tk.Tk()
AU_mainwindow.geometry("800x500")
AU_mainwindow.resizable(False,False)
AU_mainwindow.title("FACE RECOGNIZER")

style = ttk.Style(AU_mainwindow)

cam_frame = tk.Frame(AU_mainwindow)
cam_frame.pack(side = 'left',fill = 'both',expand=True)
cam_frame['borderwidth'] = 1
cam_frame['relief'] = 'solid'

ttk.Separator(AU_mainwindow,orient='vertical').pack(side = 'left' , fill='y',padx =(0,5))

button_frame = ttk.Frame(AU_mainwindow)
button_frame.pack(side = 'left',fill = 'both')
button_frame['borderwidth'] = 1
button_frame['relief'] = 'solid'
button_frame.grid_columnconfigure(0,weight=1)
button_frame.grid_rowconfigure(0,weight=1)

running_state = tk.BooleanVar()

init_msg = tk.StringVar(value = "Camera Will Open In Here")
label1 = ttk.Label(cam_frame)
label1.pack(fill='both',expand=True)

label1_img= ImageTk.PhotoImage(file = r'C:\Users\risha\OneDrive\Desktop\Emotion-detection\src\images\bg.jpg')
label1['style'] = 'CustomLabelStyle.TLabel'
style.configure('CustomLabelStyle.TLabel',image = label1_img)

label2_img= ImageTk.PhotoImage(file = r'C:\Users\risha\OneDrive\Desktop\Emotion-detection\src\images\text.png')
label2 = tk.Label(cam_frame,image = label2_img,bd = 0,bg = '#fefefe',activebackground = '#fefefe')
label2.place(x = 231,y = 230)

b_img= ImageTk.PhotoImage(file = r'C:\Users\risha\OneDrive\Desktop\Emotion-detection\src\images\background.jpg')
label = tk.Label(button_frame,image=b_img)
label.place(x=0,y=0)

name_label = tk.Label(button_frame)
name_label.grid(row=0,column=0,padx=(50,85),pady=(5,5))

action_img= ImageTk.PhotoImage(file = r'C:\Users\risha\OneDrive\Desktop\Emotion-detection\src\images\song.png')

b1_img= ImageTk.PhotoImage(file = r'C:\Users\risha\OneDrive\Desktop\Emotion-detection\src\images\button_detect.ico')
b2_img= ImageTk.PhotoImage(file = r'C:\Users\risha\OneDrive\Desktop\Emotion-detection\src\images\button_stop.ico')
b3_img= ImageTk.PhotoImage(file = r'C:\Users\risha\OneDrive\Desktop\Emotion-detection\src\images\button_quit.png')

action_button = tk.Button(button_frame,image = action_img,bd = 0,bg = '#fefefe',activebackground = '#fefefe', command = action_process)
action_button.place(x = 39, y= 43)

start_button = tk.Button(button_frame, image = b1_img , bd=0 , bg='#fefefe',activebackground='#fefefe',command = start_process)
start_button.place(x = 21, y = 120)

stop_button = tk.Button(button_frame, image = b2_img, bd=0 , bg='#fefefe',activebackground='#fefefe',command = stop_process)
stop_button.place(x = 21, y = 200)

quit_button = tk.Button(button_frame, image = b3_img, bd=0 , bg='#fefefe',activebackground='#fefefe',
command = quit_process,padx=10)
quit_button.place(x = 21, y = 275)

text_area = tk.Text(button_frame,height = 7,width = 15,font = ("Arial",9))
text_area.place(x = 12, y = 357)

AU_mainwindow.mainloop()
