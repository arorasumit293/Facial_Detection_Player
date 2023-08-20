import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk
import os
import pygame
from pygame import mixer
import sys
import random

selected_music = ""

def play():
    global current_file, selected_music
    folder_path = os.path.join(r"C:\Users\risha\OneDrive\Desktop\Emotion-detection\src\Music", received_emo)
    music_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
    to_be_played_file = random.choice(music_files)
    selected_music = pygame.mixer.Sound(os.path.join(folder_path, to_be_played_file))
    selected_music.play()
    current_file = os.path.basename(os.path.join(folder_path,to_be_played_file))

def action():
    global selected_music
    selected_music.stop()

mixer.init()
received_emo = sys.argv[1]
current_file = ""
play()


AU_mainwindow = tk.Tk()
AU_mainwindow.geometry("805x506")
AU_mainwindow.resizable(False,False)
AU_mainwindow.title("PLAYER")

style = ttk.Style(AU_mainwindow)

cam_frame = tk.Frame(AU_mainwindow)
cam_frame.pack(side = 'left',fill = 'both',expand=True)
cam_frame['borderwidth'] = 1
cam_frame['relief'] = 'solid'

running_state = tk.BooleanVar()

label1 = ttk.Label(cam_frame)
label1.pack(fill='both',expand=True)

label1_img= ImageTk.PhotoImage(file = r'C:\Users\risha\OneDrive\Desktop\Emotion-detection\src\images\bg22.png')
label1['style'] = 'CustomLabelStyle.TLabel'
style.configure('CustomLabelStyle.TLabel',image = label1_img)

start_button = tk.Button(cam_frame, bd=0,height = 0,bg = '#060805',activebackground='#060805', width = 0,command = action)
start_button.place(x = 381, y = 245)

text_area = tk.Text(cam_frame,height = 1,width = 43, border = 0, font = ("Arial",11), bg = '#060805', fg = 'green')
text_area.place(x = 250, y = 380)
text_area.insert("1.0",f"Now Playing '{current_file}' For your {received_emo} mood")
AU_mainwindow.mainloop()