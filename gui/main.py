import os
import sys
from pathlib import Path
from tkinter import Tk, Canvas, Button, PhotoImage, Frame, filedialog
from pydub import AudioSegment
from pydub.playback import play
import pandas as pd
import numpy as np
# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import IPython.display as ipd
import simpleaudio as sa
from PIL import Image, ImageTk
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import threading
import time
# to play the audio files
from IPython.display import Audio
import sounddevice as sd
import wavio

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH_FRAME0 = OUTPUT_PATH / Path(r"assets\frame0")
ASSETS_PATH_FRAME1 = OUTPUT_PATH / Path(r"assets\frame1")

############## Main functions ##############
# Getting the images from the correct frame directory
def relative_to_assets(path: str, frame: int) -> Path:
    if frame == 0:
        return ASSETS_PATH_FRAME0 / Path(path)
    else:
        return ASSETS_PATH_FRAME1 / Path(path)
    
def plot_waveform(file_path):
    data, sampling_rate = librosa.load(file_path, sr=44100)
    
    plt.figure(figsize=(4.84, 0.89))  # 484x89 pixels
    librosa.display.waveshow(data, sr=sampling_rate)
    plt.xlabel('Time (s)')
    plt.axis('off')  # Hide the axis except for the x-axis
    plt.tight_layout()
    
    plt.savefig("waveform.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    data, sampling_rate = librosa.load(file_path, sr=44100)
    
    plt.figure(figsize=(4.84, 0.89))  # 484x89 pixels
    librosa.display.waveshow(data, sr=sampling_rate)
    plt.axis('off')  # Hide the axis
    plt.tight_layout()
    
    plt.savefig("waveform.png", bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_empty_waveform():
    plt.figure(figsize=(4.84, 0.89))  # 484x89 pixels
    plt.plot([0, 1], [0, 0], color='black')  # A line at 0 amplitude
    plt.xlabel('Time (s)')
    plt.axis('off')  # Hide the axis except for the x-axis
    plt.tight_layout()
    
    plt.savefig("empty_waveform.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure(figsize=(4.84, 0.89))  # 484x89 pixels
    plt.plot([0, 1], [0, 0], color='black')  # A line at 0 amplitude
    plt.axis('off')  # Hide the axis
    plt.tight_layout()
    
    plt.savefig("empty_waveform.png", bbox_inches='tight', pad_inches=0)
    plt.close()

# Trasfer from one page to another
def switch_to_frame(frame):
    if frame == 1:
        analyze_frame.pack_forget()
        insert_frame.pack(fill="both", expand=True)
    elif frame == 0:
        insert_frame.pack_forget()
        analyze_frame.pack(fill="both", expand=True)


############## Record button functions ##############
# Global variable to track if the red dot is currently visible
red_dot_visible = False
red_dot = None
recording = False
audio_data = []
fs = 44100  # Sample rate

def toggle_red_dot():
    global red_dot_visible, red_dot

    if red_dot_visible:
        # If the red dot is visible, stop recording
        insert_canvas.delete(red_dot)
        red_dot_visible = False
    else:
        # If the red dot is not visible, start recording
        red_dot = insert_canvas.create_oval(345, 200, 355, 213, fill="#F70000", outline="")
        red_dot_visible = True

def record_audio():
    global recording, audio_data

    audio_data = []

    def callback(indata, frames, time, status):
        if recording:
            audio_data.extend(indata[:, 0])

    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        while recording:
            sd.sleep(100)

def toggle_recording():
    global recording, audio_data, file_path

    if not recording:
        # Start recording
        recording = True
        toggle_red_dot()  # להפעיל את נקודת האור האדומה
        record_button.config(image=button_stop_img)  # Change button image to stop
        threading.Thread(target=record_audio).start()
    else:
        # Stop recording
        recording = False
        toggle_red_dot()  # לכבות את נקודת האור האדומה
        record_button.config(image=record_button_img)  # Revert button image to record

        # Convert audio data to numpy array
        audio_data = np.array(audio_data)

        # Save the recorded audio
        file_path = "recorded_audio.wav"  # עדכון המשתנה file_path
        wavio.write(file_path, audio_data, fs, sampwidth=2)

        # Display waveform
        plot_waveform(file_path)
        
        waveform_image = Image.open("waveform.png")
        waveform_photo = ImageTk.PhotoImage(waveform_image)
        
        insert_canvas.create_image(
            375, 259.5, image=waveform_photo, anchor="center"
        )
        insert_canvas.image = waveform_photo

        # Update time display
        duration_seconds = len(audio_data) // fs
        minutes = duration_seconds // 60
        seconds = duration_seconds % 60
        insert_canvas.itemconfig(audio_time_text, text=f"{minutes:02d}:{seconds:02d}")

############## Upload button functions ##############
# Global variable to store the loaded audio
loaded_audio = None
file_path = None
def upload_audio():
    global loaded_audio
    global file_path

    file_path = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.ogg *.flac")]
    )

    if file_path:
        try:
            loaded_audio = AudioSegment.from_file(file_path)
            print(f"Audio file '{file_path}' loaded successfully.")
            print(f"Audio duration: {len(loaded_audio) / 1000.0} seconds")
            
            plot_waveform(file_path)
            
            waveform_image = Image.open("waveform.png")
            waveform_photo = ImageTk.PhotoImage(waveform_image)
            
            insert_canvas.create_image(
                375, 259.5, image=waveform_photo, anchor="center"
            )
            insert_canvas.image = waveform_photo

            duration_seconds = int(len(loaded_audio) / 1000)
            minutes = duration_seconds // 60
            seconds = duration_seconds % 60
            insert_canvas.itemconfig(audio_time_text, text=f"{minutes:02d}:{seconds:02d}")

        except Exception as e:
            print(f"Error loading audio file: {e}")

############## Trash button functions ##############
def Trash_function():
    global audio_data, file_path

    # Reset the audio data
    audio_data = []
    file_path = None

    # Plot and display the initial empty waveform
    plot_empty_waveform()
    waveform_image = Image.open("empty_waveform.png")
    waveform_photo = ImageTk.PhotoImage(waveform_image)

    insert_canvas.create_image(
        375, 259.5, image=waveform_photo, anchor="center"
    )
    insert_canvas.image = waveform_photo

    # Reset the time display
    insert_canvas.itemconfig(audio_time_text, text="00:00")

    print("Recording reset successfully.")

############## Play button functions ##############
# Global variables to track playback state and the play object
is_playing = False
play_obj = None

def play_audio():
    global is_playing, play_obj

    # Load the audio file with librosa
    data, sampling_rate = librosa.load(file_path, sr=44100)
    
    # Convert the numpy array to 16-bit PCM format
    audio_data = (data * 32767).astype('int16')

    # Play the audio using simpleaudio
    play_obj = sa.play_buffer(audio_data, 1, 2, sampling_rate)
    play_obj.wait_done()

    # After the sound finishes, reset the button image
    play_button.config(image=play_button_image)
    is_playing = False

def toggle_play():
    global is_playing, play_obj

    # Check if there is an audio file to play
    if not file_path:
        print("No audio file available to play.")
        return

    if not is_playing:
        # Start playback in a new thread
        is_playing = True
        play_button.config(image=button_stop_img)  # Change button image to stop
        threading.Thread(target=play_audio).start()
    else:
        # Stop playback
        if play_obj:
            play_obj.stop()
        play_button.config(image=play_button_image)  # Revert button image to play
        is_playing = False

# Analyze button functions ##############



window = Tk()
window.geometry("752x459")
window.configure(bg="#5F95FF")

######### INSERT FRAME #########
insert_frame = Frame(window)
insert_frame.pack(fill="both", expand=True)

# TOP HEADING
insert_canvas = Canvas(
    insert_frame,
    bg="#5F95FF",
    height=459,
    width=752,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
insert_canvas.place(x=0, y=0)
insert_canvas.create_text(
    92.0,
    24.0,
    anchor="nw",
    text="Speech Emotion Recognition App",
    fill="#4D4D4D",
    font=("MontserratRoman Bold", 32 * -1)
)

# LOWER HEADING
insert_canvas.create_rectangle(
    0.0,
    86.0,
    752.0,
    459.0,
    fill="#FFFFFF",
    outline=""
)

# RECORD BUTTON
# Load the stop button image
button_stop_img = PhotoImage(
    file=relative_to_assets("button_stop.png", 1))
record_button_img = PhotoImage(
    file=relative_to_assets("button_5.png", 1))
record_button = Button(
    insert_frame,
    image=record_button_img,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: toggle_recording(),
    relief="flat"
)
record_button.place(
    x=108.0,
    y=113.0,
    width=50.975341796875,
    height=68.06185913085938
)

# UPLOAD BUTTON
upload_button_img = PhotoImage(
    file=relative_to_assets("button_2.png", 1))
upload_button = Button(
    insert_frame,
    image=upload_button_img,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: upload_audio(),
    relief="flat"
)
upload_button.place(
    x=592.0,
    y=113.0,
    width=50.975341796875,
    height=68.06185913085938
)

# TRASH BUTTON
trash_button_img = PhotoImage(
    file=relative_to_assets("button_1.png", 1))
trash_button = Button(
    insert_frame,
    image=trash_button_img,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: Trash_function(),
    relief="flat"
)
trash_button.place(
    x=63.0,
    y=360.0,
    width=50.975341796875,
    height=68.06185913085938
)

# PLAY BUTTON
play_button_image = PhotoImage(
    file=relative_to_assets("button_3.png", 1))
play_button = Button(
    insert_frame,
    image=play_button_image,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: toggle_play(),
    relief="flat"
)
play_button.place(
    x=351.0,
    y=360.0,
    width=50.975341796875,
    height=68.06185913085938
)

# ANALYZE BUTTON
analyze_button_img = PhotoImage(
    file=relative_to_assets("button_4.png", 1))
analyz_button = Button(
    insert_frame,
    image=analyze_button_img,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: switch_to_frame(0),
    relief="flat"
)
analyz_button.place(
    x=618.0,
    y=360.0,
    width=50.975341796875,
    height=68.06185913085938
)

# AUDIO GRAPH RECTANGLE
insert_canvas.create_rectangle(
    133.0,
    215.0,
    617.0,
    304.0,
    fill="#D9D9D9",
    outline=""
)

# Plot and display the initial empty waveform
plot_empty_waveform()
waveform_image = Image.open("empty_waveform.png")
waveform_photo = ImageTk.PhotoImage(waveform_image)

insert_canvas.create_image(
    375, 259.5, image=waveform_photo, anchor="center"  # Center of the rectangle
)
insert_canvas.image = waveform_photo  # Keep a reference to avoid garbage collection


# TIME OF THE AUDIO
audio_time_text = insert_canvas.create_text(
    358.0,
    200.0,
    anchor="nw",
    text="00:00",
    fill="#000000",
    font=("MontserratRoman Regular", 12 * -1)
)

######### ANALYZE FRAME #########
analyze_frame = Frame(window)

# TOP HEADING
analyze_canvas = Canvas(
    analyze_frame,
    bg="#5F95FF",
    height=459,
    width=752,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
analyze_canvas.place(x=0, y=0)
analyze_canvas.create_text(
    92.0,
    24.0,
    anchor="nw",
    text="Speech Emotion Recognition App",
    fill="#4D4D4D",
    font=("MontserratRoman Bold", 32 * -1)
)
# LOWER HEADING
analyze_canvas.create_rectangle(
    0.0,
    86.0,
    752.0,
    459.0,
    fill="#FFFFFF",
    outline=""
)
analyze_canvas.create_rectangle(
    80.0,
    152.0,
    686.0,
    332.0,
    fill="#D9D9D9",
    outline=""
)

# BACK BUTTON
back_button_img = PhotoImage(
    file=relative_to_assets("button_1.png", 0))
back_button = Button(
    analyze_frame,
    image=back_button_img,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: switch_to_frame(1),
    relief="flat"
)
back_button.place(
    x=351.0,
    y=361.0,
    width=50.975341796875,
    height=68.06185913085938
)

window.resizable(False, False)
window.mainloop()
