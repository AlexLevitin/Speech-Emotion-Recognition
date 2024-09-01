import os
import sys
from pathlib import Path
from tkinter import Tk, Canvas, Button, PhotoImage, Frame, filedialog, messagebox
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
from tensorflow.keras.models import load_model
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import SpeechEmotionRecognition as p2

# Model Variables
model = load_model('SER Model.keras')
original_labels = np.array(['angry', 'disgust', 'fear', 'happy' ,'neutral' ,'sad' , 'surprise']).reshape(-1, 1)  # Replace with your actual labels
# Create and fit the encoder
# encoder = OneHotEncoder()
# encoder.fit(original_labels)
# long_angry = "DataSets\long_angry_sample.mp3"
# data, sample_rate = librosa.load(long_angry,offset=0.6, sr=16000)
# segs = p2.split_audio(data, sample_rate, 3)
# seg_features = p2.get_features_segments(segs, sample_rate)
# p2.padFeature(seg_features, 94)
# seg_features = p2.applyScaler(seg_features)
# print("Check segmentation prediction:\n")
# predictions = model.predict(np.array(seg_features))
# combined_prediction = p2.combine_predictions(encoder.inverse_transform(predictions))
# print(str(predictions))
# print("Predictions ang are: " + str(encoder.inverse_transform(predictions)))
# print("Combined prediction is: " + str(combined_prediction))
# file_path = long_angry
#     # Create and fit the encoder
# encoder = OneHotEncoder()
# encoder.fit(original_labels)
# data, sample_rate = librosa.load(file_path,offset=0.6, sr=16000)
# segs = p2.split_audio(data, sample_rate, 3)
# seg_features = p2.get_features_segments(segs, sample_rate)
# p2.padFeature(seg_features, 94)
# seg_features = p2.applyScaler(seg_features)
# print("Check segmentation prediction:\n")
# predictions = model.predict(np.array(seg_features))
# combined_prediction = p2.combine_predictions(encoder.inverse_transform(predictions))
# print("Predictions ang are: " + str(encoder.inverse_transform(predictions)))
# print("Combined prediction is: " + str(combined_prediction))

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
    global current_position
    current_position = 0
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

        # Reset the line to the start position after recording
        insert_canvas.coords(line_insert, insert_graph_start_x, insert_graph_start_y, insert_graph_start_x, insert_graph_end_y)
        analyze_canvas.coords(line_analyze, analyze_graph_start_x, analyze_graph_start_y, analyze_graph_start_x, analyze_graph_end_y)

        # Reset current position to zero
        current_position = 0

        loaded_audio = AudioSegment.from_file(file_path)
        # Update time display
        duration_seconds = int(len(loaded_audio) / 1000)
        minutes = duration_seconds // 60
        seconds = duration_seconds % 60
        insert_canvas.itemconfig(audio_time_text, text=f"{minutes:02d}:{seconds:02d}")


############## Upload button functions ##############
# Global variable to store the loaded audio
loaded_audio = None
file_path = None
def upload_audio():
    global loaded_audio, file_path, current_position

    file_path = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.ogg *.flac")]
    )

    if file_path:
        try:
            loaded_audio = AudioSegment.from_file(file_path)
            print(f"Audio file '{file_path}' loaded successfully.")
            print(f"Audio duration: {len(loaded_audio) / 1000.0} seconds")
            
            # Plot and display the waveform
            plot_waveform(file_path)
            
            waveform_image = Image.open("waveform.png")
            waveform_photo = ImageTk.PhotoImage(waveform_image)
            
            insert_canvas.create_image(
                375, 259.5, image=waveform_photo, anchor="center"
            )
            insert_canvas.image = waveform_photo

            # Reset the line to the start position
            insert_canvas.coords(line_insert, insert_graph_start_x, insert_graph_start_y, insert_graph_start_x, insert_graph_end_y)
            analyze_canvas.coords(line_analyze, analyze_graph_start_x, analyze_graph_start_y, analyze_graph_start_x, analyze_graph_end_y)

            # Reset current position to zero
            current_position = 0

            # Update time display
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
# Global variables to track playback state
is_playing = False
play_obj = None
current_position = 0  # Keeps track of the current position in the audio in seconds
stop_playback = False  # Flag to stop playback when pause is clicked
total_duration = 0  # Total duration of the audio

# Update positions and constraints based on graph positions
insert_graph_start_x = 170  # מיקום ה-X ההתחלתי של הגרף
insert_graph_start_y = 215  # מיקום ה-Y ההתחלתי של הגרף
insert_graph_end_x = 580    # מיקום ה-X הסופי של הגרף
insert_graph_end_y = 304    # מיקום ה-Y הסופי של הגרף

# הגדרת מיקומים עבור הקו בפריים analyze (ייתכן שהוא נמצא נמוך יותר)
analyze_graph_start_x = 170  # מיקום ה-X ההתחלתי של הגרף בפריים השני
analyze_graph_start_y = 232  # מיקום ה-Y ההתחלתי של הגרף בפריים השני
analyze_graph_end_x = 580    # מיקום ה-X הסופי של הגרף בפריים השני
analyze_graph_end_y = 320    # מיקום ה-Y הסופי של הגרף בפריים השני

# Function to update the line and the audio position in the insert canvas
def update_line_and_audio_position_insert():
    global current_position, stop_playback

    while is_playing and current_position < total_duration and not stop_playback:
        # Calculate the position of the line
        x_position = (current_position / total_duration) * (insert_graph_end_x - insert_graph_start_x)

        # Update the position of the vertical line on the insert canvas
        insert_canvas.coords(line_insert, x_position + insert_graph_start_x, insert_graph_start_y, x_position + insert_graph_start_x, insert_graph_end_y)

        # Sleep for a short duration to update the line smoothly
        time.sleep(0.02)  # Update every 20 milliseconds
        current_position += 0.02  # Move 20ms forward in the audio

# Function to update the line and the audio position in the analyze canvas
def update_line_and_audio_position_analyze():
    global current_position, stop_playback

    while is_playing and current_position < total_duration and not stop_playback:
        # Calculate the position of the line
        x_position = (current_position / total_duration) * (analyze_graph_end_x - analyze_graph_start_x)

        # Update the position of the vertical line on the analyze canvas
        analyze_canvas.coords(line_analyze, x_position + analyze_graph_start_x, analyze_graph_start_y, x_position + analyze_graph_start_x, analyze_graph_end_y)

        # Sleep for a short duration to update the line smoothly
        time.sleep(0.02)  # Update every 20 milliseconds
        current_position += 0.02  # Move 20ms forward in the audio

# Function to move the vertical line when the user drags it
def move_line(event, canvas_type="insert"):
    global current_position

    # Check if an audio file is loaded
    if not file_path:
        return

    if canvas_type == "insert":
        x_position = event.x
        if x_position < insert_graph_start_x:
            x_position = insert_graph_start_x
        elif x_position > insert_graph_end_x:
            x_position = insert_graph_end_x
        
        # Update the vertical line position on the insert canvas
        insert_canvas.coords(line_insert, x_position, insert_graph_start_y, x_position, insert_graph_end_y)
        
        # Update the current position in the audio
        current_position = ((x_position - insert_graph_start_x) / (insert_graph_end_x - insert_graph_start_x)) * total_duration

    elif canvas_type == "analyze":
        x_position = event.x
        if x_position < analyze_graph_start_x:
            x_position = analyze_graph_start_x
        elif x_position > analyze_graph_end_x:
            x_position = analyze_graph_end_x
        
        # Update the vertical line position on the analyze canvas
        analyze_canvas.coords(line_analyze, x_position, analyze_graph_start_y, x_position, analyze_graph_end_y)
        
        # Update the current position in the audio
        current_position = ((x_position - analyze_graph_start_x) / (analyze_graph_end_x - analyze_graph_start_x)) * total_duration

    # If the audio is playing, stop it and resume playback from the new position
    if is_playing:
        stop_playback = True
        play_audio(canvas_type)

# Function to handle when the user clicks on the graph to move the line and stop playback
def move_line_on_click(event, canvas_type="insert"):
    global current_position, is_playing, stop_playback

    # עצירה של כל פעולות השמעה נוכחיות
    stop_playback = True
    if play_obj:
        play_obj.stop()
    is_playing = False


    # חישוב המיקום החדש לפי הלחיצה
    if canvas_type == "insert":
        play_button.config(image=play_button_image)  # Switch back to play button
        x_position = event.x
        if x_position < insert_graph_start_x:
            x_position = insert_graph_start_x
        elif x_position > insert_graph_end_x:
            x_position = insert_graph_end_x
        
        # עדכון מיקום הקו ב-canvas של insert
        insert_canvas.coords(line_insert, x_position, insert_graph_start_y, x_position, insert_graph_end_y)
        
        # עדכון המיקום הנוכחי באודיו לפי המיקום שבו לחצו
        current_position = ((x_position - insert_graph_start_x) / (insert_graph_end_x - insert_graph_start_x)) * total_duration

    elif canvas_type == "analyze":
        analyze_play_button.config(image=play_button_image)  # Switch back to play button
        x_position = event.x
        if x_position < analyze_graph_start_x:
            x_position = analyze_graph_start_x
        elif x_position > analyze_graph_end_x:
            x_position = analyze_graph_end_x
        
        # עדכון מיקום הקו ב-canvas של analyze
        analyze_canvas.coords(line_analyze, x_position, analyze_graph_start_y, x_position, analyze_graph_end_y)
        
        # עדכון המיקום הנוכחי באודיו לפי המיקום שבו לחצו
        current_position = ((x_position - analyze_graph_start_x) / (analyze_graph_end_x - analyze_graph_start_x)) * total_duration

    # ההשמעה תתחיל מהמיקום החדש כשילחץ על כפתור play
    # אם המשתמש רוצה להמשיך את ההשמעה מהמיקום החדש, הוא ילחץ שוב על כפתור play

# PLAY/PAUSE BUTTON
def toggle_play_pause(canvas_type="insert"):
    global is_playing, stop_playback, current_position

    # בדיקה אם יש קובץ טעון
    if not file_path:
        messagebox.showerror("Error", "No Audio is Recorded or Uploaded")
        return

    if not is_playing:
        # Start playback
        is_playing = True
        stop_playback = False
        if canvas_type == "insert":
            play_button.config(image=pause_button_img)  # Switch to pause button for insert frame
        else:
            analyze_play_button.config(image=pause_button_img)  # Switch to pause button for analyze frame

        # אם ההשמעה הסתיימה וצריך להתחיל מחדש, נאתחל את current_position
        if current_position >= total_duration:
            current_position = 0

        threading.Thread(target=play_audio, args=(canvas_type,)).start()
    else:
        # Pause playback
        stop_playback = True
        if play_obj:
            play_obj.stop()
        if canvas_type == "insert":
            play_button.config(image=play_button_image)  # Switch to play button for insert frame
        else:
            analyze_play_button.config(image=play_button_image)  # Switch to play button for analyze frame
        is_playing = False

def play_audio(canvas_type="insert"):
    global is_playing, current_position, stop_playback, play_obj, total_duration

    # Stop previous playback if needed
    stop_playback = False

    # Calculate total duration
    total_duration = librosa.get_duration(filename=file_path)

    # Calculate how much of the audio has already been played, and start from there
    data, sampling_rate = librosa.load(file_path, sr=44100, offset=current_position)
    audio_data = (data * 32767).astype('int16')

    # Start playing the audio
    play_obj = sa.play_buffer(audio_data, 1, 2, sampling_rate)

    # Start updating the line
    if canvas_type == "insert":
        threading.Thread(target=update_line_and_audio_position_insert).start()
    else:
        threading.Thread(target=update_line_and_audio_position_analyze).start()

    # Wait for the audio to finish
    play_obj.wait_done()

    # Once done playing, reset the line to the start position and switch button back to "play"
    if not stop_playback:
        if canvas_type == "insert":
            insert_canvas.coords(line_insert, insert_graph_start_x, insert_graph_start_y, insert_graph_start_x, insert_graph_end_y)
            play_button.config(image=play_button_image)
        else:
            analyze_canvas.coords(line_analyze, analyze_graph_start_x, analyze_graph_start_y, analyze_graph_start_x, analyze_graph_end_y)
            analyze_play_button.config(image=play_button_image)
        
        # Reset playback state
        is_playing = False
        current_position = total_duration  # Mark that the playback has completed
        
# PLAY/PAUSE BUTTON עבור פריים analyze
def toggle_play_pause_analyze():
    toggle_play_pause(canvas_type="analyze")

# PLAY/PAUSE BUTTON עבור פריים insert
def toggle_play_pause_insert():
    toggle_play_pause(canvas_type="insert")

############ Analyze button functions ##############
# הגדרת משתנה גלובלי לאחסון התמונות
emotion_images_ids = []
current_pop_image_id = None
current_original_image_id = None
current_original_image = None


def analyze():
    global file_path, emotion_images_ids, combined_emotion

    # ניקוי תמונות קודמות אם קיימות
    clear_emotion_images()

    if not file_path:
        messagebox.showerror("Error", "No Audio was Uploaded or Recorded")
        return

    try:
        data, sample_rate = librosa.load(file_path, offset=0.6, sr=16000)
        segs = p2.split_audio(data, sample_rate, 3)
        seg_features = p2.get_features_segments(segs, sample_rate)
        p2.padFeature(seg_features, 94)
        seg_features = p2.applyScaler(seg_features)

        predictions = model.predict(np.array(seg_features))
        combined_prediction = p2.combine_predictions(encoder.inverse_transform(predictions))

        print("Predictions are: " + str(encoder.inverse_transform(predictions)))
        print("Combined prediction is: " + str(combined_prediction))

        plot_waveform(file_path)
        waveform_image = Image.open("waveform.png")
        waveform_photo = ImageTk.PhotoImage(waveform_image)

        analyze_canvas.create_image(
            376.0, 276.0,
            image=waveform_photo,
            anchor="center"
        )
        analyze_canvas.image = waveform_photo

        total_duration = librosa.get_duration(filename=file_path)
        total_minutes = int(total_duration // 60)
        total_seconds = int(total_duration % 60)
        analyze_canvas.itemconfig(analyze_timer, text=f"00:00 / {total_minutes:02d}:{total_seconds:02d}")

        # הצגת רגשות על הגרף
        display_emotion_popups_analyze(predictions, segs, total_duration)

        # Clear the previous combined_emotion text by deleting it if it exists
        if 'combined_emotion' in globals():
            analyze_canvas.delete(combined_emotion)

        # Create the combined_emotion text
        combined_emotion = analyze_canvas.create_text(
            200.0, 140.0,  # Position below the main title
            anchor="nw",
            text=f"Overall Emotion: {combined_prediction}",
            fill="#4D4D4D",
            font=("MontserratRoman Bold", 32 * -1)
        )

        switch_to_frame(0)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def display_emotion_popups_analyze(predictions, segs, total_duration):
    global analyze_graph_start_x, analyze_graph_end_x, emotion_images_ids

    emotion_images = []
    segment_duration = 3  # Duration of each segment

    for i, prediction in enumerate(predictions):
        # Get the predicted probabilities for the current segment
        prediction_probabilities = prediction.flatten()

        # Extract the emotion names and corresponding probabilities
        emotion_names = original_labels.flatten()
        significant_emotions = []
        significant_percentages = []
        other_percentage = 0.0

        # Identify significant emotions and combine less significant ones
        for j, percentage in enumerate(prediction_probabilities):
            if percentage > 0.10:  # Threshold for significant emotion
                significant_emotions.append(emotion_names[j])
                significant_percentages.append(percentage)
            else:
                other_percentage += percentage

        if other_percentage > 0:
            significant_emotions.append('Others')
            significant_percentages.append(other_percentage)

        # Calculate the x position for the emotion image based on time
        start_time = (i + 1) * segment_duration
        if start_time >= total_duration:
            start_time = total_duration

        x_position = analyze_graph_start_x + (start_time / total_duration) * (analyze_graph_end_x - analyze_graph_start_x)

        # Ensure x_position is within graph boundaries
        x_position = max(min(x_position, analyze_graph_end_x), analyze_graph_start_x)

        # Get the main emotion with the highest percentage for this segment
        max_index = np.argmax(significant_percentages)
        main_emotion = significant_emotions[max_index].lower()
        emotion_image_path = f"{main_emotion}.png"
        emotion_image = PhotoImage(file=relative_to_assets(emotion_image_path, 0))

        # Display the emotion image on the graph
        img_id = analyze_canvas.create_image(
            x_position, 230,  # Y position is fixed
            image=emotion_image,
            anchor="center"
        )

        # Store image ID and details for later use
        emotion_images_ids.append(img_id)
        emotion_images.append(emotion_image)

        # Bind click event to show pie chart on pop-up
        analyze_canvas.tag_bind(img_id, "<Button-1>", lambda event, img_id=img_id, emotion_image=emotion_image, percentages=significant_percentages, labels=significant_emotions: on_emotion_image_click(event, img_id, emotion_image, percentages, labels))

    # Store emotion images in canvas to avoid garbage collection
    analyze_canvas.emotion_images = emotion_images

def on_emotion_image_click(event, img_id, emotion_image, percentages, labels):
    global current_pop_image_id, current_original_image_id, current_original_image

    # Revert to original image if the same pop image is clicked
    if current_pop_image_id == img_id:
        analyze_canvas.itemconfig(img_id, image=current_original_image)
        analyze_canvas.coords(img_id, analyze_canvas.coords(img_id)[0], 230)  # Reset Y position
        current_pop_image_id = None
        current_original_image_id = None
        current_original_image = None
        return

    # Revert any other pop images to original
    if current_pop_image_id is not None:
        analyze_canvas.itemconfig(current_pop_image_id, image=current_original_image)
        analyze_canvas.coords(current_pop_image_id, analyze_canvas.coords(current_pop_image_id)[0], 230)  # Reset Y position

    # Create and display the pie chart
    pie_chart_path = "pie_chart.png"
    create_pie_chart(percentages, labels, pie_chart_path)

    # Load pop image and overlay pie chart
    pop_image = Image.open(relative_to_assets("pop.png", 0))
    pie_chart_image = Image.open(pie_chart_path)
    pop_image.paste(pie_chart_image, (10, 10))  # Adjust positioning within the pop image

    # Convert to PhotoImage for Tkinter
    pop_image = ImageTk.PhotoImage(pop_image)

    # Adjust Y position for pop image
    pop_image_y = 170
    analyze_canvas.coords(img_id, analyze_canvas.coords(img_id)[0], pop_image_y)
    analyze_canvas.itemconfig(img_id, image=pop_image)
    analyze_canvas.tag_raise(img_id)

    # Store current state
    current_pop_image_id = img_id
    current_original_image_id = img_id
    current_original_image = emotion_image

    # Keep pop image in memory
    analyze_canvas.pop_image = pop_image

def create_pie_chart(emotion_percentages, emotion_labels, pie_chart_path):
    # Filter out insignificant emotions (less than 10%)
    significant_emotions = []
    significant_percentages = []
    other_percentage = 0

    for label, percentage in zip(emotion_labels, emotion_percentages):
        if percentage > 0.10:  # Increased threshold to combine smaller emotions
            significant_emotions.append(label)
            significant_percentages.append(percentage)
        else:
            other_percentage += percentage

    # If there's a significant "Others" percentage, add it to the chart
    if other_percentage > 0:
        significant_emotions.append('Others')
        significant_percentages.append(other_percentage)

    # Create the pie chart with reduced explosion
    plt.figure(figsize=(1.3, 1.6))  # Slightly larger figure size for readability
    plt.pie(significant_percentages, labels=significant_emotions, autopct='%1.1f%%',
            startangle=90, explode=[0.05] * len(significant_emotions), textprops={'fontsize': 8})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Save the pie chart as an image
    plt.savefig(pie_chart_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def clear_emotion_images():
    global emotion_images_ids, current_pop_image_id

    # Delete images using the IDs we saved
    for img_id in emotion_images_ids:
        analyze_canvas.delete(img_id)

    # Clear the list
    emotion_images_ids = []

    # Clear the pop image if it's displayed
    if current_pop_image_id is not None:
        analyze_canvas.delete(current_pop_image_id)
        current_pop_image_id = None
        current_original_image_id = None
        current_original_image = None

window = Tk()
window.geometry("752x459")
window.configure(bg="#5F95FF")
window.title("Speech Emotion Recogniton")

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
    135.0,
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
    file=relative_to_assets("stop.png", 1))
record_button_img = PhotoImage(
    file=relative_to_assets("record.png", 1))
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
    file=relative_to_assets("upload.png", 1))
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
    file=relative_to_assets("trash.png", 1))
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
    file=relative_to_assets("play.png", 1))
play_button = Button(
    insert_frame,
    image=play_button_image,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: toggle_play_pause_insert(),
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
    file=relative_to_assets("analyze.png", 1))
analyz_button = Button(
    insert_frame,
    image=analyze_button_img,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: analyze(),
    relief="flat"
)
analyz_button.place(
    x=618.0,
    y=360.0,
    width=50.975341796875,
    height=68.06185913085938
)

# AUDIO GRAPH RECTANGLE
graph_background = PhotoImage(
    file=relative_to_assets("graph_place.png",1))
image_1 = insert_canvas.create_image(
    375.0,
    259.0,
    image=graph_background
)

# Plot and display the initial empty waveform
plot_empty_waveform()
waveform_image = Image.open("empty_waveform.png")
waveform_photo = ImageTk.PhotoImage(waveform_image)
insert_canvas.bind("<Button-1>", lambda event: move_line_on_click(event, canvas_type="insert"))
line_insert = insert_canvas.create_line(
    insert_graph_start_x, insert_graph_start_y, 
    insert_graph_start_x, insert_graph_end_y, 
    fill="red", width=2
)

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
    135.0,
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

back_button_img = PhotoImage(
    file=relative_to_assets("pop.png", 0))
analyze_graph_background = analyze_canvas.create_image(
    376.0,
    276.0,
    image=graph_background
)
analyze_timer = analyze_canvas.create_text(
    333.0,
    332.0,
    anchor="nw",
    text="00:00 / 00:00",
    fill="#000000",
    font=("MontserratRoman Regular", 12 * -1)
)

analyze_canvas.bind("<Button-1>", lambda event: move_line_on_click(event, canvas_type="analyze"))
line_analyze = analyze_canvas.create_line(
    analyze_graph_start_x, analyze_graph_start_y, 
    analyze_graph_start_x, analyze_graph_end_y, 
    fill="red", width=2
)

# BACK BUTTON
back_button_img = PhotoImage(
    file=relative_to_assets("back.png", 0))
back_button = Button(
    analyze_frame,
    image=back_button_img,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: switch_to_frame(1),
    relief="flat"
)
back_button.place(
    x=80.0,
    y=365.0,
    width=50.975341796875,
    height=68.06185913085938
)

# PLAY BUTTON
pause_button_img = PhotoImage(
    file=relative_to_assets("pause.png", 0))
analyze_play_button = Button(
    analyze_frame,
    image=play_button_image,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: toggle_play_pause_analyze(),#toggle_play_pause_analyze(),
    relief="flat"
)
analyze_play_button.place(
    x=351.0,
    y=365.0,
    width=50.975341796875,
    height=68.06185913085938
)

window.resizable(False, False)
window.mainloop()
