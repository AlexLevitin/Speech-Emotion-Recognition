import tkinter as tk
from tkinter import Canvas, Button, Label, filedialog
import os
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import tempfile
from .design import *
import time # Use system clock for timing
from keras.models import load_model  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # For decoding predictions
import SpeechEmotionRecognition as p2  # For your custom helper functions
import sys

# Load the pre-trained model
if getattr(sys, 'frozen', False):  # If running from an executable
    model_path = os.path.join(sys._MEIPASS, 'SER Model (1).keras')
else:  # If running as a script
    model_path = ('SER Model (1).keras')
model = load_model(model_path)

# Define the emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
encoder = OneHotEncoder()
encoder.fit(np.array(emotion_labels).reshape(-1, 1))

# Global variables for the analyze
is_analyzed = False
overall_emotion, segment_predictions = None, None
emotion_colors = {
    "angry": "#F12E2E",
    "disgust": "#00C605",
    "fear": "#B115AD",
    "happy": "#FFD858",
    "neutral": "#969A8D",
    "sad": "#2E35F1",
    "surprise": "#F1882E"
}

emotion_sample = {
    "angry": [],
    "disgust": [],
    "fear": [],
    "happy": [],
    "neutral": [],
    "sad": [],
    "surprise": []
}

emotion_live = {
    "angry": 0,
    "disgust": 0,
    "fear": 0,
    "happy": 0,
    "neutral": 0,
    "sad": 0,
    "surprise": 0
}

ploted_emotion = ['#B5EBE9'] * 301

# Global variables to store the audio file path, name, duration, and waveform data
audio_file_path = None
audio_file_name = None
audio_duration = None
formatted_duration = None
audio_len = None

# Global variables for storing audio data and sampling rate
audio_data = None
sampling_rate = None
plot_step = None

# Global variable for higlighting bar
current_bar_index = 0
percentage_idx = 0

# Global variables for tracking playback
is_playing = False
playback_start_time = 0
playback_stream = None
volume_level = 100

# Global variables for plot
zooming_window = None # in seconds
max_amplitude = None
is_zoomable = False
zooming_level = 0
zooming_size = None
graph_cords = None
threshold_value = 80

audio_window = {
    "data" : None,
    "start" : None,
    "end" : None,
    "sample_place" : None
}

# For recording
recording_state = {
    "is_recording": False,
    "data": None,
    "rate": 16000,
    "duration": 0,
    "file_path": None
}



def plot_waveform_as_bars(
        self,
        canvas: Canvas,
        emotions_hist_canvas,
        x1: int, y1: int, x2: int, y2: int, 
        audio_data: np.ndarray = None, 
        tag="audio_waveform",
        button: Button = None
        ):    
    """
    Plots a simple waveform as vertical bars on a canvas using provided audio data.
    Parameters:
    - canvas: The canvas to draw the waveform on.
    - x1, y1, x2, y2: Coordinates of the plotting area.
    - audio_data: The audio data as a NumPy array.
    - sampling_rate: Sampling rate of the audio data.
    - tag: Tag to associate with the plotted elements for easy deletion.
    """
    global plot_step,audio_len ,current_bar_index, max_amplitude, zooming_size ,is_zoomable, zooming_window, graph_cords, audio_window, formatted_duration
    graph_cords = x1, y1, x2, y2
    # Draw vertical lines
    delta_lines = 0
    formatted_duration = format_time(audio_duration)
    #spacing = int(np.round(audio_duration)) + 1
    if formatted_duration <= "01:00":
        delta_lines = int(np.ceil((audio_duration)/10))
    elif "01:00" <= formatted_duration <= "01:40":
        delta_lines = 10
    elif "01:40" < formatted_duration <= "05:00":
        delta_lines = 30
    elif "05:00" < formatted_duration <= "10:00":
        delta_lines = 60
    elif "10:00" < formatted_duration <= "50:00":
        delta_lines = 300

    zooming_window = [3,10,30,60,300,600] # in sec
    # Allowing zooming if needed space
    if audio_duration > 30: # if longer then 30 sec
        is_zoomable = True
        filtered_window = [value for value in zooming_window if value <= audio_duration]
        filtered_window.append(audio_duration)
        zooming_window = list(reversed(filtered_window))
        zooming_size = len(zooming_window)
    else :
        is_zoomable = False

    spacing = int(np.ceil(audio_duration / delta_lines))
    proportion = spacing * delta_lines / audio_duration - 1


    # calulating width for the time indicators
    window_width = x2+(x2-x1)*proportion
    for index, x in enumerate(np.linspace(x1, window_width , spacing + 1)):
        if x == x1 or x == window_width:
            continue
        canvas.create_line(x, y1, x, y2, fill="#36424C", dash=(2, 2),
                                tags=tag)
        canvas.create_text(x, y2 - 10, text=f"{format_time(delta_lines * index)}",
                                fill="#506170", font=("Dubai Medium", 10), tags=tag)
        

    # Set dimensions and limits
    width = x2 - x1
    height = y2 - y1 -5
    bar_width = 1  # Width of each bar
    spacing = 1  # Space between bars
    max_bars = width // (bar_width + spacing)  # Maximum number of bars

    try:
        step = max(1, len(audio_data) // max_bars)
        plot_step = step
        downsampled_data = audio_data[::step][:max_bars]
        max_amplitude = np.max(np.abs(audio_data))
        normalized_data = downsampled_data / max_amplitude  # Normalize data
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    middle_y = (y1 + y2) / 2  # Vertical center for the waveform display

    # Plot bars
    for i, sample in enumerate(normalized_data):
        bar_height = float(sample * (height / 2))  # Scale height to canvas
        line_x = x1 + i * (bar_width + spacing)
        # Top and Bottom parts are the same for simplicity
        canvas.create_rectangle(line_x, middle_y - bar_height, line_x + bar_width,
                                 middle_y + bar_height, fill="#B5EBE9", outline="",
                                   tags=(tag, f"bar_{i}"))
    highlight_bar(self, canvas, emotions_hist_canvas, current_bar_index)
    audio_window["start"] = 0
    audio_window["end"] = len(audio_data)
    audio_len = len(audio_data)

def on_canvas_click(self, event, canvas, emotions_hist_canvas,  button, graph_time):
    """Handle click on the canvas to update the highlighted bar and pause playback."""
    global is_playing ,formatted_duration, plot_step
    index = (event.x-201)  // 2

    if 0 <= index <= 298 and index != current_bar_index and 26 <= event.y <= 280:
        # Update UI labels
        update_time = ((index * plot_step) + audio_window["start"]) // sampling_rate
        graph_time.config(text=f"🕒: {format_time(update_time)} / {formatted_duration}")
        update_highlighted_bar(self, event, canvas, emotions_hist_canvas)
        if is_playing:
            stop_playback(button, None)  # Pause playback
    
def on_mouse_drag(self, event, canvas, emotions_hist_canvas, button, graph_time):
    """Handle bar selection during drag and pause playback."""
    global is_playing ,formatted_duration, plot_step
    index = (event.x-201)  // 2

    if 0 <= index <= 298 and index != current_bar_index and 26 <= event.y <= 280:
        # Update UI labels
        update_time = ((index * plot_step) + audio_window["start"]) // sampling_rate
        graph_time.config(text=f"🕒: {format_time(update_time)} / {formatted_duration}")
        update_highlighted_bar(self, event, canvas, emotions_hist_canvas)
        if is_playing:
           stop_playback(button, None)  # Pause playback
    
def update_highlighted_bar(self, event, canvas, emotions_hist_canvas):
    """Update the bar highlight based on mouse position."""
    global current_bar_index, audio_window, percentage_idx, segment_predictions, emotion_live
    bar_width = 1
    spacing = 1
    index = (event.x-201)  // (bar_width + spacing)
    if is_analyzed :
        percentage_idx = 0
        for name in emotion_live:
            emotion_live[name] = 0
        emotion_live[segment_predictions[percentage_idx]["emotions"][0]] += 1
        while ((index * plot_step) + audio_window["start"]) / sampling_rate >= segment_predictions[percentage_idx]["timestamp"]:
            percentage_idx += 1
            emotion_live[segment_predictions[percentage_idx]["emotions"][0]] += 1
        update_persentages(self)
        
    highlight_bar(self, canvas, emotions_hist_canvas,  index)

def highlight_bar(self, canvas: Canvas, emotions_hist_canvas : Canvas,  index, color = "#B5EBE9"):
    """Highlight a specific bar and reset the previous one."""
    global current_bar_index, percentage_idx, emotion_live

    current_bar_index = int(current_bar_index)
    if is_analyzed :
        canvas.itemconfig(f"bar_{current_bar_index}", fill=ploted_emotion[current_bar_index])
        if index != 0:
            canvas.itemconfig(f"bar_{current_bar_index-1}", fill=ploted_emotion[current_bar_index-1])
        if index != 298:
            canvas.itemconfig(f"bar_{current_bar_index+1}", fill=ploted_emotion[current_bar_index+1])

        while ((index * plot_step) + audio_window["start"]) / sampling_rate >= segment_predictions[percentage_idx]["timestamp"]:
            percentage_idx += 1
            emotion_live[segment_predictions[percentage_idx]["emotions"][0]] += 1
        update_emotions_hist(emotions_hist_canvas)
        update_persentages(self)

    # Return to the old color
    else :
        canvas.itemconfig(f"bar_{current_bar_index}", fill="#B5EBE9")
        if index != 0:
            canvas.itemconfig(f"bar_{current_bar_index-1}", fill="#B5EBE9")
        if index != 298:
            canvas.itemconfig(f"bar_{current_bar_index+1}", fill="#B5EBE9")

    # Highlighting the currect color
    if index != 0:
        canvas.itemconfig(f"bar_{index-1}", fill="#6b2c31")
    canvas.itemconfig(f"bar_{index}", fill="#f7636f")
    if index != 298:
        canvas.itemconfig(f"bar_{index+1}", fill="#6b2c31")

    current_bar_index = index

def highligh_emotion_bar(canvas: Canvas, index, color):
    canvas.itemconfig(f"bar_{index}", fill = color)

def upload_audio_file(
        graph_name_label, 
        graph_time_label, 
        main_canvas: Canvas, 
        graph_x, graph_y, graph_x_end, graph_y_end, 
        highlight_frame,
        emotions_hist_canvas,
        button,
        self
        ):
    """
    Opens a file dialog for the user to upload an audio file, saves the path and name, updates UI elements.
    
    Parameters:
    - graph_name_label: Label widget to display the audio file name.
    - graph_time_label: Label widget to display the audio duration.
    - main_canvas: Canvas widget where the waveform is displayed.
    - graph_x, graph_y, graph_x_end, graph_y_end: Coordinates of the canvas area to plot the waveform.
    - highlight_frame: Frame widget to clear and display new highlights.
    - emotions_hist_canvas : Canvas widget where the spectrum is displayed
    """
    global audio_file_path, audio_file_name, audio_duration, audio_data, sampling_rate, formatted_duration, audio_window, emotion_live
    global current_bar_index , is_analyzed

    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav")], title="Select an Audio File")
    if file_path:
        is_analyzed = False

        for name in emotion_sample:
            emotion_sample[name] = []
            self.emotion_btns[name].configure(fg = "#28374A")
            emotion_live[name] = 0
        bubbles(self)

        # Save file details
        audio_file_path = file_path
        audio_file_name = os.path.basename(file_path)
        data, sr = librosa.load(file_path, offset=0.6 ,sr=16000)
        audio_data = data
        sampling_rate = sr
        
        audio_duration = librosa.get_duration(y=data, sr=sr)
        formatted_duration = format_time(audio_duration)

        # Update UI labels
        graph_name_label.config(text=audio_file_name)
        graph_time_label.config(text=f"🕒: 00:00 / {formatted_duration}")

        # Clear existing highlights and previous waveform
        for widget in highlight_frame.winfo_children():
            widget.destroy()
        main_canvas.delete("audio_waveform")  # Delete any existing waveform tagged with "audio_waveform"

        # Binding functions
        main_canvas.bind("<Button-1>", lambda event: on_canvas_click(self, event, main_canvas, emotions_hist_canvas, button, graph_time_label))
        main_canvas.bind("<B1-Motion>", lambda event: on_mouse_drag(self, event, main_canvas, emotions_hist_canvas, button, graph_time_label))

        current_bar_index = 0
        # Plot the new waveform with the same tag
        plot_waveform_as_bars(self, main_canvas, emotions_hist_canvas, graph_x, graph_y + 20, graph_x_end, graph_y_end - 20, audio_data=audio_data, tag="audio_waveform", button=button)
        # Plot mel scale while reset for the beggining
        create_emotions_hist(emotions_hist_canvas, x=10, y=10, width=180, height=130, reset=True)
        update_persentages(self, True)

def format_time(seconds):
    """Converts a duration in seconds to mm:ss format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02}:{seconds:02}"

def start_recording():
    """Start recording audio."""
    global recording_state
    recording_state["data"] = []
    sd.default.samplerate = recording_state["rate"]
    sd.default.channels = 1

    def callback(indata, frames, time, status):
        recording_state["data"].append(indata.copy())

    recording_state["stream"] = sd.InputStream(callback=callback)
    recording_state["stream"].start()
    recording_state["is_recording"] = True

def stop_recording(target_rms=0.0253):
    """
    Stop recording, normalize the audio to the target RMS, and save the audio file.
    
    Args:
    - target_rms: Desired RMS value for normalization.
    """
    global recording_state, audio_data, audio_duration, sampling_rate
    recording_state["stream"].stop()
    recording_state["stream"].close()
    recording_state["is_recording"] = False

    # Concatenate recorded audio data
    audio_data = np.concatenate(recording_state["data"], axis=0)
    # Flatten the recorded audio to ensure it's 1D
    if len(audio_data.shape) > 1 and audio_data.shape[1] == 1:
        audio_data = audio_data.flatten()

    audio_duration = len(audio_data) / recording_state["rate"]
    sampling_rate = recording_state["rate"]

    # Normalize the audio data to the target RMS
    audio_data = normalize_audio(audio_data, target_rms)

    # Save normalized audio to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, recording_state["rate"], audio_data)
    temp_file.close()

    recording_state["file_path"] = temp_file.name

    print(f"Recording stopped. File saved to {recording_state['file_path']}")

def normalize_audio(audio_data, target_rms):
    """
    Normalize the audio data to match the target RMS.
    
    Args:
    - audio_data: NumPy array containing the audio data.
    - target_rms: Desired RMS value for normalization.
    
    Returns:
    - normalized_audio: NumPy array with adjusted volume.
    """
    # Calculate current RMS
    recorded_rms = np.sqrt(np.mean(audio_data**2))
    
    # Avoid division by zero
    if recorded_rms == 0:
        print("Recorded RMS is zero; returning original audio.")
        return audio_data

    # Calculate gain factor
    gain_factor = target_rms / recorded_rms
    
    # Apply gain factor to normalize
    normalized_audio = audio_data * gain_factor
    
    # Ensure no clipping
    normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
    
    return normalized_audio

def toggle_record(
                  button,
                  canvas, 
                  graph_name_label, 
                  graph_time_label, 
                  graph_bounds, 
                  highlight_frame, 
                  emotions_hist_canvas,
                  play_btn: Button,
                  self
                  ):
    """
    Toggle recording state and update the UI.
    - button: Tkinter Button to update text and color.
    - canvas: Main canvas to plot the waveform.
    - graph_name_label: Label for graph name.
    - graph_time_label: Label for graph time.
    - graph_bounds: (x1, y1, x2, y2) bounds for waveform plotting.
    - clear_highlights_func: Function to clear highlights from UI.
    """
    global sampling_rate ,current_bar_index, is_analyzed , emotion_live, emotion_sample
    if not recording_state["is_recording"]:
        start_recording()
        button.config(text="•Stop Recording", bg="#FF5C5C")
    else:
        stop_recording()
        button.config(text="•Record", bg="#C0C5B1")

        sampling_rate = recording_state["rate"]
        # Update UI
        graph_name_label.config(text="Recorded Audio")
        graph_time_label.config(text=f"🕒: 00:00 / {int(audio_duration // 60):02}:{int(audio_duration % 60):02}")
        x1, y1, x2, y2 = graph_bounds

        # Clear existing highlights and previous waveform
        for widget in highlight_frame.winfo_children():
            widget.destroy()
        
        for name in emotion_sample:
            emotion_sample[name] = []
            self.emotion_btns[name].configure(fg = "#28374A")
            emotion_live[name] = 0
        bubbles(self)
        # Clear previous waveform and plot new one
        canvas.delete("audio_waveform")        

        # Binding functions
        canvas.bind("<Button-1>", lambda event: on_canvas_click(self, event, canvas, emotions_hist_canvas,play_btn, graph_time_label))
        canvas.bind("<B1-Motion>", lambda event: on_mouse_drag(self, event, canvas, emotions_hist_canvas, play_btn, graph_time_label))

        is_analyzed = False
        current_bar_index = 0
        # Plot the new waveform with the same tag
        plot_waveform_as_bars(self, canvas, emotions_hist_canvas, x1, y1 + 20, x2, y2 - 20, audio_data=audio_data, tag="audio_waveform", button=play_btn)
        create_emotions_hist(emotions_hist_canvas, x=10, y=10, width=180, height=130, reset=True)
        update_persentages(self, True)

def reset_graph(self, graph_name_label, graph_time_label, main_canvas, graph_x, graph_y, graph_x_end, graph_y_end, highlight_frame, scroll_canvas, emotions_hist_canvas):
    """
    Resets the graph, clears highlights, and updates the graph name to default.

    Parameters:
    - graph_name_label: Label widget to display the default graph name.
    - graph_time_label: Label widget to reset the time display.
    - main_canvas: Canvas widget where the waveform is displayed.
    - graph_x, graph_y, graph_x_end, graph_y_end: Coordinates of the canvas area to reset the waveform.
    - highlight_frame: Frame widget to clear and display new highlights.
    """
    global current_bar_index, is_zoomable ,audio_data ,sampling_rate
    is_analyzed = False
    audio_data = None
    sampling_rate = None
    current_bar_index = 0
    # Reset graph name and time
    graph_name_label.config(text="Audio Graph")
    graph_time_label.config(text="🕒: 00:00 / 00:00")

    for name in emotion_sample:
        emotion_sample[name] = []
        self.emotion_btns[name].configure(fg = "#28374A")
        bubbles(self)
    update_persentages(self, True)
    reset_highlights_frame(highlight_frame, scroll_canvas)
    create_emotions_hist(emotions_hist_canvas, x=10, y=10, width=180, height=130)
    
    # Reset the graph area
    is_zoomable = False
    main_canvas.delete("audio_waveform")
    reset_plot_waveform_as_bars(main_canvas, graph_x, graph_y+20, graph_x_end, graph_y_end-20)

def play_audio(self, button, canvas, bar_width=1, spacing=1, graph_time: Label = None, emotions_hist_canvas: Canvas = None):
    """
    Toggle play/pause for the audio playback.
    - Plays audio from the `current_bar_index`.
    - Updates the button text to pause (▌▌) while playing and back to play (▶) when paused.
    - Highlights bars on the waveform in sync with audio playback.
    """
    global is_playing, playback_start_time, playback_stream, current_bar_index, volume_level, formatted_duration, emotion_live

    if not is_playing:
        # Calculate the starting sample based on the current_bar_index
        if current_bar_index is None:
            start_sample = 0
        else:
            start_sample = current_bar_index *  plot_step + audio_window["start"]
        # Update the button to pause
        button.config(text="▌▌")
        button.config(font=("Dubai Medium", 7))

        sd.default.samplerate = sampling_rate

        def audio_callback(outdata, frames, time, status):
            nonlocal start_sample
            global volume_level, current_bar_index, formatted_duration, audio_window, percentage_idx
            if status:
                print(f"Playback Status: {status}")

            end_sample = min(len(audio_data), start_sample + frames)
            frame_data = audio_data[start_sample:end_sample].flatten()
            # Calculate the frequency spectrum
            fft_data = np.abs(np.fft.rfft(frame_data))[:15]
            spectrum = fft_data / np.max(fft_data)  # Normalize

            if np.max(np.abs(fft_data)) > 0:  # Ensure max value is non-zero
                spectrum = fft_data / np.max(fft_data)  # Normalize
            else:
                spectrum = np.zeros_like(fft_data)  # Handle case where max is 0
            # Update the visualizer
            update_emotions_hist(emotions_hist_canvas)

            # Scale audio data based on the global volume level
            volume_scale =  volume_level / 100.0
            frame_data = frame_data * volume_scale
            # Update UI labels
            graph_time.config(text=f"🕒: {format_time(start_sample//sampling_rate)} / {formatted_duration}")
            # Ensure the output fits the stream
            outdata[:len(frame_data)] = frame_data.reshape(-1, 1)
            if end_sample >= len(audio_data):
                highlight_bar(self, canvas, emotions_hist_canvas, 1)
                highlight_bar(self, canvas, emotions_hist_canvas, 0)
                create_emotions_hist(emotions_hist_canvas, x=10, y=10, width=180, height=130, reset=True)
                if is_analyzed: 
                    percentage_idx = 0
                    update_persentages(self)
                    for name in emotion_live:
                        emotion_live[name] = 0
                    emotion_live[segment_predictions[percentage_idx]["emotions"][0]] += 1
                stop_playback(button, graph_time)  # Stop if reached the end
                return

            start_sample += frames 
            audio_window["sample_place"] = start_sample
            calc_index = (audio_window["sample_place"] - audio_window["start"]) // plot_step
            # Update the current playing bar index
            if current_bar_index != calc_index:
                highlight_bar(self, canvas, emotions_hist_canvas, calc_index)
                if zooming_level != 0 :
                    update_zoom_graph(canvas)


        # Start the stream with the callback
        playback_stream = sd.OutputStream(callback=audio_callback)
        playback_stream.start()
        playback_start_time = time.time()  # Record system time
        is_playing = True
    else:
        stop_playback(button,None)

def stop_playback(button, graph_time: Label):
    """
    Stop the audio playback and reset button.
    """
    global is_playing, playback_stream
    if playback_stream is not None:
        playback_stream.stop()
        playback_stream.close()
        playback_stream = None
    # Update UI labels
    if graph_time is not None:
        graph_time.config(text=f"🕒: 00:00 / {formatted_duration}")
    button.config(text="▶")
    button.config(font=("Dubai Medium", 14))

    is_playing = False

def create_volume_meter(canvas, x, y, width=180, initial_volume=50):
    """Creates a horizontal volume meter with a line and dot."""
    # Draw the base line for the volume meter
    global volume_level
    line = canvas.create_line(x, y + 10, x + width, y + 10, fill="#28374A", width=2)

    # Calculate initial dot position based on the initial volume level (0-100)
    dot_x = x + (initial_volume / 100) * width
    dot = canvas.create_oval(dot_x - 5, y + 5, dot_x + 5, y + 15, fill="#28374A", outline="")

    # Function to update dot position and volume level
    def move_dot(new_x):
        nonlocal dot_x
        global volume_level
        # Constrain dot within line boundaries
        dot_x = min(max(new_x, x), x + width)
        canvas.coords(dot, dot_x - 5, y + 5, dot_x + 5, y + 15)

        # Calculate volume level based on dot position
        volume_level = int((dot_x - x) / width * 100)

    # Event handler for dragging the dot
    def on_drag(event):
        move_dot(event.x)

    # Event handler for clicking on the line
    def on_click(event):
        move_dot(event.x)

    # Bind events
    canvas.tag_bind(dot, "<B1-Motion>", on_drag)
    canvas.tag_bind(dot, "<Button-1>", on_click)
    canvas.tag_bind(line, "<Button-1>", on_click)    

def stop_audio_playback(self, canvas, button, graph_time, emotions_hist_canvas):
    """
    Stop the audio playback, reset the current bar index, and update UI elements.
    - Resets the play button to "▶".
    - Highlights the first bar on the waveform.
    """
    global is_playing, playback_stream, current_bar_index, percentage_idx, emotion_live

    stop_playback(button, graph_time)
    # Highlight the first bar on the waveform
    highlight_bar(self, canvas, emotions_hist_canvas, 0)
    if is_analyzed: 
        percentage_idx = 0
        update_persentages(self)
        for name in emotion_live:
            emotion_live[name] = 0
        emotion_live[segment_predictions[percentage_idx]["emotions"][0]] += 1
    create_emotions_hist(emotions_hist_canvas, x=10, y=10, width=180, height=130, reset=True)
    current_bar_index = 0  # Reset to the beginning

def zoom_in(self, canvas, emotions_hist_canvas):
    """
    - canvas: The canvas widget to update
    """
    global is_zoomable, zooming_level, zooming_size, zooming_window , graph_cords
    global current_bar_index, plot_step, audio_data
    if not is_zoomable or is_zoomable is None:
        print("Zooming not avaible for this window")
        return
    
    # (current_bar_index * plot_step)/ sampling_rate = index duration
    # sample_rate 
    if 0 <= zooming_level < zooming_size-1:
        zooming_level += 1
    else : return

    audio_window["sample_place"] = current_bar_index * plot_step + audio_window["start"]
    zooming(self, canvas, emotions_hist_canvas)
        
def zoom_out(self ,canvas ,emotions_hist_canvas):
    """
    - canvas: The canvas widget to update
    """
    global is_zoomable, zooming_level, zooming_size, zooming_window
    x1, y1, x2, y2 = graph_cords
    if not is_zoomable or is_zoomable is None:
        print("Zooming not avaible for this window")
        return

    if 0 < zooming_level <= zooming_size-1:
        zooming_level -= 1
    else : return

    if zooming_level == 0:
        canvas.delete("audio_waveform")
        plot_waveform_as_bars(self, canvas, emotions_hist_canvas, x1, y1, x2, y2, audio_data)
        if is_analyzed :
            zoomed_emotions(canvas)
        return
    
    audio_window["sample_place"] = current_bar_index * plot_step + audio_window["start"]
    zooming(self ,canvas, emotions_hist_canvas)

def plot_waveform_window(
        canvas: Canvas, 
        tag="audio_waveform",
        ):
    """
    Plots a simple waveform as vertical bars on a canvas using provided audio data.
    Parameters:
    - canvas: The canvas to draw the waveform on.
    - tag: Tag to associate with the plotted elements for easy deletion.
    """
    global audio_window, plot_step, current_bar_index

    x1, y1, x2, y2 = graph_cords
    canvas.delete("audio_waveform")
    
    delta_line = 0 
    if zooming_window[zooming_level] == 3 or zooming_window[zooming_level] == 10 :
        delta_line = 1
    elif zooming_window[zooming_level] == 30 :
        delta_line = 3
    elif zooming_window[zooming_level] == 60 :
        delta_line = 5
    elif zooming_window[zooming_level] == 300 :  
        delta_line = 30
    elif zooming_window[zooming_level] == 600 :       
        delta_line = 60      
         
    audio_window["data"] = audio_data[audio_window["start"]:audio_window["end"]]
    spacing = int(np.ceil(zooming_window[zooming_level] / delta_line))


    # calulating width for the time indicators
    window_width = x2
    for index, x in enumerate(np.linspace(x1, window_width , spacing + 1)):
        if x == x1 or x == window_width:
            continue
        canvas.create_line(x, y1, x, y2, fill="#36424C", dash=(2, 2),
                                tags=tag)
        canvas.create_text(x, y2 - 10, text=f"{format_time(delta_line * index + audio_window['start']//sampling_rate)}",
                                fill="#506170", font=("Dubai Medium", 10), tags=tag)


    middle_y = (y1 + y2) / 2  # Vertical center for the waveform display
    # Set dimensions and limits
    width = x2 - x1
    height = y2 - y1 -5
    bar_width = 1  # Width of each bar
    spacing = 1  # Space between bars
    max_bars = width // (bar_width + spacing)  # Maximum number of bars

    
    try:
        step = max(1, len(audio_window["data"]) // max_bars)
        plot_step = step
        downsampled_data = audio_window["data"][::step][:max_bars]
        normalized_data = downsampled_data / max_amplitude  # Normalize data
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
    
    # Plot bars
    for i, sample in enumerate(normalized_data):
        bar_height = float(sample * (height / 2))  # Scale height to canvas
        line_x = x1 + i * (bar_width + spacing)
        # Top and Bottom parts are the same for simplicity
        canvas.create_rectangle(line_x, middle_y - bar_height, line_x + bar_width,
                                 middle_y + bar_height, fill="#B5EBE9", outline="",
                                   tags=(tag, f"bar_{i}"))

def update_zoom_graph(canvas):
    global audio_window, current_bar_index
    change = False  
    if ((zooming_window[zooming_level]*2)//3) * sampling_rate < audio_window["sample_place"] < audio_len - ((zooming_window[zooming_level]*2)//3) * sampling_rate :
        # case the sample place is in good position create window around 
        if current_bar_index >= 223:
            change = True
            audio_window["start"] = audio_window["sample_place"] - (zooming_window[zooming_level]//3) * sampling_rate
            audio_window["end"] = audio_window["sample_place"] + ((zooming_window[zooming_level]*2)//3) * sampling_rate
    elif audio_window["sample_place"] <= ((zooming_window[zooming_level]*2)//3) * sampling_rate:
        # case window is on the left part, making room to the right
        audio_window["start"] = 0
        audio_window["end"] = zooming_window[zooming_level] * sampling_rate
        change = False
    elif audio_window["sample_place"] >= audio_len - ((zooming_window[zooming_level]*2)//3) * sampling_rate:
        # case window is on the right part, making room to the left
        audio_window["start"] = audio_len - zooming_window[zooming_level] * sampling_rate
        audio_window["end"] = audio_len
        change = False

    if change and is_playing:
        plot_waveform_window(canvas)
        if is_analyzed :
            zoomed_emotions(canvas)
        
def zooming(self, canvas, emotions_hist_canvas):
    #function for the zooming functions
    global audio_window, current_bar_index

    round_window = round(zooming_window[zooming_level]/2)
    if round_window * sampling_rate < audio_window["sample_place"] < audio_len - (round_window) * sampling_rate :
        # case the sample place is in good position create window around 
        audio_window["start"] = int(audio_window["sample_place"] - (round_window) * sampling_rate)
        audio_window["end"] = int(audio_window["sample_place"] + (round_window) * sampling_rate)
        
    elif audio_window["sample_place"] <= (round_window) * sampling_rate:
        # case window is on the left part, making room to the right
        audio_window["start"] = 0
        audio_window["end"] = zooming_window[zooming_level] * sampling_rate
        
    elif audio_window["sample_place"] >= audio_len - (round_window) * sampling_rate:
        # case window is on the right part, making room to the left
        audio_window["start"] = audio_len - zooming_window[zooming_level] * sampling_rate
        audio_window["end"] = audio_len

    plot_waveform_window(canvas)
    if is_analyzed :
        zoomed_emotions(canvas)
    highlight_bar(self, canvas, emotions_hist_canvas, round((audio_window["sample_place"] - audio_window["start"]) // plot_step))

def create_emotions_hist(canvas: Canvas, x: int, y: int, width: int, height: int, bars: int = 7, spacing: int = 5, reset: bool = False):
    """
    Draws a vertical spectrum visualizer with a specified number of bars and symbols above each bar.
    
    Parameters:
    - canvas: The canvas on which to draw the visualizer.
    - x, y: Starting position of the visualizer.
    - width: Total width of the visualizer.
    - height: Total height of the visualizer.
    - bars: Number of bars in the visualizer.
    - spacing: Spacing between bars.
    """
    bar_width = (width - (bars - 1) * spacing) // bars  # Calculate bar width based on available space
    bar_height = height

    # Emojis or symbols corresponding to each emotion
    symbols = ["😠", "🤢", "😨", "😊", "😐", "😢", "😲"]

    # Clear previous bars and symbols
    canvas.delete("spectrum_bar")
    canvas.delete("spectrum_symbol")

    for i, name in enumerate(emotion_labels):
        bar_x = x + i * (bar_width + spacing)
        if reset:
            bar_y_end = y + bar_height  # Uniform height for reset state
        else:
            bar_y_end = y + np.random.uniform(0.1, 0.9) * bar_height  # Simulate random amplitude for now

        # Draw bar
        canvas.create_rectangle(bar_x, y + bar_height, bar_x + bar_width, bar_y_end, fill=emotion_colors[name],outline=emotion_colors[name], tags=("spectrum_bar", f"spec_bar_{name}"))

        # Add symbol above the bar
        canvas.create_text(
            bar_x + bar_width / 2,  # Center the symbol horizontally
            bar_y_end - 15,  # Position slightly above the top of the bar area
            text=symbols[i],  # Get the symbol for the corresponding emotion
            font=("Arial", 14),
            fill="#28374A",  # White color for the symbol
            tags=("spectrum_symbol", f"symbol_{name}")
        )

def update_emotions_hist(canvas):
    """
    Updates the spectrum visualizer with the current sum of emotions.

    Parameters:
    - canvas: The canvas where the bars and symbols are drawn.
    - emotion_counts: A dictionary where keys are emotion labels and values are the counts of each emotion.
    """
    global emotion_labels, emotion_colors, emotion_live

    # Parameters for bar layout
    x, y = 10, 10
    width = 180
    height = 130
    bars = len(emotion_labels)
    spacing = 5
    bar_width = (width - (bars - 1) * spacing) // bars  # Calculate bar width
    max_count = max(max(emotion_live.values()) ,1)   # Avoid division by zero

    for i, emotion in enumerate(emotion_labels):
        bar_x = x + i * (bar_width + spacing)

        # Normalize the bar height to the max_count
        normalized_height = (emotion_live.get(emotion, 0) / max_count) * height *0.9

        # Update the bar
        bar_y_end = y + height - normalized_height 
        canvas.coords(f"spec_bar_{emotion}", bar_x, y + height, bar_x + bar_width, bar_y_end)

        # Update the symbol position
        canvas.coords(
            f"symbol_{emotion}",
            bar_x + bar_width / 2,
            bar_y_end - 15,  # Position slightly above the top of the bar area
        )

def reset_highlights_frame(highlight_frame, scroll_canvas):
    """
    Reset the highlights frame by clearing all widgets and updating the scroll region.

    Args:
        highlight_frame (Frame): The frame containing the highlight widgets.
        scroll_canvas (Canvas): The canvas used for scrolling.
        scrollbar (Scrollbar): The scrollbar controlling the canvas.
    """
    # Reset the scroll position to the top
    scroll_canvas.yview_moveto(0)

    # Clear previous highlights
    for widget in highlight_frame.winfo_children():
        widget.destroy()

    minutes, seconds = divmod(1 * 3, 60)
    time_str = f"{minutes:02}:{seconds:02}"
            
    # Adjusted "Time", "Emotion", and "Confidence" columns with smaller width
    Label(highlight_frame, text=time_str, font=("Dubai Medium", 10), bg="#212E38", fg="white", width=6, anchor="w").grid(row=1*2, column=0, padx=(2, 0), sticky="w")
    Label(highlight_frame, text="Example", font=("Dubai Medium", 10), bg="#212E38", fg="white", width=8, anchor="w").grid(row=1*2, column=1, padx=(2, 0), sticky="w")
    Label(highlight_frame, text=f"80.{1}%", font=("Dubai Medium", 10), bg="#212E38", fg="white", width=10, anchor="w").grid(row=1*2, column=2, padx=(2, 0), sticky="w")

def extract_emotion_predictions_from_data(
        self,
        canvas, 
        emotions_hist_canvas,
        highlight_frame, 
        scroll_canvas, 
        play_btn, 
        graph_time,
        segment_duration=3,
        overlap_duration=0.5, 
        max_length=110):
    """
    Extract emotion predictions and their classification percentages for raw audio data.

    Args:
    - segment_duration: Duration (in seconds) of each audio segment.
    - max_length: Maximum length for padding features.

    Returns:
    - overall_emotion: The combined emotion prediction for the entire audio data.
    - segment_predictions: List of dictionaries containing emotions and their percentages for each segment.
    """
    global audio_data, sampling_rate, overall_emotion, segment_predictions, zooming_level, is_analyzed, current_bar_index ,ploted_emotion, emotion_live, percentage_idx

    for key in emotion_sample:
        emotion_sample[key] = []
    for name in emotion_live:
        emotion_live[name] = 0
    percentage_idx = 0
    stop_playback(play_btn, graph_time)

    try:
        # Ensure audio data and sampling rate are available
        if audio_data is None or sampling_rate is None:
            raise ValueError("Audio data or sampling rate is missing.")

        overall_emotion, segment_predictions = None, None

        reset_highlights_frame(highlight_frame, scroll_canvas)
        # Clear previous highlights
        for widget in highlight_frame.winfo_children():
            widget.destroy()

        if is_zoomable and zooming_level != 0 :
            zooming_level = 0
            x1, y1, x2, y2 = graph_cords
            canvas.delete("audio_waveform")
            plot_waveform_as_bars(self, canvas, emotions_hist_canvas, x1, y1, x2, y2, audio_data)
        else :
            current_bar_index = 0

        # Split audio into fixed-duration segments
        segments = p2.split_audio(audio_data, sampling_rate, segment_duration, overlap_duration)
        # Extract features for each segment
        features = p2.get_features_segments(segments, sampling_rate)
        # Pad and scale features for model input
        p2.padFeature(features, max_length)
        scaled_features = p2.applyScaler(features)
        # Predict emotions for each segment 
        predictions = model.predict(np.array(scaled_features))
        # Decode predictions into human-readable emotion labels
        decoded_predictions = encoder.inverse_transform(predictions)
        # Calculate overall emotion
        overall_emotion = p2.combine_predictions(decoded_predictions)


        # Convert predictions to percentages
        segment_predictions = []
        timestamp = segment_duration  # Start at the first segment duration
        
        # index = 0
        # last_precentage = 0
        for idx, prediction in enumerate(predictions):
            percentages = (prediction / np.sum(prediction)) * 100  # Normalize to percentage
            emotions_with_percentages = list(zip(encoder.categories_[0].tolist(), percentages.tolist()))

            # Sort emotions by percentage in descending order
            sorted_emotions = sorted(emotions_with_percentages, key=lambda x: x[1], reverse=True)
            if timestamp <= audio_duration:
                emotion_sample[sorted_emotions[0][0]].append(timestamp)
            else:
                emotion_sample[sorted_emotions[0][0]].append(audio_duration)
                timestamp = audio_duration

            segment_predictions.append({
                "timestamp": timestamp,
                "emotions": [emotion for emotion, _ in sorted_emotions],
                "percentages": [percentage for _, percentage in sorted_emotions]
            })

            

            if sorted_emotions[0][1] >= threshold_value and timestamp <= audio_duration:
                # Format timestamp into mm:ss
                minutes, seconds = divmod(int(timestamp), 60)
                time_str = f"{minutes:02}:{seconds:02}"

                # Add clickable Time Label
                time_label = Label(
                    highlight_frame,
                    text=time_str,
                    font=("Dubai Medium", 10),
                    bg="#212E38",
                    fg="white",
                    width=6,
                    anchor="w"
                )
                time_label.grid(row=idx * 2, column=0, padx=(2, 0), sticky="w")
                time_label.bind(
                    "<Button-1>",
                    lambda event, timestamp = timestamp : highlight_jump(self, timestamp, canvas, emotions_hist_canvas)
                )

                # Add clickable Emotion Label
                emotion_label = Label(
                    highlight_frame,
                    text=sorted_emotions[0][0].capitalize(),
                    font=("Dubai Medium", 10),
                    bg="#212E38",
                    fg="white",
                    width=8,
                    anchor="w"
                )
                emotion_label.grid(row=idx * 2, column=1, padx=(2, 0), sticky="w")
                emotion_label.bind(
                    "<Button-1>",
                    lambda event, timestamp = timestamp : highlight_jump(self, timestamp, canvas, emotions_hist_canvas)
                )

                # Add clickable Confidence Label
                confidence_label = Label(
                    highlight_frame,
                    text=f"{sorted_emotions[0][1]:.1f}%",
                    font=("Dubai Medium", 10),
                    bg="#212E38",
                    fg="white",
                    width=10,
                    anchor="w"
                )
                confidence_label.grid(row=idx * 2, column=2, padx=(2, 0), sticky="w")
                confidence_label.bind(
                    "<Button-1>",
                    lambda event, timestamp = timestamp : highlight_jump(self, timestamp, canvas, emotions_hist_canvas)
                )

                # Separator line
                Canvas(
                    highlight_frame,
                    height=1,
                    width=180,
                    bg="#3A3A3A",
                    highlightthickness=0
                ).grid(row=idx * 2 + 1, columnspan=3, padx=5, pady=2, sticky="ew")

            # Increment timestamp by 2.5 seconds for the next segment
            timestamp += segment_duration - overlap_duration

        bubbles(self)
        for emotion, button in self.emotion_btns.items() :
            button.configure(fg = emotion_colors[emotion])
        zoomed_emotions(canvas)
        is_analyzed = True
        update_persentages(self)
        emotion_live[segment_predictions[percentage_idx]["emotions"][0]] += 1


    except Exception as e:
        print(f"Error extracting emotion predictions: {e}")
        return 
    
def zoomed_emotions(canvas):
    global ploted_emotion

    ploted_emotion = ['#B5EBE9'] * 301
    index = 0
    last_precentage = 0
    start_time = audio_window["start"] // sampling_rate
    end_time = audio_window["end"] // sampling_rate
    
    # running over all of the sorted predictions
    for prediction in segment_predictions:
        tmp_index = int(((prediction['timestamp']- start_time) * sampling_rate)//plot_step)

        # check for prediction in the time range
        if  start_time <= prediction['timestamp'] <= end_time:
            # print(f"start - {start_time}")
            # print(f"timing - {(prediction['timestamp'] * plot_step) // sampling_rate}")
            # print(f"tmp_index = {tmp_index}")
            # print(f"timestemp - {prediction['timestamp']}")
            # print(f"timstap real plot - {((prediction['timestamp']- start_time)*sampling_rate) / plot_step}")
            # print(f"emotion - {prediction['emotions'][0]}")
            # print(f"index retracktion - {(tmp_index * plot_step) / sampling_rate}")
            # setting the index for the color inside the window
            # print(f"tmp_index = {tmp_index}")
            # check if the index is going to be the same dou to size
            if  tmp_index == index and prediction['percentages'][0] > last_precentage:
                highligh_emotion_bar(canvas, tmp_index, emotion_colors[prediction['emotions'][0]])
                #print(f"prediction['percentages'][0] = {prediction['percentages'][0]}")
                #ploted_emotion.pop()
                ploted_emotion[tmp_index] = emotion_colors[prediction['emotions'][0]]
                #print(f"emotion_colors[prediction['emotions'][0]] = {emotion_colors[prediction['emotions'][0]]}")
            elif tmp_index != index:
                highligh_emotion_bar(canvas, tmp_index, emotion_colors[prediction['emotions'][0]])
                ploted_emotion[tmp_index] = emotion_colors[prediction['emotions'][0]]
                #print(f"emotion_colors[prediction['emotions'][0]] = {emotion_colors[prediction['emotions'][0]]}")
            else:
                pass # do nothing
        index = tmp_index
        last_precentage = prediction['percentages'][0]

def bubbles(self):
    for name in emotion_labels:
        bubble_canvas = self.bubbles[name]
        text_item_id = bubble_canvas.find_all()[1]  # Get the second item (the text)
        bubble_canvas.itemconfig(text_item_id, text=str(np.size(emotion_sample[name])))

def highlight_jump(self, timestamp, canvas, emotions_hist_canvas):
    if is_zoomable:
        while zooming_level != 0:
            zoom_out(self, canvas, emotions_hist_canvas)
        highlight_bar(self, canvas, emotions_hist_canvas, ((timestamp)*sampling_rate)//plot_step)
        while zooming_window[zooming_level] != 10:
            zoom_in(self, canvas, emotions_hist_canvas)
    else:
        highlight_bar(self, canvas, emotions_hist_canvas,((timestamp - 3)*sampling_rate)//plot_step)

def toggle_emotion(self, name, canvas):
    if is_analyzed:
        colors = {
            "angry": "#F12E2E",
            "disgust": "#00C605",
            "fear": "#B115AD",
            "happy": "#FFD858",
            "neutral": "#969A8D",
            "sad": "#2E35F1",
            "surprise": "#F1882E"
        }
        color = self.emotion_btns[name].cget("fg")

        if color == "#28374A":
            emotion_colors[name] = colors[name]
        else:
            emotion_colors[name] = "#28374A"
        self.emotion_btns[name].configure(fg = emotion_colors[name])
        zoomed_emotions(canvas)
    else:
        pass

def update_persentages(self, delete = False):
    if delete:
        for index ,name in enumerate(emotion_labels):
            self.emotion_precentage[name].configure(text = "0.00%")
    else:
        for index ,name in enumerate(segment_predictions[percentage_idx]["emotions"]):
            self.emotion_precentage[name].configure(text = f"{segment_predictions[percentage_idx]['percentages'][index]:.2f}%")

def adjust_threshold(change, self, main_canvas, emotions_hist_canvas, threshold_text, scroll_canvas, highlight_frame):
    """
    Adjust the threshold value and update the displayed text.

    Args:
    - change: Integer, +1 for increasing, -1 for decreasing the threshold.
    - canvas: The canvas on which the threshold text is displayed.
    - text_id: The ID of the text element to update.
    """
    global threshold_value

    # Adjust the threshold value
    threshold_value = max(0, min(100, threshold_value + change))  # Ensure it stays between 0% and 100%

    # Update the text on the canvas
    main_canvas.itemconfig(threshold_text, text=f"Threshold: {threshold_value}%")

    if is_analyzed:
        # Reset the scroll position to the top
        scroll_canvas.yview_moveto(0)

        # Clear previous highlights
        for widget in highlight_frame.winfo_children():
            widget.destroy()

        for idx, prediction in enumerate(segment_predictions):
            # Format timestamp into mm:ss
            precentages = prediction["percentages"][0]
            timestamp = prediction["timestamp"]
            if idx == 0 and timestamp <= audio_duration:
                timestamp = audio_duration
            if precentages >= threshold_value and timestamp <= audio_duration:
                if idx == 0 : timestamp = prediction["timestamp"]
                minutes, seconds = divmod(int(timestamp), 60)
                time_str = f"{minutes:02}:{seconds:02}"

                # Add clickable Time Label
                time_label = Label(
                    highlight_frame,
                    text=time_str,
                    font=("Dubai Medium", 10),
                    bg="#212E38",
                    fg="white",
                    width=6,
                    anchor="w"
                )
                time_label.grid(row=idx * 2, column=0, padx=(2, 0), sticky="w")
                time_label.bind(
                    "<Button-1>",
                    lambda event, timestamp = timestamp : highlight_jump(self, timestamp, main_canvas, emotions_hist_canvas)
                )

                # Add clickable Emotion Label
                emotion_label = Label(
                    highlight_frame,
                    text=prediction['emotions'][0].capitalize(),
                    font=("Dubai Medium", 10),
                    bg="#212E38",
                    fg="white",
                    width=8,
                    anchor="w"
                )
                emotion_label.grid(row=idx * 2, column=1, padx=(2, 0), sticky="w")
                emotion_label.bind(
                    "<Button-1>",
                    lambda event, timestamp = timestamp : highlight_jump(self, timestamp, main_canvas, emotions_hist_canvas)
                )

                # Add clickable Confidence Label
                confidence_label = Label(
                    highlight_frame,
                    text=f"{precentages:.1f}%",
                    font=("Dubai Medium", 10),
                    bg="#212E38",
                    fg="white",
                    width=10,
                    anchor="w"
                )
                confidence_label.grid(row=idx * 2, column=2, padx=(2, 0), sticky="w")
                confidence_label.bind(
                    "<Button-1>",
                    lambda event, timestamp = timestamp : highlight_jump(self, timestamp, main_canvas, emotions_hist_canvas)
                )

                # Separator line
                Canvas(
                    highlight_frame,
                    height=1,
                    width=180,
                    bg="#3A3A3A",
                    highlightthickness=0
                ).grid(row=idx * 2 + 1, columnspan=3, padx=5, pady=2, sticky="ew")

        
