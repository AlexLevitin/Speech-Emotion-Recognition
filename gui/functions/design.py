from tkinter import Canvas
import numpy as np
import librosa
import librosa.display
from PIL import Image
import matplotlib.pyplot as plt


def create_rounded_rectangle(canvas: Canvas, x1: int, y1: int, x2: int, y2: int, radius=20, color="#C0C5B1"):
    """Creates a rounded rectangle on a canvas by combining ovals and rectangles."""
    canvas.create_oval(x1, y1, x1 + radius*2, y1 + radius*2, fill=color, outline="")
    canvas.create_oval(x2 - radius*2, y1, x2, y1 + radius*2, fill=color, outline="")
    canvas.create_oval(x1, y2 - radius*2, x1 + radius*2, y2, fill=color, outline="")
    canvas.create_oval(x2 - radius*2, y2 - radius*2, x2, y2, fill=color, outline="")
    canvas.create_rectangle(x1 + radius, y1, x2 - radius, y2, fill=color, outline="")
    canvas.create_rectangle(x1, y1 + radius, x2, y2 - radius, fill=color, outline="")

def draw_grid(canvas: Canvas, width: int, height: int, interval=50):
    """Draws a grid with labels to assist in layout design."""
    for x in range(0, width, interval):
        canvas.create_line(x, 0, x, height, fill="red", dash=(2, 4))
        canvas.create_text(x + 5, 10, text=str(x), anchor="nw", fill="red", font=("Arial", 8),)

    for y in range(0, height, interval):
        canvas.create_line(0, y, width, y, fill="red", dash=(2, 4))
        canvas.create_text(5, y + 5, text=str(y), anchor="nw", fill="red", font=("Arial", 8))

def reset_graph_grid(canvas: Canvas, x1: int, y1: int, x2: int, y2: int, spacing: int = 200, color: str = "#36424C", tag=""):
    """Draws a grid within the specified rectangle area on the canvas."""
    # Draw vertical lines
    for x in range(x1, x2, spacing):
        canvas.create_line(x, y1, x, y2, fill=color, dash=(2, 2), tags=tag)

    # # Draw horizontal lines
    # for y in range(y1-3, y2+3, 15):
    #     canvas.create_line(x1, y, x2, y, fill=color, dash=(2, 2), tags=tag)

def reset_plot_waveform_as_bars(canvas: Canvas, x1: int, y1: int, x2: int, y2: int, tag="audio_waveform"):
    """Plots a simple waveform as vertical bars on a canvas."""
    
    # Draw vertical lines
    for index ,x in enumerate(np.linspace(x1, x2, 4)):
        if x == x1 or x == x2:
            continue
        canvas.create_line(x, y1, x, y2, fill="#36424C", dash=(2, 2), tags=tag)
        canvas.create_text(x, y2 - 10, text=f"00:0{index}", fill="#506170", font=("Dubai Medium", 10), tags=tag)

    # Set dimensions and limits
    width = x2 - x1
    height = y2 - y1 -5
    bar_width = 1  # Width of each bar
    spacing = 1  # Space between bars
    max_bars = width // (bar_width + spacing)  # Maximum number of bars
    normalized_data = np.full(max_bars, 0.03)  # Silent waveform placeholder

    middle_y = (y1 + y2) / 2  # Vertical center for the waveform display

    # Plot bars
    for i, sample in enumerate(normalized_data):
        bar_height = sample * (height / 2)  # Scale height to canvas
        line_x = x1 + i * (bar_width + spacing)
        # Top and Bottom parts are the same for simplicity
        canvas.create_rectangle(line_x, middle_y - bar_height, line_x + bar_width, middle_y + bar_height, fill="#B5EBE9", outline="", tags=tag)




# def plot_waveform_as_bars(canvas: Canvas, x1: int, y1: int, x2: int, y2: int, file_path: str = None, no_input: bool = False, tag="audio_waveform"):    
#     """Plots a simple waveform as vertical bars on a canvas."""
#     # Draw a grid in the background
#     graph_grid(canvas, x1, y1, x2, y2, spacing=20, color="#36424C", tag=tag)

#     # Set dimensions and limits
#     width = x2 - x1
#     height = y2 - y1 -5
#     bar_width = 1  # Width of each bar
#     spacing = 1  # Space between bars
#     max_bars = width // (bar_width + spacing)  # Maximum number of bars

#     if no_input or not file_path:
#         normalized_data = np.full(max_bars, 0.03)  # Silent waveform placeholder
#     else:
#         try:
#             data, sampling_rate = librosa.load(file_path, sr=44100)
#             step = max(1, len(data) // max_bars)
#             downsampled_data = data[::step][:max_bars]
#             max_amplitude = np.max(np.abs(downsampled_data))
#             normalized_data = downsampled_data / max_amplitude  # Normalize data
#         except Exception as e:
#             print(f"Error loading audio file: {e}")
#             return

#     middle_y = (y1 + y2) / 2  # Vertical center for the waveform display

#     # Plot bars
#     for i, sample in enumerate(normalized_data):
#         bar_height = sample * (height / 2)  # Scale height to canvas
#         line_x = x1 + i * (bar_width + spacing)
#         # Top and Bottom parts are the same for simplicity
#         canvas.create_rectangle(line_x, middle_y - bar_height, line_x + bar_width, middle_y + bar_height, fill="#B5EBE9", outline="", tags=tag)





