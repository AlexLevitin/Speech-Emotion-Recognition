from tkinter import Canvas, Frame, Button, Label, PhotoImage, Scrollbar, VERTICAL, Y
from pathlib import Path
from PIL import Image, ImageTk
from ..functions.design import *# ,create_rounded_rectangle, draw_grid, plot_waveform_as_bars, create_volume_meter   # type: ignore
from ..functions.audio_functions import * # type: ignore


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"../assets/analyze_assets")  

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class AnalyzeFrame(Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.configure(bg="#292B34", height=506, width=1012)

        # Postion Variable
        graph_x, graph_y = 201, 6
        graph_x_end, graph_y_end = 800, 300

        # Main canvas for drawing content
        main_canvas = Canvas(
            self,
            bg="#292B34",
            height=506,
            width=1012,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        main_canvas.place(x=0, y=0)

        # Bright rounded inner canvas
        create_rounded_rectangle(
            main_canvas,
            x1=5, y1=5, x2=1007, y2=501,  # Adjusted height to fit inside main canvas
            radius=5,
            color="#C0C5B1"
        )

        # draw_grid(main_canvas, 1007, 501, interval=50)  # Adjust interval for spacing

        # Control Panel on the Left
        Label(self, text="Control Panel", bg="#C0C5B1", fg="#292B34", font=("Dubai Medium", 25),).place(x=6, y=6, width=193, height=93)
        
        # Buttons for Control Paner on the left
        # Upload Button
        upload_button = Button( self, text="•Upload", font=("Dubai Medium", 14), fg="#292B34", bg="#C0C5B1", bd=0,
                command=lambda: upload_audio_file(
                graph_name,
                graph_time,
                main_canvas,
                graph_x, graph_y,
                graph_x_end, graph_y_end,
                highlight_frame,
                spectrum_canvas,
                play_button,
                self
            ))
        upload_button.place(x=5, y=101,width=194,height=149-100)

        # Record Button
        record_button = Button(
            self, 
            text="•Record", 
            bg="#C0C5B1", 
            fg="#292B34", 
            bd=0, 
            font=("Dubai Medium", 14),
            command=lambda: toggle_record(
                button=record_button, 
                canvas=main_canvas, 
                graph_name_label=graph_name, 
                graph_time_label=graph_time, 
                graph_bounds=(graph_x, graph_y , graph_x_end, graph_y_end), 
                highlight_frame = highlight_frame,
                spectrum_canvas= spectrum_canvas,
                play_btn=play_button,
                self=self
            ))
        record_button.place(x=5, y=151,width=194,height=199-150)

        # Delete Button
        delete_button = Button(
            self,
            text="•Delete", 
            bg="#C0C5B1", 
            fg="#292B34", 
            bd=0, 
            font=("Dubai Medium", 14),
            command=lambda: reset_graph(
                self=self,
                graph_name_label=graph_name,
                graph_time_label=graph_time,
                main_canvas=main_canvas,
                graph_x=graph_x,
                graph_y=graph_y,
                graph_x_end=graph_x_end,
                graph_y_end=graph_y_end,
                highlight_frame=highlight_frame,
                scroll_canvas=scroll_canvas
            ))
        delete_button.place(x=5, y=201,width=194, height=249-200)
        # Lines to wrap Control Panel
        main_canvas.create_line(200,5,200,501,fill="#969A8D")
        main_canvas.create_line(5,100,200,100,fill="#969A8D")
        main_canvas.create_line(5,150,200,150,fill="#969A8D")
        main_canvas.create_line(5,200,200,200,fill="#969A8D")
        main_canvas.create_line(5,250,200,250,fill="#969A8D")

        # Sound Part
        create_rounded_rectangle(
            main_canvas,
            x1=graph_x, y1=graph_y, x2=graph_x_end, y2=graph_y_end,  # Adjusted height to fit inside main canvas
            radius=15,
            color="#28374A"
        )

        # Sound Graph (Navy Blue Background) with Playback and Time Display
        sound_graph_bg = "#292B34"
        sound_graph = main_canvas.create_rectangle(graph_x, graph_y+20, graph_x_end, graph_y_end-20, fill="#212E38", outline="")
        # Display initial "silent" waveform (no input)
        # Draw horizontal lines
        for y in range(graph_y-3, graph_y_end+3, 15):
            if y == 3 or y == 18 or y == 288:
                continue
            main_canvas.create_line(graph_x, y, graph_x_end, y, fill="#36424C", dash=(2, 2))
        reset_plot_waveform_as_bars(main_canvas, graph_x, graph_y+20, graph_x_end, graph_y_end-20)
        #plot_waveform_as_bars(main_canvas, graph_x, graph_y+20, graph_x_end, graph_y_end-20, no_input=True)

        # Load, resize, and set up zoom-in icon
        original_zoom_in = Image.open(relative_to_assets("zoom-in.png"))
        resized_zoom_in = original_zoom_in.resize((15,15), Image.LANCZOS)
        self.zoom_in_icon = ImageTk.PhotoImage(resized_zoom_in)

        # Create Zoom In button
        zoom_in_button = Button(
            self,
            image=self.zoom_in_icon,
            bg="#28374A",
            activebackground="#28374A",
            bd=0,
            highlightthickness=0,
            command=lambda: zoom_in(main_canvas)
        )
        zoom_in_button.place(x=graph_x_end - 25, y=282)

        # Load, resize, and set up zoom-out icon
        original_zoom_out = Image.open(relative_to_assets("zoom-out.png"))
        resized_zoom_out = original_zoom_out.resize((15,15), Image.LANCZOS)
        self.zoom_out_icon = ImageTk.PhotoImage(resized_zoom_out)

        # Create Zoom Out button
        zoom_out_button = Button(
            self,
            image=self.zoom_out_icon,
            bg="#28374A",
            activebackground="#28374A",
            bd=0,
            highlightthickness=0,
            command=lambda: zoom_out(main_canvas)
        )
        zoom_out_button.place(x=graph_x_end - 50, y=282)

        # Graph name label
        graph_name = Label(
            self, 
            text="Audio Graph", 
            bg="#28374A", 
            fg="#B5EBE9", 
            font=("Dubai Medium", 12),
            highlightthickness=0,
            anchor="center",
            padx=0,  
            pady=0   
        )
        graph_name.place(x=graph_x+30 ,y=graph_y, width=graph_x_end-graph_x-60,height=20)

        # Create time label representing mm:ss
        graph_time = Label(
            self, 
            text="🕒: 00:00 / 00:00", 
            bg="#28374A", 
            fg="#B5EBE9", 
            font=("Dubai Medium", 10),
            highlightthickness=0,
            padx=0,  
            pady=0   
        )
        graph_time.place(x= (graph_x+graph_x_end)/2 - 75 , y=280,height=20)

        #Symbols to Work With, Play - ▶ , Pause - ▌▌ , Stop - ■
        # Create a button with the play symbol
        play_button = Button(
            self,
            text="▶",
            font=("Dubai Medium", 14),
            bg="#28374A",
            activebackground="#28374A",
            fg="#B5EBE9",
            bd=0,
            highlightthickness=0,
            command=lambda: play_audio(self, play_button, main_canvas, bar_width=1, spacing=1, graph_time=graph_time, spectrum_canvas=spectrum_canvas)
        )
        play_button.place(x=graph_x + 8, y=280,height=20, width=20)

        # Create a button with the stop symbol
        stop_button = Button(
            self,
            text="■",
            font=("Dubai Medium", 14),
            bg="#28374A",
            activebackground="#28374A",
            fg="#B5EBE9",
            bd=0,
            highlightthickness=0,
            command=lambda: stop_audio_playback(self, main_canvas, play_button, graph_time, spectrum_canvas)
        )
        stop_button.place(x=graph_x + 25, y=280,height=20)        

        # Highlights Section
        create_rounded_rectangle(
            main_canvas,
            x1=graph_x_end+2, y1=graph_y, x2=1002, y2=graph_y_end,  # Adjusted height to fit inside main canvas
            radius=5,
            color="#28374A"
        )
        main_canvas.create_rectangle(graph_x_end+2, graph_y+20, 1002, graph_y_end-20, fill="#212E38", outline="")
        
        # Highlights Heading
        Label(self, 
            text="Highlights", 
            font=("Dubai Medium", 12), 
            bg="#28374A", 
            fg="#B5EBE9", 
            highlightthickness=0).place(x=865, y=graph_y, height=20, width=70)

        # Columns Names Title for the Table Under Highlights
        headings = ["Time", "Emotion", "Confidence"]
        heading_x_positions = [graph_x_end+3, graph_x_end+55, graph_x_end+120]  # Adjust x positions for each column heading

        # Create each column heading label
        for i, heading in enumerate(headings):
            Label(
                self, 
                text=heading, 
                font=("Dubai Medium", 10), 
                bg="#28374A", 
                fg="#B5EBE9"
            ).place(x=heading_x_positions[i], y=graph_y_end-20, height=20, width=60)

        # Scrollable Frame for Highlights
        scroll_canvas = Canvas(self, width=170, height=graph_y_end-graph_y-40, bg="#212E38", bd=0, highlightthickness=0)
        scrollbar = Scrollbar(self, orient=VERTICAL, command=scroll_canvas.yview)
        highlight_frame = Frame(scroll_canvas, bg="#212E38")

        # Ensure the frame resizes dynamically but has a minimum height
        def update_scroll_region(event):
            scroll_canvas.configure(scrollregion=(0, 0, event.width, max(event.height, graph_y_end - graph_y - 40)))

        highlight_frame.bind("<Configure>", update_scroll_region)
        scroll_canvas.create_window((0, 0), window=highlight_frame, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        # Place scroll_canvas and scrollbar within specified y-range
        scroll_canvas.place(x=graph_x_end+5, y=graph_y+20, width=170, height=graph_y_end-graph_y-40)
        scrollbar.place(x=1002-15, y=graph_y+20, height=graph_y_end-graph_y-40)

        # Enable mouse wheel scrolling
        def on_mouse_wheel(event):
            scroll_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        scroll_canvas.bind_all("<MouseWheel>", on_mouse_wheel)

        # Populate highlight_frame with example data, formatted as mm:ss
        for i in range(30):  # Adding 30 example rows
            minutes, seconds = divmod(i * 3, 60)
            time_str = f"{minutes:02}:{seconds:02}"
            
            # Adjusted "Time", "Emotion", and "Confidence" columns with smaller width
            Label(highlight_frame, text=time_str, font=("Dubai Medium", 10), bg="#212E38", fg="white", width=6, anchor="w").grid(row=i*2, column=0, padx=(5, 0), sticky="w")
            Label(highlight_frame, text="Example", font=("Dubai Medium", 10), bg="#212E38", fg="white", width=8, anchor="w").grid(row=i*2, column=1, padx=(5, 0), sticky="w")
            Label(highlight_frame, text=f"80.{i}%", font=("Dubai Medium", 10), bg="#212E38", fg="white", width=10, anchor="w").grid(row=i*2, column=2, padx=(5, 0), sticky="w")

            # Separator line between each detail row
            Canvas(highlight_frame, height=1, width=180, bg="#3A3A3A", highlightthickness=0).grid(row=i*2+1, columnspan=3, padx=5, pady=2, sticky="ew")


        # Emotion buttons under the graph
        emotions = ["😠", "🤢", "😨", "😊", "😐", "😢", "😲"]
        emotion_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        # Dictionaries to store the buttons and labels with tags
        self.emotion_btns = {}
        self.bubbles = {}
        self.emotion_precentage = {}
        

        # Position each emotion button and label under the graph
        for index, (symbol, name) in enumerate(zip(emotions, emotion_names)):
            # Create the button
            button = Button(
                self,
                text=symbol,
                font=("Arial", 50),
                bg="#C0C5B1",
                fg="#28374A",
                activebackground="#C0C5B1",
                activeforeground="#28374A",
                bd=0,
                highlightthickness=0,
                command=lambda name=name: toggle_emotion(self, name, main_canvas)
            )
            button.place(x=graph_x + index * 85.7, y=graph_y_end + 20, width=85.7, height=85.7)
            self.emotion_btns[name] = button  # Save the button in the dictionary
            
            # Create the canvas for the red circle (bubble)
            canvas = Canvas(
                self,
                width=17,  # Circle size
                height=17,
                bg="#C0C5B1",  # Match background
                highlightthickness=0  # Remove canvas border
            )
            canvas.place(x=(graph_x + index * 85.7) + 60, y=(graph_y_end + 40))  # Adjust position

            # Draw the red circle
            canvas.create_oval(
                -1, -1, 17, 17,  # Circle dimensions
                fill="red",  # Bright red color
                outline="red"  # No outline for smoothness
            )

            # Add the number inside the circle
            canvas.create_text(
                8, 8,  # Center of the circle
                text= '0',  # Dynamic numbering
                fill="black",  # Text color
                font=("Dubai Medium", 7, "bold"),
            )
            self.bubbles[name] = canvas
            
            # Create the label
            label = Label(
                self,
                text=name,
                font=("Dubai Medium", 12),
                bg="#C0C5B1",
                fg="#28374A",
            )
            label.place(x=graph_x + index * 85.7, y=graph_y_end + 110, width=85.7, height=20)

            # Create the percentage text under the emotion name
            percentage_label = Label(
                self,
                text="0.00%",  # Initial percentage text
                font=("Dubai Medium", 12),
                bg="#C0C5B1",
                fg="#28374A",
            )
            percentage_label.place(x=graph_x + index * 85.7, y=graph_y_end + 140, width=85.7, height=15)
            self.emotion_precentage[name] = percentage_label  # Save the percentage label in the dictionary

        # Analyze Button rectangle
        create_rounded_rectangle(
            main_canvas,
            x1=graph_x_end + 2, y1=graph_y_end + 5, x2=1000, y2=495,  # Adjusted height to fit inside main canvas
            radius=7,
            color="#28374A"
        )

        # Load, resize, and set up Analyze icon
        original_analyze = Image.open(relative_to_assets("analyze.png"))
        resized_analyze = original_analyze.resize((180, 150), Image.LANCZOS)
        self.analyze_icon = ImageTk.PhotoImage(resized_analyze)

        # Create Zoom In button
        analyze_button = Button(
            self,
            image=self.analyze_icon,
            bg="#28374A",
            activebackground="#28374A",
            bd=0,
            highlightthickness=0,
            command=lambda: extract_emotion_predictions_from_data(
                self,
                main_canvas, 
                highlight_frame, 
                scroll_canvas, 
                play_button, 
                graph_time
                )
        )
        analyze_button.place(x=graph_x_end + 10, y=graph_y_end + 20)

        # Create the label under each button
        Label(
            self,
            text="Analyze",
            font=("Dubai Medium", 20),
            bg="#28374A",
            fg="#B5EBE9"
        ).place(x=graph_x_end + 10, y=graph_y_end + 170, width=180, height=20)          


        # Under Delete Button - Volume Meter
        volume_canvas = Canvas(self, width=190, height=25, bg="#C0C5B1", highlightthickness=0, bd=0)
        volume_canvas.place(x=5, y=280)

        # Place the "Volume" label above the volume meter
        volume_label = Label(
            self,
            text="Volume",
            font=("Dubai Medium", 14, "bold"),
            bg="#C0C5B1",
            fg="#28374A"
        )
        volume_label.place(x=68, y=265, height=20)  # Position above the volume meter


        # Add volume meter to the volume_canvas
        create_volume_meter(volume_canvas, x=20, y=5, width=160, initial_volume=100)

        # Under Volume Meter - Mel Scale Visualizer
        spectrum_canvas = Canvas(self, width=190, height=150, bg="#C0C5B1", highlightthickness=0, bd=0)
        spectrum_canvas.place(x=5, y=310)

        # Initialize Mel Scale with default 15 bars
        create_spectrum_visualizer(spectrum_canvas, x=10, y=10, width=180, height=130)


        # Place the "Spectrum" label above the audio visualizer
        Label(
            self,
            text="Spectrum Visualizer",
            font=("Dubai Medium", 10, "bold"),
            bg="#C0C5B1",
            fg="#28374A",
            anchor='center'
        ).place(x=40, y=465, height=20)  # Position above the under audio visualizer
