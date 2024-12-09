from tkinter import Canvas, Frame, Button, PhotoImage, Label
from pathlib import Path
from PIL import Image, ImageTk

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"../assets/title_assets")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class TitleFrame(Frame):
    def __init__(self, parent, close_command, minimize_window, move_window):
        super().__init__(parent)
        self.configure(bg="#1B1C22", height=40, width=1012)

        # Load, resize, and add icon image
        original_icon = Image.open(relative_to_assets("icon.png"))
        resized_icon = original_icon.resize((25, 25), Image.LANCZOS)  # Resize to 24x24 pixels
        self.icon_image = ImageTk.PhotoImage(resized_icon)

        # Icon label
        self.icon_label = Label(self, image=self.icon_image, bg="#1B1C22")
        self.icon_label.pack(side="left", padx=5)

        # Title label
        self.title_label = Label(
            self,
            text="• SPEECH EMOTION RECOGNITION •", 
            bg="#1B1C22", 
            fg='white',
            font=("Dubai Medium",14)
            )
        self.title_label.pack(side="top", padx=10, anchor='center')
        self.title_label.place(relx=0.5, rely=0.6, anchor="center")

        # Close button
        self.close_button = Button(self, text='🗙', command=close_command, bg='#1B1C22', fg='white', bd=0, width=4)
        self.close_button.pack(side="right", padx=0)  # Removed space between buttons
        self.close_button.bind("<Enter>", lambda e: self.close_button.config(bg='#FF5C5C'))  # Hover color to red
        self.close_button.bind("<Leave>", lambda e: self.close_button.config(bg='#1B1C22'))

        # Minimize button
        self.minimize_button = Button(self, text='—', command=minimize_window, bg='#1B1C22', fg='white', bd=0, width=4)
        self.minimize_button.pack(side="right", padx=0)  # Removed space between buttons
        self.minimize_button.bind("<Enter>", lambda e: self.minimize_button.config(bg='#2C2D33'))  # Brighten color
        self.minimize_button.bind("<Leave>", lambda e: self.minimize_button.config(bg='#1B1C22'))


        # Bind window dragging events
        self.bind("<Button-1>", move_window['start_move'])  
        self.bind("<B1-Motion>", move_window['do_move'])    
        # Bind the same events to the title label
        self.title_label.bind("<Button-1>", move_window['start_move'])
        self.title_label.bind("<B1-Motion>", move_window['do_move'])
        # Bind the same events to the icon label
        self.icon_label.bind("<Button-1>", move_window['start_move'])
        self.icon_label.bind("<B1-Motion>", move_window['do_move'])
