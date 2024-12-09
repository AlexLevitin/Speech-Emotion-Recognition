import tkinter as tk
from ctypes import windll
from gui.frames.title_frame import TitleFrame
from gui.frames.analyze_frame import AnalyzeFrame

class CustomWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set up custom window without native decorations
        self.overrideredirect(True)

        # Initialize offset variables for window dragging
        self.offset_x = 0
        self.offset_y = 0

        # Center the window on the screen
        self.center_window(1012, 546)

        # Title frame (custom title bar)
        self.title_bar = TitleFrame(self, self.close_window, self.minimize_window , {'start_move': self.start_move, 'do_move': self.do_move})
        self.title_bar.pack(side="top", fill="x")

        # Analyze frame
        self.analyze_frame = AnalyzeFrame(self)
        self.analyze_frame.pack(side="top", expand=True, fill="both")

        # Set up the taskbar presence
        self.after(100, self.set_taskbar_window)

    def start_move(self, event):
        # Capture the mouse position relative to the root window
        self.offset_x = event.x_root - self.winfo_x()
        self.offset_y = event.y_root - self.winfo_y()

    def do_move(self, event):
        # Update window position based on the captured offset
        x = event.x_root - self.offset_x
        y = event.y_root - self.offset_y
        self.geometry(f'+{x}+{y}')

    def set_taskbar_window(self):
        """Set the application to appear in the taskbar."""
        GWL_EXSTYLE = -20
        WS_EX_APPWINDOW = 0x00040000
        WS_EX_TOOLWINDOW = 0x00000080

        hwnd = windll.user32.GetParent(self.winfo_id())
        if hwnd:
            # Modify the extended window style to include WS_EX_APPWINDOW
            style = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style = (style & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
            windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            # Force the style change to take effect by toggling visibility
            self.withdraw()
            self.after(10, self.deiconify)  # Delay slightly to ensure it refreshes

    def close_window(self):
        self.destroy()

    def minimize_window(self):
        """Minimizes the window to the taskbar."""
        hwnd = windll.user32.GetParent(self.winfo_id())
        if hwnd:
            windll.user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE flag is 6

    def center_window(self, width, height):
        """Centers the window on the screen."""
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

if __name__ == "__main__":
    app = CustomWindow()
    app.mainloop()
