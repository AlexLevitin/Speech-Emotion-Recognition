import tkinter as tk
from gui.main_window.main import MainWindow
from gui.analyze.results import AnalyzeWindow

# Main window constructor
root = tk.Tk()  # Make temporary window for app to start
root.withdraw()  # WithDraw the window


if __name__ == "__main__":

    #loginWindow()
    #MainWindow()
    AnalyzeWindow()
    root.mainloop()
