from tkinter import *
import tkinter as tk
from tkinter import ttk
import os
import sys


class View(tk.Tk):


    def __init__(self, controller):
        super().__init__()
        self.title('Bull Predict (v1.0)')
        # self.wm_iconbitmap(bitmap = "fish_icon.ico")
        self.path = self.resource_path("fish_app.ico")
        self.wm_iconbitmap(bitmap = self.path)
        self.grid()
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        # self.width = int(self.screen_width * 0.31)
        # self.height = int(self.screen_height * 0.35)
        # self.geometry(f"{self.width}x{self.height}")
        self.controller = controller
        self.file_entry = tk.StringVar()
        self.model_entry = tk.StringVar()
        self._first_frame()
        self._second_frame()
        self._make_labels()
        self._make_entry()
        self._make_buttons()
        # self.progress_window = None
        # self.progress_bar = None
        # self.process_label = None
        self.progress_window_pred = None


    # This function helps to add the icon in the app with --onefile 
    def resource_path(self, relative_path):
        try:
            self.base_path = sys._MEIPASS
        except Exception:
            self.base_path = os.path.abspath(".")
        return os.path.join(self.base_path, relative_path)
    

    def main(self):
        self.mainloop()


    def _first_frame(self):
        self.main_frm = Frame(self, highlightbackground="#B2BEB5", highlightthickness=1, padx=5, pady=5)
        self.main_frm.pack(padx=12, pady=12)


    def _second_frame(self):
        self.second_frm = Frame(self, highlightbackground="#B2BEB5", highlightthickness=1, padx=5, pady=5)
        self.second_frm.pack(padx=12, pady=12)


    def _make_entry(self):
        file_entry = ttk.Entry(self.main_frm, width=60, textvariable=self.file_entry)
        file_entry.grid(row=1, column=0, sticky=W, pady=10)

        model_entry = ttk.Entry(self.second_frm, width=60, textvariable=self.model_entry)
        model_entry.grid(row=4, column=0, sticky=W, pady=10)


    def _make_buttons(self):
        browse_file = ttk.Button(self.main_frm, text="Browse", command=self.controller.browse_file_clicked)
        browse_file.grid(row=1, column=1, sticky=E, pady=10)

        process_button = ttk.Button(self.main_frm, text='Process', command=self.controller.process_clicked)
        process_button.grid(row=2, column=1, sticky=E, pady=10)

        browse_model = ttk.Button(self.second_frm, text="Model", command=self.controller.browse_model_clicked)
        browse_model.grid(row=4, column=1, sticky=E, pady=10)

        predict_button = ttk.Button(self.second_frm, text='Predict', command=self.controller.save_predict_clicked)
        predict_button.grid(row=5, column=1, sticky=E, pady=10)


    def _make_labels(self):
        file_label = ttk.Label(self.main_frm, text="Insert the audio files (.wav) you want to predict: ")
        file_label.grid(row=0, column=0, sticky=W)

        process_label = ttk.Label(self.main_frm, text="Process the audio files: ")
        process_label.grid(row=2, column=0, sticky=W, pady=10)

        model_label = ttk.Label(self.second_frm, text="Insert the model file (.pth) you want to use for prediction: ")
        model_label.grid(row=3, column=0, sticky=W)

        predict_label = ttk.Label(self.second_frm, text="Make the prediction: ")
        predict_label.grid(row=5, column=0, sticky=W)


    # Popup window for process function
    def show_progress_window(self):
        self.progress_window = Toplevel()
        self.progress_window.wm_iconbitmap(bitmap = self.path)
        self.progress_window.geometry(f"{int(self.screen_width * 0.25)}x{int(self.screen_height * 0.15)}")
        self.progress_window.title("Processing...")


        self.progress_bar = ttk.Progressbar(self.progress_window, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(pady=25)
        self.progress_bar["value"] = 0
        self.progress_bar.update()

        self.process_label = Label(self.progress_window, text="Files are being segmented, please wait.")
        self.process_label.pack(pady=10)

        self.progress_window.protocol("WM_DELETE_WINDOW", self.controller.processing_window_closed)



    def update_progress(self, i, audio_segments, j, files_path):
            self.progress_bar["value"] = int((j+1) * 100 / len(audio_segments))
            self.progress_bar.update()
            self.progress_window.title(f"Processing... {self.progress_bar['value']}% of {i+1}/{len(files_path)}")
            


    def show_processing_finished_message(self, total_files, total_segments):
        self.progress_bar.pack_forget()
        self.process_label.config(
            text=f"Preprocessing finished!\nTotal number of audio files processed is {total_files}.\n"
                f"Total number of segmented audio files is {total_segments}."
        )
        close_button = Button(self.progress_window, text="Close", command=self.progress_window.destroy)
        close_button.pack(pady=5)
        self.progress_window.wait_window()
        

    
    #Popup window for predict function
    def show_progress_window_predict(self):
        self.progress_window_pred = Toplevel()
        self.progress_window_pred.wm_iconbitmap(bitmap = self.path)
        self.progress_window_pred.title("Predict")
        self.progress_window_pred.geometry(f"{int(self.screen_width * 0.25)}x{int(self.screen_height * 0.13)}")

        self.progress_label = ttk.Label(self.progress_window_pred, text="Predicting all selected files: ")
        self.progress_label.pack(pady=10)

        self.progress_bar = ttk.Progressbar(self.progress_window_pred, length=200, mode='determinate')
        self.progress_bar.pack(pady=10)

        self.progress_window_pred.protocol("WM_DELETE_WINDOW", self.controller.predict_window_closed)


    def update_progress_predict(self, batch_idx, num_batches):
        self.progress_percentage = (((batch_idx + 1) / num_batches) * 100)
        self.progress_bar['value'] = self.progress_percentage
        self.progress_label.config(text=f"Predicting all selected files: {int(self.progress_percentage)}%")
        self.progress_window_pred.update()


    def show_predicting_finished_message(self):
        self.progress_bar.pack_forget()
        self.progress_label.config(
            text="Predicting finished."
        )
        self.close_button = Button(self.progress_window_pred, text="Close", command=self.progress_window_pred.destroy)
        self.close_button.pack(pady=5)
        self.progress_window_pred.wait_window()


    
    #Save predict popup
    def save_predict_popup(self):
        self.save_window = Toplevel()
        self.save_window.wm_iconbitmap(bitmap = self.path)
        self.save_window.title("Save Location")
        self.save_window.geometry(f"{int(self.screen_width * 0.20)}x{int(self.screen_height * 0.10)}")
        self.save_button = ttk.Button(self.save_window, text="Select Save Location", command=self.controller.predict)
        self.save_button.pack(pady=10)

        self.save_window.protocol("WM_DELETE_WINDOW", self.controller.predict_window_closed)
        self.save_window.mainloop()


    def close_save_predict(self):
        self.save_window.destroy()


