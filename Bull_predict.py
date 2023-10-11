from model import Model
from view import View
from tkinter.messagebox import showinfo
from my_models import  model_vgg16_gn, model_crnn
from tkinter import filedialog
import threading
import torch


class Controller:


    def __init__(self):
        self.model = Model()
        self.view = View(self)
        self.files_path = ''
        self.model_path = ''
        self.processed_file_path = ''
        self.files_process = False
        self.processing_in_progress = False
        self.predict_in_progress = False


    def main(self):
        self.view.main()


    def browse_file_clicked(self):
        self.files_path = filedialog.askopenfilenames(title = 'Select (.wav) audio files',filetypes=[("Audio file", "*.wav")])
        # self.files_path = self.model.browse_file(folder_path)
        self.view.file_entry.set(self.files_path)


    def browse_model_clicked(self):
        self.model_path = filedialog.askopenfilename(filetypes= (('.pth files', '*.pth'), ('all files', '*.*')), 
                                               title='Select the model file (.pth)')
        self.view.model_entry.set(self.model_path)


    # There are some if conditions to debug the app 
    def process_clicked(self):
        if self.processing_in_progress:
            # showinfo(message='Processing in progress. Please wait.')
            return
        file_entry = self.view.file_entry.get() 
        if self.files_path and file_entry and file_entry.strip():
            if file_entry == self.processed_file_path:
                showinfo(message='Files have already been processed.')
            else:
                self.view.show_progress_window()
                processing_thread = threading.Thread(
                    target=self.model.process,
                    args=(self.files_path , self.update_progress_callback, self.processing_finished_callback)
                )
                processing_thread.start()
                self.processing_in_progress = True
                self.processed_file_path = ''
        else:
            showinfo(title='No audio files selected', message='Please select one or more audio files (.wav) to process')
        
   
    def processing_window_closed(self):
        self.processing_in_progress = False
        self.view.progress_window.destroy()


    def update_progress_callback(self, i, audio_segments, j, files_path):
        self.view.update_progress(i, audio_segments, j, files_path)


    def processing_finished_callback(self, total_files, total_segments):
        self.view.show_processing_finished_message(total_files, total_segments)
        self.processing_in_progress = False
        self.files_process = True
        self.processed_file_path = self.view.file_entry.get()


    def predict(self):
        folder_selected_save = filedialog.askdirectory()
        if folder_selected_save:
            try:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model = torch.load(self.model_path, map_location=device)
                if model.__class__.__name__ == 'model_vgg16_gn' or model.__class__.__name__ == 'model_crnn':
                    predicting_thread = threading.Thread(
                        target=self.model.predict,
                        args=(self.model_path, folder_selected_save, self.save_predict_finished, self.update_progress_callback_predict, 
                                    self.predicting_finished_callback, self.show_progress_window_predict_callback)
                    )
                    predicting_thread.start()
            except AttributeError:
                showinfo(title = 'Not a correct model', message= "Please choose one of the correct model files")
                self.view.save_window.destroy()
                self.predict_in_progress = False
    

    def show_progress_window_predict_callback(self):
        self.view.show_progress_window_predict()


    def update_progress_callback_predict(self, batch_idx, num_batches):
        self.view.update_progress_predict(batch_idx, num_batches)


    def predicting_finished_callback(self):
        self.view.show_predicting_finished_message()
        self.predict_in_progress = False


    def save_predict_finished(self):
        self.view.close_save_predict()

    
    # There are some if conditions to debug the app 
    def save_predict_clicked(self):
        if self.predict_in_progress:
            return
        model_file = self.view.model_entry.get() 
        if self.model_path and model_file: 
            if self.files_process:
                self.predict_in_progress = True
                self.view.save_predict_popup()
            else:
                showinfo(title='No files processed to predict', message='Please process the files before predicting.')
        else:
            showinfo(title='No model file selected', message='Please select an model file (.pth) to predict')
        
        
    def predict_window_closed(self):
        self.predict_in_progress = False
        self.view.save_window.destroy()
        if self.view.progress_window_pred:
            self.view.progress_window_pred.destroy()    


if __name__ == '__main__':
    bull_prediction = Controller()
    bull_prediction.main()