import torch
import torchvision.transforms as transforms
import librosa
import numpy as np
import os


class Model:
    
    ## To select the whole folder
    # def browse_file(self, folder_path):
    #     files_path = []
    #     files_name = []

    #     if folder_path:
    #         for path, subdirs, files in os.walk(folder_path):
    #             files.sort()
    #             for name in files :
    #                 if name.endswith(".wav"):
    #                     files_path.append(path + os.path.sep + name)
    #                     files_name.append(name)
    #         # print("-", len(files_path), "files found in the directory", folder_path,'\n')
    #     return files_path
    
    
    
    def process(self, files_path, progress, progress_finished):
        melspectrograms = []
        global mel_specs_dict
        mel_specs_dict = {}

        for i, audio in enumerate(files_path):
            y, sr = librosa.load(audio, sr=44100)
            segment_length = 15 
            overlap = 5       
            # Calculate segment start and end times
            endStart = len(y) - segment_length * sr + 0.1 
            starts_time = np.arange(0, endStart, overlap * sr) / sr

            segment_samples = int(segment_length * sr)
            starts_sample = [int(start * sr) for start in starts_time]
            ends_sample = [start + segment_samples for start in starts_sample]

            audio_segments = [y[start:end] for start, end in zip(starts_sample, ends_sample)] # Segment audio data

            # Turn audio to mel-spectrogram
            for j, segment in enumerate(audio_segments):
                mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
                tensor = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 646)),
                transforms.ToTensor()
                ])(mel_spec)
                melspectrograms.append(tensor)

                audio_segment_name = f"{os.path.splitext(os.path.basename(audio))[0]}_{j*overlap:04d}" # Add the mel spectrogram tensor to the dictionary with a unique key
                mel_specs_dict[audio_segment_name] = tensor


                # Call the progress callback function to update the progress
                progress(i, audio_segments, j, files_path)
                    
            
        progress_finished(len(files_path), len(mel_specs_dict))


        return mel_specs_dict

    

    def predict(self, model_path, folder_selected, close_save_predict, predict, predict_finished, show_progress):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = torch.load(model_path, map_location=device)
        model.eval()
        results = []

        total_files = len(mel_specs_dict)
        batch_size = 8
        num_batches = int(np.ceil(total_files / batch_size))

        if folder_selected:
            close_save_predict()
            show_progress()

            with torch.no_grad():
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, total_files)

                    batch_results = []

                    mel_spec_tensors = []
                    for name in list(mel_specs_dict.keys())[start_idx:end_idx]:
                        mel_spec_tensor = mel_specs_dict[name].to(device)
                        mel_spec_tensors.append(mel_spec_tensor)

                    mel_spec_tensor_batch = torch.stack(mel_spec_tensors)
                    y_pred_batch = model(mel_spec_tensor_batch)  # forward computes the logits

                    sf_y_pred_batch = torch.nn.Softmax(dim=1)(y_pred_batch)  # softmax to obtain the probability distribution
                    _, predictions_batch = torch.max(sf_y_pred_batch, 1)  # decision rule, we select the max

                    for name, prediction in zip(list(mel_specs_dict.keys())[start_idx:end_idx], predictions_batch.tolist()):
                        # print(str(name) + "\t" + str(prediction))
                        batch_results.append([str(name), str(prediction)])

                    results.extend(batch_results)

                    predict(batch_idx, num_batches)


            model_name = os.path.basename(model_path)
            folder_name = os.path.join(folder_selected, "predictions", model_name)
            os.makedirs(folder_name, exist_ok=True)  # Create the folder if it doesn't exist

            # print("Predictions computed")
            positive_pred = {}
            negative_pred = {}

            for name, prediction in results:
                file_name, time_str = name.rsplit('_', 1)
                time_str = os.path.splitext(time_str)[0]
                prediction = bool(int(prediction))

                if file_name not in positive_pred:
                    positive_pred[file_name] = []
                if file_name not in negative_pred:
                    negative_pred[file_name] = []

                if prediction:
                    if file_name not in positive_pred:
                        positive_pred[file_name] = [int(time_str)]
                    else:
                        positive_pred[file_name].append(int(time_str))
                else:
                    if file_name not in negative_pred:
                        negative_pred[file_name] = [int(time_str)]
                    else:
                        negative_pred[file_name].append(int(time_str))

            # print(positive_pred)
            # print("number of files: ", len(positive_pred))
            # print("total number of positive segments: ", sum([len(x) for x in positive_pred.values()]))

            duration_length = 15

            for file in positive_pred:
                filename = file + "-predictions.txt"
                file_path = os.path.join(folder_name, filename)
                times = positive_pred[file]
                # print(file)
                if times:
                    starts = [times[0]]
                    ends = []
                    current_time = starts[-1]

                    for idx, time in enumerate(times):
                        current_end = current_time + duration_length
                        if time > current_end:
                            ends.append(current_end)
                            if idx < len(times):
                                starts.append(time)
                                current_time = time
                        current_time = time
                    current_end = current_time + duration_length
                    ends.append(current_end)

                    with open(file_path, 'w') as f:
                        for start, end in zip(starts, ends):
                            # print(start, "\t", end)
                            f.write(str(start) + ".00" + "\t" + str(end) + ".00" + "\n")
                else:
                    with open(file_path, 'w') as f:
                        f.write('')
            predict_finished()

        return results

