# Bull Prediction App

### This is an application to predict the 'bull' events from the given audio files.
The 'bull' event is a characteristic splash sound made by an migratory fish called alosa (Alosa fallax also known as twait shad) during spawning in the surface of the water with their fin. Even though they lives in seas for spawning they move to rivers and monitoring these species provides a good indicator for rivers health. Stakeholders involved in the rehabilitation of freshwater ecosystems rely on staff to aurally count the bulls during spring nights and then estimate the alosa population in different sites. In order to reduce the human costs and expand the scope of the analyze, we have built deep learning models to automatically detect the sound event from dozens of GB of audio files recorded from the riverbanks. Here is presented a graphical user interface to predict the start and the end time of a bull event.

<strong>Keywords</strong>: bioacoustics, deep learning, freshwater, audio event detection, graphical user interface.


## How to use it ?

1. Firstly choose the folder where you have the audio files in the '.wav' format by clicking the browse button. It will select the audio files automatically. 
2. Secondly click the process button to segment the selected audio files. Wait until it is completely finished and the finishing message is displayed. Close the process window by clicking the 'close' or the 'x' button to continue on the next step. 
3. Thirdly click the 'model' button to select the model file in the '.pth' format. Till now there are only two models that can be used with this app (vgg16 am21 and crnn from fa2023). If you choose a model that is not include in the app it will show you a message to choose the correct model. 
4. Finally by clicking the 'predict' button you will be asked to choose the directory/location where you want to save the prediction files. After you choose the location,  the prediction of all selected audio files will start automatically. Wait until the prediction is finished and you see the displayed message. Close the predict window and check for the prediction files. These files are saved at your computer inside a folder called 'predictions'. Each file contains the start and the end of the bull event.

## Use other models
To make the app work with other models, the architecture of the new model should be added on the my_models.py notebook. Also you should import the new model to Bull_predict.py and change the 'if condition' on the predict function.

## Recommendations

It is not recommended to select more than 10 audio files at once because the app will get very heavy and the processing time will last longer. The selection of audio files depends on the power of computer you are using and the length of the audio files too, but still do not exceed the number 10.


## Other Informations

Version: 1.0

Year : 2023
