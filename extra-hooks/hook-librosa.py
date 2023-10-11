
""" 
In case there is a problem with importing librosa when converitng the py file to exe file use this two lines of code below using auto-py-to-exe, built in PyInstaller.
Tell PyInstaller wehre to locate the file by adding --additional-hooks-dir "PATH_TO_YOUR_PROJECT/extra-hooks" command on auto-py-to-exe settings -> Manual argument input.
"""
from PyInstaller.utils.hooks import collect_data_files
datas = collect_data_files('librosa')