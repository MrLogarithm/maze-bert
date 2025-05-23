---
layout: default
---

# Installation instructions for Maze

This covers the basics of installing on Windows, MacOS, and Linux.

Note: Maze needs specific versions of Python and some packages. If you use Python for other projects, we recommend using [conda](https://docs.conda.io/projects/conda/en/latest/) to manage environments so it's easy to switch between what Maze needs and what your other projects use. 

### Windows
1. Download the files by going to <https://github.com/vboyce/Maze>, clicking the green "Clone or Download" button, and selecting Download Zip. Once the zip file downloads, extract the maze_automate folder to the desired location. 
2. Install python3 and pip3 by going to <https://www.python.org/downloads/windows/> and selecting under Stable Releases > Python 3.7.7 > "Download Windows x86-64 executable installer" for 64-bit computers or "Download Windows x86 executable installer" for 32-bit computers. Run the installer and complete the installation. (Earlier versions of Python 3 (3.6 etc) may also work, Python 3.8 or newer will not.)  
IMPORTANT: make sure the box that says "Add Python 3.X to PATH" is checked, otherwise you may not be able to use the python/python3 command in the command prompt.  
To check if pip3 is installed on the computer, open command prompt and type either of the following:
 ```
 pip --version
 pip3 --version
 ```
 Make sure pip3 is updated to the latest version, which could be done using the following command:
 ```
 python3 -m pip install --upgrade pip
 ```
3. Install needed packages. 
```
pip3 install nltk
pip3 install wordfreq
pip3 install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```
The command for downloading torch depends on the versions you're using and going to install, look here for more details: <https://pytorch.org/get-started/locally/>.

Continue under All Operating Systems.
### Mac OS X
1. Download the files by going to <https://github.com/vboyce/Maze>, clicking the green "Clone or Download" button, and selecting Download Zip. Once the zip file downloads, extract the maze_automate folder to the desired location. 
2. Install python3 and pip3 by going to <https://www.python.org/downloads/mac-osx/> and selecting under Stable Releases > 3.7.7 > "Download macOS 64-bit installer" for 64-bit computers or "Download macOS 64-bit/32-bit installer" for 32-bit computers. Run the installer and complete the installation. (Earlier versions of Python 3 (3.6 etc) may also work, Python 3.8 or newer will not.)  
To check if pip3 is installed on the computer, open command prompt and type the following:
```
pip3 --version
```
Make sure pip3 is updated to the latest version, which could be done using the following command:
```
python3 -m pip install --upgrade pip
```
3. Install needed packages.
```
pip3 install nltk
pip3 install wordfreq
pip3 install torch==1.3.1
```
If any dialog box pops up to install gcc, follow their instructions.
Continue under All Operating Systems.

### Linux
1. Download the files by going to <https://github.com/vboyce/Maze>, clicking the green "Clone or Download" button, and selecting Download Zip. Once the zip file downloads, extract the maze_automate folder to the desired location. 
2. Install python3 and pip3 (copy/paste the shown commands into terminal/command line one by one, when prompted, type your password). 
```
sudo apt-get install python3.7
sudo apt update
sudo apt-get install python3-pip
```
(Note: Earlier versions of Python 3 (3.6 etc) may also work, Python 3.8 or newer will not.)
3. Install needed packages.
```
pip3 install nltk
pip3 install torch==1.3.1
pip3 install wordfreq
```
Continue under All Operating Systems.

### All Operating Systems

4. Make commands executable
Navigate into the maze_automate folder (command will differ depending on where you put the folder; use cd to move into a folder and ls to see the contents)
Example
```
cd (where the repository is stored)/Maze/maze_automate
```
Make files exectutable
```
chmod +x set_up.py
chmod +x distract.py
```
5. Download model files and complete installation
From the maze_automate directory, run the appropriate command. For Windows, ignore the ```./``` part of the commands and use ```python3 ...``` instead.
```
./set_up.py --gulordava
```
If there is an error indicating that it cannot import wget, use the command ```pip3 install wget```. (If that doesn't work, try also ```pip3 install python-wget```.)
6. Do a test run of distractor automation
test_input.txt contains a few sample sentences; replace output_location.txt with the name of the file to write test Maze materials to. 
To test Gulordava model
```
./distract.py test_input.txt output_location.txt
```
This may take a few minutes to run, but when it finishes you can check that the output file contains Maze materials for the input file.

## Troubleshooting notes:

If you get errors about pytorch version 1.3.1 not being available, check what version of Python you have; if it's 3.8 or newer, you need to downgrade to 3.7 to be able to install the right version of pytorch. I recommend using [conda](https://docs.conda.io/projects/conda/en/latest/) to create a separate environment for Maze. 

If you get errors about `no such file or directory`, check that you ran set_up.py. (Note: if set_up.py isn't working, you can also manually download the files mentioned in set_up.py and move them to the appropriate locations.)

If you get an error about `'LSTM' object has no attribute '_flat_weights_names'` it means you need to downgrade torch to an older version (1.2 and 1.3.1 should work, 1.4 does not). 

