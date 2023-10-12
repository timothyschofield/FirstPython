"""

VS Code
This all relates to teething problems getting Python to work in VS Code

10 October 2023

prob1 : python not found in terminal i.e. python --version does not work
    Go to Command palette Python: Select Interpreter
python --version still not work

prob2: Create a virtual environment
    Manage > Command prompt... Python Create Enviroment
    This appears to work - but pip (which is in the .venv > Lib) is still unknown

prob3: Activate the virtual environment
    In the terminal, navigate to the folder (probably the project folder) 
    that contains the .venv folder
    In the terminal .venv\Scripts\activate

SUCCESS! pip is now recognized and so is python

And there is a green (.venv) to the left of the terminal prompt which means
    the virtual environment is working

It seems that after you have done this once - for this project at least - 
the enviroment is remembered and you don't have to activate the .venv again

And because all the Python examples use the same .venv
you don't have to install the libraries such as numpy over and over again.


"""

# pip install numpy
import numpy as np


msg = "Roll a dice!"
print(msg)

print(np.random.randint(1, 9))








