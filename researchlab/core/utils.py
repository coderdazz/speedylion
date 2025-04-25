from inspect import getargvalues, stack
import os
import ctypes
import pyarrow.feather as feather
import pickle
import sys
import pandas as pd


# Function to automatically update all input arguments into class

def arguments():
    """

    Allows for automatic update of input arguments into object instance



    Example:



    """

    args = getargvalues(stack()[1][0])[-1]

    del args['self']

    if 'kwargs' in args:
        args.update(args['kwargs'])

        del args['kwargs']

    return args


def check_make_folder(dir):
    """ Check for existence of a directory if not make one """

    # Check for existence of folder to save output, if not make one

    if not os.path.exists(dir):
        os.makedirs(dir)


# send alert

def send_alert(text='', stop=False):
    """ Sends windows alert when running a programme """

    ctypes.windll.user32.MessageBoxW(0, text, "ATTENTION", 1)

    if stop == True:
        print('Because of this error, the program will terminate now. Investigate, fix and re-run.')

        sys.exit("Error message")


def saveObj(obj, name: str):
    """ Object to pickle file"""

    dir = os.getcwd()

    with open(dir + '\\' + name + '.pkl', 'wb') as file:
        pickle.dump(obj, file)


# pick up pickle objects

def loadObj(name: str):
    dir = os.getcwd()

    with open(dir + '\\' + name + '.pkl', 'rb') as file:
        output = pickle.load(file)

    return output


def load_data(path):
    """Load index data from the given path."""

    return pd.read_excel(path, index_col=0, parse_dates=True)


class MissingDataError(Exception):
    pass