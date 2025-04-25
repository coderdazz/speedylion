"""
datefuncs module contains functions to handle date and time operations.
Includes functions to get quarter ends, to convert date frequency to integer multiplier.
"""

from datetime import datetime, timedelta
import math
import calendar

class Timer(object):
    """
    Timer object to time functions

    Examples:
        with Timer() as timer:
            # code to be timed
        print(timer.output())
    """
    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, *args):
        self.end = datetime.now()
        diff = self.end - self.start
        self.microseconds = diff.seconds * 1000000 + diff.microseconds

    def output(self):
        return str(self.microseconds / 1e6) + ' seconds'

def findQend():
    """ Find the previous quarter end date """
    date = datetime.now()
    qnow = math.floor((date.month - 1) / 3)
    Qend = datetime(date.year, 3 * qnow + 1, 1) + timedelta(days=-1)
    return Qend

def findQint():
    """ Find the previous quarter in integer """
    month = datetime.now().month
    Qlist = [0, 4, 4, 4, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    Q = Qlist[month]
    return Q

def findQno():
    """ Get string value for previous quarter e.g. Q4 """
    date = findQend()
    Q = findQint()
    Y = date.year
    return 'Q' + str(Q), str(Y)

# Find current quarter
def findCQ():
    """ Get integer value for current quarter e.g. 4 """
    month = datetime.now().month
    Qlist = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    qnow = Qlist[month]
    return qnow

def last_month_date():
    """ Get last month and this month date """
    today = datetime.today()
    this_month = today.strftime("%B %Y")
    first = today.replace(day=1)
    last_month = first - datetime.timedelta(days=1)
    last_month = last_month.strftime("%B %Y")
    return [last_month, this_month]

# Convert date to string
def convertDate(x):
    """ Convert date to string """
    if x.__class__.__name__ == 'list':
        return [convertDate(_) for _ in x]
    if x.__class__.__name__ == 'Timestamp':
        return datetime.strftime(x,'%Y-%m-%d')
    return x

def freqInt(freq):
    """ Get integer adjustment value for date frequency """
    freq_list = ['M', 'MS', 'BM', 'D', 'B', 'Q', 'QS', 'BQS', 'S', 'Y', 'W', 'W-SAT']
    if freq.upper().replace('E') not in freq_list:
        return "Error: Frequency of time series not recognized."
    freq_int_list = [12] * 3 + [250] * 2 + [4] * 3 + [2, 1, 52, 52]
    return freq_int_list[freq_list.index(freq)]

def compute_multiplier(date):
    """ Get number of days in quarter """
    date_object = datetime.strptime(date, '%m/%d/%y')
    current_quarter = math.ceil(date_object.month / 3)
    first_date = datetime(date_object.year, 3 * current_quarter - 2, 1)
    last_date = datetime(date_object.year, 3 * current_quarter + 1, 1) + timedelta(days=-1)
    number_days_quarter = (last_date - first_date).days
    tot_number_days_in_year = 366 if calendar.isleap(date_object.year) else 365
    multiplier = number_days_quarter / tot_number_days_in_year
    return multiplier

def readDate(s):
    """ Read Date """
    if s.__class__.__name__ == 'list':
        return [readDate(_) for _ in s]
    try:
        return datetime.strptime(s,'%Y-%m-%d')
    except:
        return datetime.strptime(s,'%Y%m%d')

def writeDate(d):
    return datetime.strftime(d,'%Y-%m-%d')

