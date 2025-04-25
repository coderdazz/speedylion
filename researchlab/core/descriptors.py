""" Module for data descriptors to describe data structures """
import copy

# List out unique items

def unique(itemslist: list):
    """ Get unique items in a list """
    output = []
    [output.append(_) for _ in itemslist if _ not in output]

    return output



# Hash

def hash(l):
    out = ''
    for s in l: out = out + str(s) + '|'

    return out[:-1]

# Unhash

def unhash(s):
    out = s.split('|')
    return out



class Recordset():

    def __init__(self,data):
        self.data = data

    def append(self,d):

        self.data.append(d)
        return self

    def select(self,tag,string):

        out = Recordset([])

        for d in self.data:
            if d[tag].lower() == string.lower():
                out.append(d)
        return out

    def filter(self,filters):

        out = copy.deepcopy(self)

        for k,v in filters.items():
           out = out.select(k,v)

        return out


    def sort(self,sortKeys):

        self.data.sort(key=lambda l: '!'.join([str(l[_]) for _ in sortKeys]))
        return self

    def accumulate(self,tags,valueTags):

        hashes = {}
        lowerhashes = {}
        if not valueTags.__class__.__name__ == 'list': valueTags = [valueTags]

        for d in self.data:
            h = hash([d[tag] for tag in tags])
            if not h.lower() in lowerhashes: lowerhashes[h.lower()] = {}
            for v in valueTags:
                if v in d:
                    if v in lowerhashes[h.lower()]:
                        lowerhashes[h.lower()][v] += d[v]
                    else:
                        lowerhashes[h.lower()][v] = d[v]

            hashes[h.lower()] = h  # last one decides capitalisation

        out = Recordset([])

        for key,value in lowerhashes.items():

            u = unhash(hashes[key])
            d = {tags[i]:u[i] for i in range(len(tags))}

            for k,v in value.items(): d[k] = v
            out.append(d)

        return out

    def unique(self,tag):

        out = {d[tag].lower():d[tag] for d in self.data}
        return sorted(out.values())



# Display object

def display(x):
    """ Displays objects in a formatted way """

    if x.__class__.__name__ == 'list':
        out = ''
        for _ in x:

            out += display(_)+'\n'
        return out

    elif x.__class__.__name__ == 'dict':

        out = ''
        for k,v in x.items():

            out += display(k) + display(v) + ' | '
        return out[:-3]

    elif x.__class__.__name__ == 'str':
        return '{:<16}'.format(x)

    return '{:>16}'.format('{:,.0f}'.format(float(x)))


# Function to describe dataframe

def describe_df(df, weights=None):

    from scipy import stats
    from numpy import average

    final_df = df.describe()

    for col in final_df.columns:

        final_df.loc['mode', col] = df[col].mode()[0]
        final_df.loc['median', col] = df[col].median()
        final_df.loc['harmonic_mean', col] = stats.hmean(df[col])

    if weights is not None:
        final_df.loc['wght_avg', col] = average(df[col], weights = df[weights])

    else:
        pass

    return final_df

