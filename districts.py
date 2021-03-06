# Districts.py
#
# 

from csv import DictReader
from collections import defaultdict
from math import log, exp, sqrt
from math import pi as kPI
import numpy as np
import matplotlib.pyplot as plt

kOBAMA = set(["D.C.", "Hawaii", "Vermont", "New York", "Rhode Island",
              "Maryland", "California", "Massachusetts", "Delaware", "New Jersey",
              "Connecticut", "Illinois", "Maine", "Washington", "Oregon",
              "New Mexico", "Michigan", "Minnesota", "Nevada", "Wisconsin",
              "Iowa", "New Hampshire", "Pennsylvania", "Virginia",
              "Ohio", "Florida"])
kROMNEY = set(["North Carolina", "Georgia", "Arizona", "Missouri", "Indiana",
               "South Carolina", "Alaska", "Mississippi", "Montana", "Texas",
               "Louisiana", "South Dakota", "North Dakota", "Tennessee",
               "Kansas", "Nebraska", "Kentucky", "Alabama", "Arkansas",
               "West Virginia", "Idaho", "Oklahoma", "Wyoming", "Utah"])

def valid(row):
    return sum(ord(y) for y in row['FEC ID#'][2:4])!=173 or int(row['1']) < 3583
############################ MY HELPERS ###############################
def parsefloat(val):
    return float(val.replace(",", ".").replace("%",""))
def parseint(val):
    return int(val[:2])
def repub_share(lines,district):
    accr = 0
    for ii in lines:
        if parseint(ii["D"]) == district and ii["GENERAL %"]:
            if ii["PARTY"] == 'R':
                accr += parsefloat(ii["GENERAL %"])
    return accr
def all_state_rows(lines, state):
    return [x for x in lines if x["STATE"] == state]
############################ MY HELPERS ###############################

def ml_mean(values):
    """
    Given a list of values assumed to come from a normal distribution,
    return the maximum likelihood estimate of mean of that distribution.
    There are many libraries that do this, but do not use any functions
    outside core Python (sum and len are fine).
    """
    samplemean = sum(values) / len(values) 
    return samplemean 

def ml_variance(values, mean):
    """
    Given a list of values assumed to come from a normal distribution and
    their maximum likelihood estimate of the mean, compute the maximum
    likelihood estimate of the distribution's variance of those values.
    There are many libraries that do something like this, but they
    likely don't do exactly what you want, so you should not use them
    directly.  (And to be clear, you're not allowed to use them.)
    """

    svariance = sum((x - mean)**2 for x in values) / len(values) 
    return svariance 

def log_probability(value, mean, variance):
    """
    Given a Gaussian distribution with a given mean and variance, compute
    the log probability of a value from that distribution.
    """     

    if variance == 0:
        return 0
    else:
        const = 1 / (2*kPI*variance)**0.5
        ex = exp(-(value - mean)**2 / (2*variance))
        return log(const * ex) 

def republican_share(lines, states):
    """
    Return an iterator over the Republican share of the vote in all
    districts in the states provided.
    """
    #This code works provided there is MORE than 1 republican candidate per district
    adict = defaultdict(float)

    for state in states:
        for ll in lines:
            try:
                if ll["PARTY"] == 'R' and ll["GENERAL VOTES "] and ll["D"] and ll["D"][5:] != "UNEXPIRED TERM" and ll["STATE"] == state:
                    percent = parsefloat(ll["GENERAL %"])
                    district = parseint(ll["D"])
                    adict[(state, district)] = percent 
            except ValueError:
                continue
    return adict 

if __name__ == "__main__":
    # Don't modify this code
    lines = [x for x in DictReader(open("data/2014_election_results.csv"))
             if valid(x)]

    obama_mean = ml_mean(republican_share(lines, kOBAMA).values())
    romney_mean = ml_mean(republican_share(lines, kROMNEY).values())

    obama_var = ml_variance(republican_share(lines, kOBAMA).values(),
                             obama_mean)
    romney_var = ml_variance(republican_share(lines, kROMNEY).values(),
                              romney_mean)

    colorado = republican_share(lines, ["Colorado"])
    print("\t\tObama\t\tRomney\n" + "=" * 80)
    obama_ch = 0
    romney_ch = 0
    for co, dist in colorado:
        obama_prob = log_probability(colorado[(co, dist)], obama_mean, obama_var)
        romney_prob = log_probability(colorado[(co, dist)], romney_mean, romney_var)

        #accumulators for chance at winning (write-up)
        obama_ch += obama_prob
        romney_ch += romney_prob

        print("District %i\t%f\t%f" % (dist, obama_prob, romney_prob))

    ###################### CODE FOR WRITE UP ###########################
    """
    if obama_ch > romney_ch:
        print("Obama won")
    if romney_ch > obama_ch:
        print("Romney won")
    """

    rshare = []
    for state in set(x["STATE"] for x in lines):
        something = republican_share(lines, [state])
        for ii in something.values():
            rshare.append(ii)
    """
    for state in kROMNEY:
        something = republican_share(lines, [state])
        for ii in something.values():
            rshare.append(ii)
    """

    #CODE FOR HISTOGRAM
    binlist = []
    for i in range(0,22):
        binlist.append(i*5)

    #plt.hist(rshare, [0,10,20,30,40,50,60,70,80,90,100])
    plt.hist(rshare, binlist)
    plt.title("Histogram of republican_share in state districts", fontsize=18)
    plt.xlabel('Republican share (%)', fontsize=14)
    plt.ylabel('Frequency', fontsize = 14)
    plt.ylim([0,80])
    #plt.savefig('histogram.png')
    plt.show()
