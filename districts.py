# Districts.py
#
# 

from csv import DictReader
from collections import defaultdict
from math import log, exp
from math import pi as kPI

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
	try:
		if val != None:
			return float(val[:-1].replace(",", "."))
	except ValueError or TypeError:
		return 0

def parseint(val):
	try:
		if val:
			return int(val[:2])
	except ValueError or TypeError:
		return 0

def margin(lines,district):
	acc = 0
	accr = 0
	for ii in lines:
		if parseint(ii["D"]) == district and ii["GENERAL VOTES "]:
			if ii["PARTY"]:
				acc += parsefloat(ii["GENERAL VOTES "])
			if ii["PARTY"] == 'R':
				accr += parsefloat(ii["GENERAL VOTES "])
	if acc != 0:
		rshare = accr * 100 / acc
		return rshare
	else:
		return 0
def all_state_rows(lines, state):
	return [x for x in lines if x["STATE"] == state]
############################ MY HELPERS ###############################

def ml_mean(values):
    samplemean = sum(x for x in values) / len(values) 
    return samplemean 

def ml_variance(values, mean):
    svariance = sum((x - mean)**2 for x in values) / len(values) 
    return svariance 

def log_probability(value, mean, variance):
	if variance == 0:
		return 0
	else:
		const = 1 / (2*kPI*variance)**0.5
		ex = exp(-(value - mean)**2 / (2*variance**2))
		return log(const * ex) 

def republican_share(lines, states):
	adict = {}
	for state in set(x["STATE"] for x in lines):
		if state in states:
			srows = all_state_rows(lines,state)
			for x in srows:
				if x["D"] and x["D"] != 'H' and x["D"][5:] != "UNEXPIRED TERM":
					adict[(state, parseint(x["D"]))] = margin(srows,parseint(x["D"]))
	return adict

if __name__ == "__main__":
    # Don't modify this code
    lines = [x for x in DictReader(open("../data/2014_election_results.csv"))
             if valid(x)]

    obama_mean = ml_mean(republican_share(lines, kOBAMA).values())
    romney_mean = ml_mean(republican_share(lines, kROMNEY).values())

    obama_var = ml_variance(republican_share(lines, kOBAMA).values(),
                             obama_mean)
    romney_var = ml_variance(republican_share(lines, kROMNEY).values(),
                              romney_mean)

    colorado = republican_share(lines, ["Colorado"])
    print("\t\tObama\t\tRomney\n" + "=" * 80)
    for co, dist in colorado:
        obama_prob = log_probability(colorado[(co, dist)], obama_mean, obama_var)
        romney_prob = log_probability(colorado[(co, dist)], romney_mean, romney_var)

        print("District %i\t%f\t%f" % (dist, obama_prob, romney_prob))
