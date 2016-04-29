#!/usr/bin/env python
# http://grantmcwilliams.com/tech/programming/python/item/581-grep-a-file-in-python
 
import re

def grep(patt,file):
    """ finds patt in file - patt is a compiled regex
        returns all lines that match patt """
    matchlines = []
    filetxt = open(file)
    lines = filetxt.readlines()
    for line in lines:
        match = patt.search(line)
        if match:
            matchline = match.group()
            matchlines.append(matchline)
    results = '\n '.join(matchlines)
    if results:
        return results
    else:
        return None

# Example use 
textfile = "/etc/hosts"
file = open(textfile)
criteria = "localhost"

expr = re.compile(r'.*%s.*' % criteria) # finds line that starts with anything, ends with anything and has criteria in it
#expr = re.compile(r'[0-9].*filename:(%s)\schecksum:.*result: (.*)' % criteria) # more complex example

# using return code
if grep(expr, file):
    print  criteria + " is in " + textfile
else:
    print criteria + " is not in " + textfile

file.seek(0) # rewind file for next test

# printing all matching lines
results = grep(expr, file)
print results

file.close()
