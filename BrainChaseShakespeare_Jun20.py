#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 18:58:54 2020

@author: hthroop
"""

# 

import string

def caesar(plaintext, shift):
    alphabet = string.ascii_lowercase
    shifted_alphabet = alphabet[shift:] + alphabet[:shift]
    table = str.maketrans(alphabet, shifted_alphabet)
    return plaintext.translate(table)

# https://gist.github.com/dssstr/aedbb5e9f2185f366c6d6b50fad3e4a4
    
def vig(txt='', key='', typ='d'):    
    if not txt:
        print('Needs text')
        return
    if not key:
        print('Needs key')
        return
    if typ not in ('d', 'e'):
        print('Type must be "d" or "e"')
        return

    k_len = len(key)
    k_ints = [ord(i) for i in key]
    txt_ints = [ord(i) for i in txt]
    ret_txt = ''
    for i in range(len(txt_ints)):
        adder = k_ints[i % k_len]
        if typ == 'd':
            adder *= -1

        v = (txt_ints[i] - 65 + adder)

        ret_txt += chr(v + 65)

    print(f'Return = {ret_txt}')
    return ret_txt

plaintext = 'ALKEGJW'
shift = 2

print(caesar(plaintext, shift))

q = vig('LETS CHECK THIS OUT', 'AAAAAAAAAAA', 'e')
print(q)

vig(q, 'AAAABCDEFGH', 'd')
