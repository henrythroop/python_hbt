#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 19:18:22 2020

@author: hthroop
"""

# For decoding Brainchase Shakespears 11-Jul-2020

#!/usr/bin/env python
from math import fmod

import string

# Caesar cipher. This is just a fixed integer (e.g., ROT13)

def caesar(plaintext, shift):
    alphabet = string.ascii_lowercase
    shifted_alphabet = alphabet[shift:] + alphabet[:shift]
    table = str.maketrans(alphabet, shifted_alphabet)
    return plaintext.translate(table)

# VIGENERE CIPHER. This is actually what Brainchase is using.
# From https://github.com/deed02392/vigenere/blob/master/vigenere.py

# The minimum and maximum valid char, ascii table defined order
ascii_min = ord('A')
ascii_max = ord('Z')

def vigenere(phrase, key, direction = 'D'):
    # Generate a string of all the possible chars
    alpha = ""
    for printable in range(ascii_min, ascii_max+1):
        alpha = alpha + chr(printable)

    # Ensure the key is at least as long as the ciphertext by cat'ing it
    while len(key) < len(phrase):
       key = key + key
    key = key[0:len(phrase)]

    out = ""
    for i in range(len(phrase)):
        index_from_phrase = (ord(phrase[i]) - ascii_min)
        index_from_key = ord(key[i]) - ascii_min
        difference = (index_from_phrase - index_from_key)

        # We want the sign of the dividend so we use fmod()
        # Use the inverse of this result (I'm not certain why - is there a simpler way?
        
        if (direction == 'D'): # decrypt
            intersect = int(fmod(index_from_phrase + index_from_key, (ascii_max - ascii_min + 1)) * +1)

        if (direction == 'E'): # encrypt
            intersect = int(fmod(index_from_phrase - index_from_key, (ascii_max - ascii_min + 1)) *  1)
            
        letter = alpha[intersect]
        out += letter

    return out, key

if __name__ == "__main__":
    # phrase = 'HELLOIAMHERE'
    # phrases = ['ALKEGJW', 'AOHUNUSK', 'GROXIVE', 'JMRHMX', 'KWWCZAKOAFW', 'LWTCIGPKTAUGDB','AAAAAA']
    
    phrases = ['LMWPEV', 'KIWXG', 'MXC']

    # keys = ['SHAKESPEARE', 'MISSING', 'BARD', 'TOBEORNOTTOBE', 'TAMINGOFTHESHREW', 'ROMEO', 'JULIET', 'AAAA']
    
    keys = ['SHAKESPEARE']
    # keys = ['OKTEA', 'SUNSTONE', 'THATSMISSING', 'WESTOLE']
    # keys = ['PSYLLA']
    # keys = ['KEY', 'VERONA', 'TATEGREYSON']
    # keys = ['TURKEY']
    # keys = ['ISTANBUL']
    # keys = ['COMEDY', 'TRAGEDY']
    # keys = ['WILLIAM', 'HAMLET']
    # keys = ['CORTEZ', 'ENGLAND']
    # keys = ['GOLD']
    # keys = ['THOMAS','CARLYLE']
    # keys = ['IAMBIC']
    # keys = ['REFLECTION']
    keys = ['TREASURE', 'BOOK']
    
    
    for phrase in phrases:
        for key in keys:        
            if len(key.strip()) > 0:
                encoded, with_key = vigenere(phrase, key.strip(), 'E')
                decoded, with_key = vigenere(phrase, key.strip(), 'D')
                encoded_r, with_key = vigenere(phrase, key.strip()[::-1], 'E')
                decoded_r, with_key = vigenere(phrase, key.strip()[::-1], 'D')
                print(f'Key   = {key}')
                print(f'In    = {phrase}')
                print()
                print(f' ENC      = {encoded}')
                print(f' DEC      = {decoded}')
                print(f' ENC r    = {encoded_r}')
                print(f' DEC r    = {decoded_r}')       
                print("---")