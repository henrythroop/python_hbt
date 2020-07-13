"""
- if lots of CPI violations but not small font, check if PDF reading in text correctly (e.g., spaces between each letter)
- Check histogram plots for lots of small font (usually figiure captions or figure embedded text)
"""

# ============== Import Packages ================

import sys, os, glob, pdb
import numpy as np
import pandas as pd
import fitz 
from collections import Counter
import textwrap
from termcolor import colored
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ============== Define Functions ===============

def get_text(d, pn):

    """
    PURPOSE:   extract text from a PDF document
    INPUTS:    d = PDF document file from fitz
               pn = page number to read (int)
    OUTPUTS:   t  = text of page (str)

    """
                
    ### LOAD PAGE
    p = d.loadPage(int(pn))

    ### GET RAW TEXT
    t = p.getText("text")
    
    return t


def get_pages(d, pl):

    """
    PURPOSE:   find start and end pages of proposal text
               [assumes TOC after budget & authors used full page limit]
    INPUTS:    d  = PDF document file from fitz
               pl = page limit of call
    OUTPUTS:   pn = number of pages
               ps = start page number
               pe = end page number

    """

    ### GET NUMBER OF PAGES
    pn = d.pageCount

    ### LOOP THROUGH PDF PAGES
    ps = 0
    for i, val in enumerate(np.arange(pn)):
            
        ### READ IN TEXT FROM THIS PAGE AND NEXT PAGE
        t1 = get_text(d, val)
        t2 = get_text(d, val + 1)
        
        ### FIND PROPOSAL START USING END OF SECTION X
        if ('SECTION X - Budget' in t1) & ('SECTION X - Budget' not in t2):
            
            ### PROPOSAL USUALLY STARTS 2 PAGES AFTER (I.E., INCLDUES TOC)
            ps += val + 2
            
            ### ASSUMES AUTHORS USED FULL PAGE LIMIT
            pe  = ps + (pl - 1)
            
        ### EXIT LOOP IF START PAGE FOUND
        if ps != 0:
            break 

    ### PRINT TO SCREEN (ACCOUNTING FOR ZERO-INDEXING)
    print("\n\tTotal pages = {},  Start page = {},   End page = {}".format(pn, ps + 1, pe + 1))

    return pn, ps, pe


def get_fonts(doc, pn):

    ### LOAD PAGE
    page = doc.loadPage(int(30))

    ### READ PAGE TEXT AS DICTIONARY
    blocks = page.getText("dict", flags=11)["blocks"]

    ### ITERATE THROUGH TEXT BLOCKS
    fn, fs, fc, ft = [], [], [], []
    for b in blocks:
        ### ITERATE THROUGH TEXT LINES
        for l in b["lines"]:
            ### ITERATE THROUGH TEXT SPANS
            for s in l["spans"]:
                fn.append(s["font"])
                fs.append(s["size"])
                fc.append(s["color"])
                ft.append(s["text"])

    d = {'Page': np.repeat(pn, len(fn)), 'Font': fn, 'Size': fs, 'Color': fc, 'Text': ft}
    df = pd.DataFrame (d, columns = ['Page', 'Font', 'Size', 'Color', 'Text'])

    return df
            

# ====================== Set Inputs =======================

PDF_Path  = '/Users/hthroop/Documents/HQ/CDAP20/PDFs'                         # PATH TO PROPOSAL PDFs
Out_Path  = '/Users/hthroop/Documents/HQ/CDAP20/PDFs/out'                             # PATH TO OUTPUT
Page_Lim  = 15                                          # PROPOSAL PAGE LIMIT

# ====================== Main Code ========================

PDF_Files = np.sort(glob.glob(os.path.join(PDF_Path, '*.pdf')))
Files_Skipped, Files_Font = [], []
for p, pval in enumerate(PDF_Files):

    ### OPEN PDF FILE
    Prop_Name = (pval.split('/')[-1]).split('.pdf')[0]
    # if Prop_Name != "20-EW20_2-0141-Umurhan":
    #     continue
    Doc = fitz.open(pval)
    print(colored("\n\n\n\t" + Prop_Name, 'green', attrs=['bold']))

    ### GET PAGES OF PROPOSAL
    try:
        Page_Num, Page_Start, Page_End = get_pages(Doc, Page_Lim)
    except RuntimeError:
        print("\tCould not read PDF")
        print(colored("\n\t!!!!!!!!!DID NOT SAVE!!!!!!!!!!!!!!!!", orange))
        Files_Skipped.append(pval)
        continue

    ### GET TEXT OF FIRST PAGE TO CHECK
    print("\n\tSample of first page:\t" + textwrap.shorten((get_text(Doc, Page_Start)[100:130]), 40))

    if Prop_Name != "20-EXO20-0016-Planavsky":

        print("\tSample of mid page:\t"     + textwrap.shorten((get_text(Doc, Page_Start + 8)[100:130]), 40))
        print("\tSample of last page:\t"    + textwrap.shorten((get_text(Doc, Page_End)[100:130]), 40))
        
    ### GRAB FONT SIZE & CPI
    cpi = []
    for i, val in enumerate(np.arange(Page_Start, Page_End)):
        cpi.append(len(get_text(Doc, val)) / 44 / 6.5)
        if i ==0:
            df = get_fonts(Doc, val)
        else:
            df = df.append(get_fonts(Doc, val), ignore_index=True)
    cpi = np.array(cpi)

    ### PRINT WARNINGS IF NEEDED (typical text font < 11.8 or CPI > 15.5 on > 1 page)
    CMF = Counter(df['Font'].values).most_common(1)[0][0]
    MFS = round(np.median(df[df['Text'].apply(lambda x: len(x) > 50)]["Size"]), 1)  ## only use text > 10 characters (excludes figure caption text)
    CPC, CPI = len(cpi[cpi > 15.5]), [round(x, 2) for x in cpi[cpi > 15.5]]
    print("\n\tMost common font:\t" + CMF)
    if MFS <= 11.8:
        print("\tMedian font size:\t", colored(str(MFS), 'yellow'))
    else:
        print("\tMedian font size:\t" + str(MFS))
    if CPC > 1:
        print("\tPages with CPI > 15.5:\t", textwrap.shorten(colored([np.arange(Page_Start, Page_End)[cpi > 15.5], CPI], 'yellow'), 70))
    if (MFS <= 11.8) | (CPC > 1):
        print(colored("\n\t!!!!! COMPLIANCE WARNING!!!!!", 'red'))

    ### PLOT 
    mpl.rc('xtick', labelsize=10)
    mpl.rc('ytick', labelsize=10)
    mpl.rc('xtick.major', size=5, pad=7, width=2)
    mpl.rc('ytick.major', size=5, pad=7, width=2)
    mpl.rc('xtick.minor', width=2)
    mpl.rc('ytick.minor', width=2)
    mpl.rc('axes', linewidth=2)
    mpl.rc('lines', markersize=5)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_title(Prop_Name + "   Size = " + str(round(np.median(df["Size"]), 1)) + "   CPI = " + str(round(np.median(cpi[cpi > 8]), 1)), size=12)
    ax.set_xlabel('Font Size', size=10)
    ax.set_ylabel('Density', size=10)
    ax.hist(df["Size"], bins=np.arange(6, 16, 0.5), density=True)
    fig.savefig(os.path.join(Out_Path, 'fc_' + pval.split('/')[-1]), bbox_inches='tight', dpi=100, alpha=True, rasterized=True)
    plt.close('all')
