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
from matplotlib.patches import Rectangle

fitz.TOOLS.mupdf_display_errors(False)

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

    check_words = ["contents", "c o n t e n t s", "budget", "cost", "costs",
                   "submitted to", "purposely left blank", "restrictive notice"]

    ### LOOP THROUGH PDF PAGES
    ps = 0
    for i, val in enumerate(np.arange(pn)):
            
        ### READ IN TEXT FROM THIS PAGE AND NEXT PAGE
        t1 = get_text(d, val)
        t2 = get_text(d, val + 1)
        
        ### FIND PROPOSAL START USING END OF SECTION X
        if ('SECTION X - Budget' in t1) & ('SECTION X - Budget' not in t2):

            ### set start page
            ps = val + 1

            ### ATTEMPT TO CORRECT FOR (ASSUMED-TO-BE SHORT) COVER PAGES
            if len(t2) < 500:
                ps += 1
                t2 = get_text(d, val + 2)

            ### ACCOUNT FOR TOC OR SUMMARIES
            if any([x in t2.lower() for x in check_words]):
                ps += 1

            ### ASSUMES AUTHORS USED FULL PAGE LIMIT
            pe  = ps + (pl - 1) 
                        
        ### EXIT LOOP IF START PAGE FOUND
        if ps != 0:
            break 

    ### ATTEMPT TO CORRECT FOR TOC > 1 PAGE OR SUMMARIES
    if any([x in get_text(d, ps).lower() for x in check_words]):
        ps += 1
        pe += 1

    ### CHECK THAT PAGE AFTER LAST IS REFERENCES
    Ref_Words = ['references', 'bibliography', "r e f e r e n c e s", "b i b l i o g r a p h y"]

    if not any([x in get_text(d, pe + 1).lower() for x in Ref_Words]):

        ### IF NOT, TRY NEXT PAGE (OR TWO) AND UPDATED LAST PAGE NUMBER
        if any([x in get_text(d, pe + 2).lower() for x in Ref_Words]):
            pe += 1
        elif any([x in get_text(d, pe + 3).lower() for x in Ref_Words]):
            pe += 2

        ### CHECK THEY DIDN'T GO UNDER THE PAGE LIMIT
        if any([x in get_text(d, pe).lower() for x in Ref_Words]):
            pe -= 1
        elif any([x in get_text(d, pe - 1).lower() for x in Ref_Words]):
            pe -= 2
        elif any([x in get_text(d, pe - 2).lower() for x in Ref_Words]):
            pe -= 3
        elif any([x in get_text(d, pe - 3).lower() for x in Ref_Words]):
            pe -= 4

    ### PRINT TO SCREEN (ACCOUNTING FOR ZERO-INDEXING)
    print("\n\tTotal pages = {},  Start page = {},   End page = {}".format(pn, ps + 1, pe + 1))

    ### PRINT IF WENT OVER PAGE LIMIT
    if pe - ps > 14:
        print(colored("\n\t!!!!! PAGE LIMIT WARNING -- OVER !!!!!", 'blue'))
    if pe - ps < 13:
        print(colored("\n\t!!!!! PAGE LIMIT WARNING -- UNDER !!!!!", 'blue'))

    return pn, ps, pe


def get_fonts(doc, pn):

    ### LOAD PAGE
    page = doc.loadPage(int(pn))

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
Out_Path  = os.path.join(PDF_Path, 'out')                             # PATH TO OUTPUT
Page_Lim  = 15                                          # PROPOSAL PAGE LIMIT

# ====================== Main Code ========================

PDF_Files = np.sort(glob.glob(os.path.join(PDF_Path, '*.pdf')))
Files_Skipped, Files_Font = [], []
for p, pval in enumerate(PDF_Files):

    ### OPEN PDF FILE
    Prop_Name = (pval.split('/')[-1]).split('.pdf')[0]
    Doc = fitz.open(pval)
    print(colored("\n\n\n\t" + Prop_Name, 'green', attrs=['bold']))

    ### GET PAGES OF PROPOSAL (DOES NOT ACCOUNT FOR ZERO INDEXING; NEED TO ADD 1 WHEN PRINTING)
    try:
        Page_Num, Page_Start, Page_End = get_pages(Doc, Page_Lim)
    except RuntimeError:
        print("\tCould not read PDF")
        print(colored("\n\t!!!!!!!!!DID NOT SAVE!!!!!!!!!!!!!!!!", orange))
        Files_Skipped.append(pval)
        continue

    ### GET TEXT OF FIRST PAGE TO CHECK
    print("\n\tSample of first page:\t" + textwrap.shorten((get_text(Doc, Page_Start)[100:130]), 40))
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
    MFS = round(np.median(df[df['Text'].apply(lambda x: len(x) > 50)]["Size"]), 1)  ## only use text > 50 characters (excludes random smaller text; see histograms for all)
    CPC, CPI = len(cpi[cpi > 15.5]), [round(x, 1) for x in cpi[cpi > 15.5]]
    print("\n\tMost common font:\t" + CMF)
    if MFS <= 11.8:
        print("\tMedian font size:\t", colored(str(MFS), 'yellow'))
    else:
        print("\tMedian font size:\t" + str(MFS))
    if CPC > 1:
        print("\tPages with CPI > 15.5:\t", textwrap.shorten(colored((np.arange(Page_Start, Page_End)[cpi > 15.5] + 1).tolist(), 'yellow'), 70))
        print("\t\t\t\t", textwrap.shorten(colored(CPI, 'yellow'), 70))
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
    ax.set_title(Prop_Name + "   Median Font = " + str(MFS) + "pt    CPI = " + str(round(np.median(cpi[cpi > 8]), 1)), size=11)
    ax.set_xlabel('Font Size', size=10)
    ax.set_ylabel('Density', size=10)
    ax.axvspan(11.8, 12.2, alpha=0.5, color='gray')
    ax.hist(df["Size"], bins=np.arange(5.4, 18, 0.4), density=True)
    fig.savefig(os.path.join(Out_Path, 'fc_' + pval.split('/')[-1]), bbox_inches='tight', dpi=100, alpha=True, rasterized=True)
    plt.close('all')
