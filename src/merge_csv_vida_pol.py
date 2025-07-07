######################################################################
# Author: Rohan Dahale, Date: 12 July 2024
######################################################################

# Import libraries
import numpy as np
import pandas as pd
import ehtim as eh
import ehtim.scattering.stochastic_optics as so
from preimcal import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import argparse
import os
import glob
from tqdm import tqdm
import itertools 
import sys
from copy import copy
from utilities import *
colors, titles, labels, mfcs, mss = common()

codedir = os.getcwd()

######################################################################
# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--truthcsv', type=str, default='none', help='path of truth .csv')
    p.add_argument('--kinecsv', type=str, default='none', help='path of kine .csv')
    p.add_argument('--ehtcsv',  type=str, default='none', help='path of ehtim .csv')
    p.add_argument('--dogcsv',  type=str, default='none', help='path of doghit .csv')
    p.add_argument('--ngcsv',   type=str, default='none', help='path of ngmem .csv')
    p.add_argument('--rescsv',  type=str, default='none', help='path of resolve .csv')
    p.add_argument('--modelingcsv',  type=str, default='none', help='path of modeling .csv')
    p.add_argument('-o', '--outpath', type=str, default='./vida_pol.csv', help='name of output file with path')

    return p

# List of parsed arguments
args = create_parser().parse_args()
outpath = args.outpath



paths={}
if args.truthcsv!='none' and os.path.exists(args.truthcsv):
    paths['truth']=args.truthcsv
if args.kinecsv!='none' and os.path.exists(args.kinecsv):
    paths['kine']=args.kinecsv
if args.rescsv!='none' and os.path.exists(args.rescsv):
    paths['resolve']=args.rescsv
if args.ehtcsv!='none' and os.path.exists(args.ehtcsv):
    paths['ehtim']=args.ehtcsv
if args.dogcsv!='none' and os.path.exists(args.dogcsv):
    paths['doghit']=args.dogcsv
if args.ngcsv!='none'and os.path.exists(args.ngcsv):
    paths['ngmem']=args.ngcsv
if args.modelingcsv!='none' and os.path.exists(args.modelingcsv):
    paths['modeling']=args.modelingcsv
    
dfs = []

for method_name, file_path in paths.items():
    df = pd.read_csv(file_path)

    # Extract filename without extension and rename the "file" column to "time"
    df["time"] = df["file"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df = df.drop(columns=["file"])

    # Rename other columns to include method name
    df = df.rename(columns=lambda col: f"{col}_{method_name}" if col != "time" else col)

    dfs.append(df)

# Merge on the "time" column to keep it as a single column
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on="time", how="outer")

# Save the merged CSV
merged_df.to_csv(outpath+'.csv', index=False)