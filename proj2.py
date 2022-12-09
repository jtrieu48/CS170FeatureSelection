import csv
import pandas as pd
import numpy as np
import time
import math
import copy
import sys


def main():
    print('Feature Selection w/ Nearest Neighbor: ')
    filename = input('\nInput File Name: ')
    file_in = filename
    filename = open(filename, 'r')

    # Using CSV Library for reading and parsing file: 
    # https://docs.python.org/3/library/csv.html
    reader = csv.reader(filename, delimiter = ' ', skipinitialspace=True)
    parse = len(next(reader))

    algorithm = int(input('Select Algorithm:\n'
                     '\n1. Foward Selection'
                     '\n2. Backward Elimination\n\n'))


    print('\nThis dataset has ' + str(parse-1) + ' features (not including class attribute).\n\n')

    #Algorithm Select
    if algorithm == 1:
        return forward_search(file_in, parse)
    elif algorithm == 2:
        return backward_search(file_in, parse)
    return 'Not valid alg choice (Choose 1 or 2)'




def forward_search(file_in, feats):
    start_time = time.time()
    seen_feats = set()
    d = {}

    print('\nRunning Forward Selection\n\n')

    for i in range(1, feats):
        max_accur = 0
        finalcol = 0


        for j in range(1, feats):
            if j not in seen_feats:
                # Using deep copy
                s_temp = copy.deepcopy(seen_feats)

                # Checking Row
                s_temp.add(j)

                # Using Panda for parsing:
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_fwf.html
                data_frame = pd.read_fwf(file_in, header=None)

                # Using Deep copy to be updated into function
                # https://docs.python.org/3/library/copy.html
                data_frame_copy = data_frame.copy(deep=True)[:-1]
                curr_accur = leave_one_out_cross_validation(feats, s_temp, data_frame_copy)
                coltemp = j
                print('Using feature(s) ' + str(s_temp) + ' accuracy is ' + "{:.1%}".format(curr_accur))

                # Updating max_accur as better accuracys are found
                if curr_accur >= max_accur:
                    max_accur = curr_accur
                    f_accur = curr_accur
                    finalcol = coltemp

        # Found best Column in set
        seen_feats.add(finalcol)
        seen_copy = copy.deepcopy(seen_feats)
        d[f_accur] = seen_copy

        print('Feature set ' + str(seen_feats) + ' was best, accuracy is ' + "{:.1%}".format(f_accur) + '\n')

    print('Finished search!! The best feature subset is ' + str(d[max(d.keys())]) + ' which has an accuracy of ' + "{:.1%}".format(max(d.keys())) + '\n')

    print('Found in: ' + str(round(time.time() - start_time, 2)) + ' seconds.')


def backward_search(file_in, feats):
    start = time.time()
    seen_feats = set()

    print('\nRunning Backwards Elimination\n\n')

    for j in range(1, feats):
        seen_feats.add(j)
    d = {}

    # Starting from 1
    s_temp = copy.deepcopy(seen_feats)

    #Reading in file
    data_frame = pd.read_fwf(file_in, header=None)
    

    data_frame_copy = data_frame.copy(deep=True)

    # Calc accuracy using leave_one_out_cross_validation
    curr_accur = leave_one_out_cross_validation(feats, s_temp, data_frame_copy)

   
    print('Using feature(s) ' + str(s_temp) + ' accuracy is ' + "{:.1%}".format(curr_accur))

    print('Feature set ' + str(s_temp) + ' was best, accuracy is ' + "{:.1%}".format(curr_accur) + '\n')

    for i in range(2, feats):
        max_accur = 0
        finalcol = 0

        for j in range(1, feats):
            if j in seen_feats:
                # Using Deep Copy
                s_temp = copy.deepcopy(seen_feats)
                
                #Temp remove 1
                s_temp.remove(j)

                # Using Pandas for Parsing
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_fwf.html
                data_frame = pd.read_fwf(file_in, header=None)

                # Deep copy the data frame
                data_frame_copy = data_frame.copy(deep=True)

                curr_accur = leave_one_out_cross_validation(feats, s_temp, data_frame_copy)
                coltemp = j

                
                print('Using feature(s) ' + str(s_temp) + ' accuracy is ' + "{:.1%}".format(curr_accur))

                # Update Max accuracys as found
                if curr_accur >= max_accur:
                    max_accur = curr_accur
                    f_accur = curr_accur
                    finalcol =coltemp

        # Remove the final column
        seen_feats.remove(finalcol)
        seen_copy = copy.deepcopy(seen_feats)
        d[f_accur] = seen_copy

       
        print('Feature set ' + str(seen_feats) + ' was best, accuracy is ' + "{:.1%}".format(f_accur) + '\n')

    print('Finished search!! The best feature subset is ' + str(d[max(d.keys())]) + ' which has an accuracy of ' + "{:.1%}".format(max(d.keys())) + '\n')

    print('Found in: ' + str(round(time.time()-start, 2)) + ' seconds.')


def leave_one_out_cross_validation(cols, seen, data_frame_copy):
    nr = len(data_frame_copy.index)
    classified = 0

    data_frame = data_frame_copy.copy(deep=True)

    # Using Numpy for program efficiency 
    # https://numpy.org/doc/stable/user/index.html#user
    tempdf = data_frame.to_numpy()
    data_frame_copy = tempdf

    # Check if Column is empty (0's)
    for i in range(1, cols):
        if i not in seen:
            data_frame_copy[:, i] = 0.0

    # Computing Distances
    for j in range(nr):
        # Classifynig Data
        obj_class = data_frame_copy[j][1:cols]
        class_name = data_frame_copy[j][0]

        nearest_dist = sys.maxsize
        nearest_location = sys.maxsize

        for l in range(nr):
            dist = 0

            # Not in same row, calculate and compare distance
            if j != l:
                d = {}

                # Using Numpy to calc distance in same row
                dist = math.sqrt(np.sum(np.power(obj_class - data_frame_copy[l][1:cols], 2)))

                # Compare & update dist
                if dist <= nearest_dist:
                    nearest_dist = dist
                    nearest_location = l + 1
                    nearest_n_label = data_frame_copy[nearest_location - 1][0]

        # Found classified & increment
        if class_name == nearest_n_label:
            classified += 1

    
    accur = classified/nr
    return accur


if __name__ == "__main__":
    main()