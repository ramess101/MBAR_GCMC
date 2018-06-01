# -*- coding: utf-8 -*-
"""
Extracting Potoff data
"""

def extract_Potoff(input_path,output_path,with_uncertainties=False):

    out_file =open(output_path,'w')
    
    if with_uncertainties:
        out_file.write('T (K) rhol urhol (gm/cm3) rhov urhov (gm/cm3) P uP (bar) Hv uHv (kJ/mol) Z uZ'+'\n')
    else:
        out_file.write('T (K) rhol (gm/cm3) rhov (gm/cm3) P (bar) Hv (kJ/mol) Z'+'\n')
    
    lines = []                  # Declare an empty list named "lines"
    with open (input_path, 'rt') as in_file:  # Open file lorem.txt for reading of text data.
        for line in in_file:  # For each line of text in in_file, where the data is named "line",
    #        lines.append(line.rstrip('\n'))   # add that line to our list of lines, stripping newlines.
            lines.append(line) #This is likely unnecessary now
        for element in lines:            # For each element in our list,
            index_open=0
            index_open0=0
            index_close=0
            index_close0=0
            
            while index_open < len(element): #While index is a number smaller than the number of letters in str.
                index_open = element.find('(', index_open0) #set index to location of first remaining occurrence of "("
                index_close = element.find(')', index_close0) #set index to location of first remaining occurrence of ")"
                if index_open == -1:         # If nothing was found,
                    break            # exit the while loop. Otherwise,
#                print(element[index_close0:index_open])
                out_file.write(element[index_close0:index_open]+'\t')
                
                if with_uncertainties:        
                    out_file.write(element[index_open+1:index_close]+'\t')
                index_open0 = index_open + len('(')      # increment the index by the number of characters in substr, and repeat.
                index_close0 = index_close + len(')')
                
            out_file.write('\n')
    
    out_file.close()    

def main():

    input_path='H:/MBAR_GCMC/Potoff_literature/practice.txt'
    output_path='H:/MBAR_GCMC/Potoff_literature/practice_out.txt' 

    extract_Potoff(input_path,output_path)

if __name__ == '__main__':
    '''
    python extract_Potoff.py
  
    '''

    main()         