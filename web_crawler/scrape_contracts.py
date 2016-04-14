##################################################################################
#import modules

# for establishing a connection between python and the targeted website
import urllib2

# for retreiving information from websites 
from bs4 import BeautifulSoup as bs 

#convert xml as string to an ordered dictionary 
import xmltodict

# to read arguments from command line
import os 

#To use numpy arrays which are more efficient than regular data structures
import numpy as np

#To read the folder structure of the target
import sys

#to add randomised waiting time
import time, random
#import xml.etree.ElementTree as ET
#import json
##################################################################################

def get_xml_content(url):
    #Description: gets xml content from a url supplied
    #read supplied url and retrieve html
    #input: url - url to the xml version of data
    #output: An ordered dict of the xml data
    
    response = urllib2.urlopen(url).read()
    xml_dict = xmltodict.parse(response)
    return dict(xml_dict)


def scrape_contracts(xml_urls, file_prefix):
    #Uses a list of the xml url provided to scrape data from them and store to disk periodically
    #Writes various logs to monitor progress
    #input: xml_urls - a list of xml urls
    #output - None
    project_info = []
    
    for idx, url in enumerate(xml_urls):
        #random delay between 1 and 2 secs
        delay = random.randint(1,2)
        #pause execution for delay seconds
        time.sleep(delay)
        print 'file: ' + file_prefix + str(idx) + ' delay= ' + str(delay) 
        try:
            #try to retrieve information from url
            project_info.append(get_xml_content(url))
            
            #add completed url to the log of completed urls
            with open("./completed_urls.txt", "a") as complete_file:
                complete_file.write(url + '\n')
                complete_file.close()
        except:
            #add rejected urls to the log of rejected urls
            with open("./rejected_urls.txt", "a") as rejected_file:
                rejected_file.write(url + '\n')
                rejected_file.close()
        
        if idx % 10000 == 0 and idx != 0:
            #periodically write the data to file and reinitialise list for memory management
            file_name = file_prefix + str(idx) + '.gz'
            np.savetxt(file_name, project_info, delimiter=',', fmt='%s')
            project_info = []
            
            #add the index of last file to be written to disk
            with open("./saved_data_index.txt", "a") as saved_file:
                saved_file.write(file_prefix + str(idx) + '\n')
                saved_file.close()
    
    #Save remaining data to file    
    file_name = file_prefix + str(idx) + '.gz'
    np.savetxt(file_name, project_info, delimiter=',', fmt='%s')
    with open("./saved_data_index.txt", "a") as saved_file:
                saved_file.write(file_prefix + str(idx) + '\n')
                saved_file.close()
    
    
def main(argv):
    #needs three arguments 
    #1. start index of urls
    #2. end index of urls
    #3. prefix
    
    start_index = int(argv[0])
    end_index = int(argv[1])
    file_prefix = argv[2]
   
    xml_urls = np.loadtxt('./sample_xml_urls.txt', dtype=str)[start_index:end_index]
    scrape_contracts(xml_urls, file_prefix)
    
    
if __name__ == '__main__':
    #argv contains the arguments passed from the command line
    main(sys.argv[1:])
