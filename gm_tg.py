#!/usr/bin/env python3

'''
GRAPHMONSTER TWITTERGRAB
'''

from credentials import consumer_key, consumer_secret, access_token_secret, access_token
import pandas as pd
import tweepy
import re
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num", default = 16)

args = parser.parse_args()

def main():
    print("\n[-=GRAPHMONSTER=-]")
    print("- twittergrab function")
    print("----- Processing csv file")
    twittergrab("gm.csv",args.num)
    print("----- Saved community-identification.txt\n")

def twittergrab(file,numcomms):
    # Get graphmonster data
    data = pd.read_csv(file)
      
    
    # Authorise with the Twitter api
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    
    # Get list of communities by size, and slice by keepers
    sizeranked_comms = data.community.value_counts().index
    keepcomms = sizeranked_comms[:int(numcomms)]
    
    # Make api call
    print("----- Calling api ...")
    with open("community-identification.txt", "w") as outfile:
        for kc in keepcomms:
            comm_df = data[data['community'] == kc].sort_values(by="degree", ascending=False)
            topnames = list(comm_df['name'][:10])  
            users = api.lookup_users(topnames)

            degree_dict = dict(zip(data.name,data.degree))

            outfile.write("Community " + str(kc) + "\n" + "="*40)
            for c,u in enumerate(users):
                outfile.write("\n" + str(c+1) + " -- degree:" + str(degree_dict[topnames[c]])+"\n")
                outfile.write("user: " + u.name + "\n")
                outfile.write("screen_name: " + u.screen_name + "\n")
                outfile.write("description: " + re.sub("\n","//", u.description) + "\n\n" + "--\n")

if __name__ == '__main__':
    main()
