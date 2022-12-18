import praw
import config
import pandas as pd
import os
import urllib.request
from datetime import datetime, timedelta


reddit = praw.Reddit(
    client_id=config.client_id,
    client_secret=config.client_secret,
    password=config.password,
    user_agent=config.user_agent,
    username=config.username,
)

class DownloadImage:
    def __init__(self, subreddit, destination_folder, number_posts, save_csv = False):
        self.subreddit = reddit.subreddit(subreddit)
        self.destination_folder = destination_folder
        self.number_posts = number_posts
        self.storage_dict = {
            "title": [],
            "score": [],
            "id": [],
            "url": [],
            "comms_num": [],
            "created": [],
            "body": []
        }
        self.save_csv = save_csv
        
    def create_df(self, keyword = ' ', date_range = False):
        
        for submission in self.subreddit.top(limit=self.number_posts):

            #add to storage dict
            self.storage_dict["title"].append(submission.title)
            self.storage_dict["score"].append(submission.score)
            self.storage_dict["id"].append(submission.id)
            self.storage_dict["url"].append(submission.url)
            self.storage_dict["comms_num"].append(submission.num_comments)
            self.storage_dict["created"].append(submission.created_utc)
            self.storage_dict["body"].append(submission.selftext)


        #clean df timestamp
        self.storage_dict["created"] = pd.to_datetime(self.storage_dict["created"], unit='s')
        
        df = pd.DataFrame(self.storage_dict)

        #filter by keyword
        if keyword != ' ':
            df = df[df['title'].str.contains(keyword, case=False)]

        #filter by date range
        if date_range == 'month':
            df = df[df['created'] > datetime.now() - timedelta(days=30)]
        elif date_range == 'week':
            df = df[df['created'] > datetime.now() - timedelta(days=7)]
        elif date_range == 'day':
            df = df[df['created'] > datetime.now() - timedelta(days=1)]
        elif date_range == 'year':
            df = df[df['created'] > datetime.now() - timedelta(days=365)]
        else:
            df = df

        #save to csv
        if self.save_csv:
            #if folder csv_output does not exist, create it
            if not os.path.exists('csv_output'):
                os.makedirs('csv_output')
            df.to_csv('csv_output/{}_{}_{}.csv'.format(self.subreddit, keyword, date_range), index=False)

        return df

    def download_images(self, df):
        #if folder images does not exist, create it
        if not os.path.exists(self.destination_folder):
            os.makedirs(self.destination_folder)

        #download images
        for index, row in df.iterrows():
            try:
                urllib.request.urlretrieve(row['url'], '{}/{}.jpg'.format(self.destination_folder, row['id']))
            except:
                print('Could not download image')

        
if __name__ == '__main__':
    download = DownloadImage('ballpython', 'images', 1000, save_csv=True)
    download.download_images(download.create_df(keyword='SNAKE', date_range='year'))



