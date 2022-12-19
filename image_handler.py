import praw
import config
import pandas as pd
import os
import urllib.request
from datetime import datetime, timedelta
import tqdm
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from PIL import Image
import requests
import urllib.request
from bs4 import BeautifulSoup


reddit = praw.Reddit(
    client_id=config.client_id,
    client_secret=config.client_secret,
    password=config.password,
    user_agent=config.user_agent,
    username=config.username,
)

class DownloadImageReddit:
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
        
    def create_df(self, keyword = [], date_range = False):
        
        for submission in tqdm.tqdm(self.subreddit.top(limit=self.number_posts), total=self.number_posts, desc='Downloading Posts'):

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
        #if one or more keywords are given key the rows that contain the keyword
        if keyword:
            df = df[df['title'].str.contains('|'.join(keyword), case=False)]

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

        #filter to only include images
        df = df[df['url'].str.contains('jpg') | df['url'].str.contains('png') | df['url'].str.contains('jpeg')]

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
        for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Downloading Images'):
            try:
                urllib.request.urlretrieve(row['url'], '{}/{}.jpg'.format(self.destination_folder, row['id']))
            except:
                print('Could not download image')

    def image_resize(self, path, output_path, size):
        #resize image and save to output_path if it does not exist then create it
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for image in tqdm.tqdm(os.listdir(path), total=len(os.listdir(path)), desc='Resizing Images'):
            try:
                img = Image.open('{}/{}'.format(path, image))
                img = img.resize(size)
                img.save('{}/{}'.format(output_path, image))
            except:
                print('Could not resize image')

class DownloadImageMorphMarket:
    def __init__(self, destination_folder, gene, snake_type = 'bp'):
        self.destination_folder = destination_folder
        self.gene = gene
        self.snake_type = snake_type

    def check_destination_folder(self,folder):
        #if folder images does not exist, create it
        if not os.path.exists(folder):
            os.makedirs(folder)

    def get_num_pages(self):
        # Make a request to the website and retrieve the HTML
        if self.snake_type == 'bp':
            response = requests.get('https://www.morphmarket.com/eu/c/reptiles/pythons/ball-pythons/gene/{}?epoch=22&page=1'.format(self.gene))
        elif self.snake_type == 'cs':
            response = requests.get('https://www.morphmarket.com/eu/c/reptiles/colubrids/corn-snakes/gene/{}?epoch=22&page=1'.format(self.gene))
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        # Find the pagination element
        pagination_element = soup.find('a', class_='page-link')
        #print(pagination_element)
        # Get the number of pages
        num_pages = pagination_element['title'].split(' ')[-1]

        return num_pages

    def scrape_images(self, num_pages):

        # Make a request to the website and retrieve the HTML
        
        #cap the number of pages to 15
        if int(num_pages) > 30:
            num_pages = 30
        #create a folder within the destination folder for the gene
        self.check_destination_folder('{}/{}'.format(self.destination_folder, self.gene))

        for i in tqdm.tqdm(range(1, int(num_pages)+1), total=int(num_pages), desc='Downloading Images of {}'.format(self.gene), unit='page'):
            page = i
            if self.snake_type == 'bp':
                url = 'https://www.morphmarket.com/eu/c/reptiles/pythons/ball-pythons/gene/{}?epoch=22&page={}'.format(self.gene, i)
            elif self.snake_type == 'cs':
                url = 'https://www.morphmarket.com/eu/c/reptiles/colubrids/corn-snakes/gene/{}?epoch=22&page={}'.format(self.gene, i)
            response = requests.get(url)
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            # Find all the listings on the page
            listings = soup.find_all('img')
            for i, listing in enumerate(listings):
                # Find the image url
                image_url = listing['src']
                try:
                    # Create a file name for the image with the name being the gene plus the page number and the image number
                    urllib.request.urlretrieve(image_url, '{}/{}/{}_{}_{}.jpg'.format(self.destination_folder, self.gene, self.gene, page, i))
                except: 
                    print('Could not download image')

    #function that checks whether the image is of the right size, otherwise deletes it, if an image cannot be opened it is also deleted
    def check_image_size(self, path, size):
        for gene in tqdm.tqdm(os.listdir('images/'), total = len(os.listdir('images/')), desc='Checking Image Size of {}'.format(path)):
            for image in os.listdir('images/{}'.format(path)):
                #handle hidden files
                if image.startswith('.'):
                    continue
                try:
                    img = Image.open('images/{}/{}'.format(path, image))
                    if img.size != size:
                        os.remove('images/{}/{}'.format(path, image))
                except:
                    os.remove('images/{}/{}'.format(path, image))

    def resize_images_inplace(self, path, size):
        for gene in tqdm.tqdm(os.listdir('images/'), total = len(os.listdir('images/')), desc='Resizing Images of {}'.format(self.gene)):
            for image in os.listdir('images/{}'.format(path)):
                img = Image.open('images/{}/{}'.format(path, image))
                img = img.resize(size)
                img.save('images/{}/{}'.format(path, image))

if __name__ == "__main__":

    morphs = ["amelanistic", "diffused", "motley","stripe","tessera","anerythristic","charcoal","scaleless","cinder","hypo"]
    for morph in morphs:
        mm = DownloadImageMorphMarket(destination_folder='images', gene=morph, snake_type='cs')
        number_pages = mm.get_num_pages()
        mm.scrape_images(number_pages)
    for morph in morphs:
        mm.check_image_size(path=morph, size=(225, 190))
    for morph in morphs:
        mm.resize_images_inplace(path=morph, size=(192, 192))
