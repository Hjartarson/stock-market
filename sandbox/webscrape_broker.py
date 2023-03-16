import urllib.request
import json
from bs4 import BeautifulSoup
import requests

html = requests.get('https://tingent.se/jobs').content
soup = BeautifulSoup(html, 'html.parser')

# TINGENT
data = json.loads(soup.find('script', type='application/json').text)
data
pd.DataFrame(data['props']['pageProps']['jobsData']).to_excel('tingent.xlsx')

# KEYMAN
html = requests.get('https://keyman.se/uppdrag/').content
soup = BeautifulSoup(html, 'html.parser')

for a in soup.find_all('div', 
     class_ = 'post-content-wrap'):
    print(a)