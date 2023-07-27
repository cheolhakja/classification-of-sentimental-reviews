import requests
from bs4 import BeautifulSoup
url = 'https://place.map.kakao.com/1622750377'

response = requests.get(url)

if response.status_code == 200:
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    print(soup)

else : 
    print(response.status_code)