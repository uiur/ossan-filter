# -*- coding: utf-8 -*-

import requests
import shutil
import os.path
import os

user_id = os.environ['AZURE_TOKEN']

url = "https://api.datamarket.azure.com/Bing/Search/Image?$format=json&Query=%27%E5%8D%92%E6%A5%AD%E5%BC%8F%27"

# url = "https://api.datamarket.azure.com/Data.ashx/Bing/Search/Image?Query='%E6%A0%A1%E9%95%B7'&$skip=500&$format=json"

while True:
    print(url)
    r = requests.get(url, auth=(user_id, user_id))

    if not r.ok:
        print(r.text)
        break

    data = r.json()
    results = data['d']['results']

    for result in results:
        uid = result['ID']
        media_url = result['MediaUrl']

        path = u'download/卒業式/' + uid + '.png'
        if os.path.exists(path):
            continue

        try:
            r = requests.get(result['MediaUrl'], stream=True)

            if r.ok:
                with open(path, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)

                print(media_url)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
            pass

    if not data['d'].has_key('__next'):
        break

    url = data['d']['__next'] + '&$format=json'
