"""
This is a crawler for auction data at Philip.
Please download chrome driver from " https://chromedriver.chromium.org/ " according to your chrome browser version and OS.
And change path variable to where your chrome driver is.
"""


from os import nice
import time
#from datetime import datetime
import requests
from selenium import webdriver # pip install selenium
#from selenium.common.exceptions import NoSuchElementException

#f = open('philip_data.txt', 'w')
#start = str(datetime.now())

philip = 'https://www.phillips.com/auctions/past/filter/Departments%3DLatin!Editions!Contemporary!Online!Photographs/sort/newest'
driver = webdriver.Chrome(executable_path = r'/Users/jchuo/Downloads/chromedriver')
# go to Philip's website
driver.get(philip)

# scroll to page bottom
driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
time.sleep(5)

# collect all urls of auctions
auctions = []
n = 1
while True:
    try:
        xpath = driver.find_element_by_xpath('//*[@id="main-list-backbone"]/li[{}]/div[2]/h2/a'.format(n))
    except:
        break
    html = xpath.get_attribute('outerHTML')
    ind = html.find('href="')
    url = 'https://www.phillips.com'
    for b in html[ind+6:]:
        if b != '"':
            url += b
        else:
            break
    print(url)
    auctions.append(url)
    n += 1
print('Total {} auctions'.format(len(auctions)))

# for each auction, collect the sold art works
ID = 1
for i in range(len(auctions)):
    print('collecting art works from the {}/{} url'.format(i+1, len(auctions)))
    driver.get(url)
    
    n = 1
    while True:

        # get image
        try:
            image_xpath = driver.find_element_by_xpath('/html/body/div[3]/div/div[2]/div/div/div/div[2]/ul/li[{}]/div/div[1]/a/div/img'.format(n))
        except:
            break
        image_html = image_xpath.get_attribute('outerHTML')
        ind = image_html.find('src="')
        src = ''
        for b in image_html[ind+5:]:
            if b != '"':
                src += b
            else:
                break
        image = requests.get(src)
        image_file = 'images_philip/{}.jpg'.format(ID)
        with open(image_file, 'wb') as f:
            f.write(image.content)
            f.close()

        # get artist name
        artist_xpath = driver.find_element_by_xpath('/html/body/div[3]/div/div[2]/div/div/div/div[2]/ul/li[{}]/div/a/p[2]'.format(n))
        artist_html = artist_xpath.get_attribute('outerHTML')
        ind = artist_html.find('title="')
        artist = ''
        for b in artist_html[ind+7:]:
            if b != '"':
                artist += b
            else:
                break
        print('artist: ' + artist)

        # get title of art work
        title_xpath = driver.find_element_by_xpath('/html/body/div[3]/div/div[2]/div/div/div/div[2]/ul/li[{}]/div/a/p[3]'.format(n))
        title_html = title_xpath.get_attribute('outerHTML')
        ind = title_html.find('title="')
        title = ''
        for b in title_html[ind+7:]:
            if b != '"':
                title += b
            else:
                break
        print('title: ' + title)

        # find pre-auction low estimate
        low_xpath = driver.find_element_by_xpath('/html/body/div[3]/div/div[2]/div/div/div/div[2]/ul/li[{}]/div/a/p[4]/span/span[1]'.format(n))
        low_html = low_xpath.get_attribute('outerHTML')
        ind = low_html.find('<span>$<!-- -->')
        low = ''
        for b in low_html[ind+15:]:
            if b != '<':
                if b != ',':
                    low += b
            else:
                break
        print('low estimate: ' + low)

        # find pre-auction high estimate
        high_xpath = driver.find_element_by_xpath('/html/body/div[3]/div/div[2]/div/div/div/div[2]/ul/li[{}]/div/a/p[4]/span/span[2]'.format(n))
        high_html = high_xpath.get_attribute('outerHTML')
        ind = high_html.find('<span>')
        high = ''
        for b in high_html[ind+6:]:
            if b != '<':
                if b != ',':
                    high += b
            else:
                break
        print('high estimate: ' + high)

        # get price
        try:
            price = driver.find_element_by_xpath('/html/body/div[3]/div/div[2]/div/div/div/div[2]/ul/li[{}]/div/a/p[5]'.format(n))
        except:
            break
        price_html = price.get_attribute('outerHTML')
        for ind in range(-4, -100, -1):
            if price_html[ind] == '>':
                break
        price = price_html[ind+1:-4]
        print('price: ' + price)

        n += 1
        ID += 1

print('total {} art works'.format(ID))