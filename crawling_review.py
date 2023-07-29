from selenium import webdriver
from selenium.webdriver.common.by import By
import time

browser = webdriver.Firefox()

browser.get("https://place.map.kakao.com/1622750377")

elements = browser.find_elements(By.XPATH, '//span[@class = "txt_more"]')

print(elements)
print(type(elements))
print(len(elements))

for elem in elements:
    print(elem.text)
