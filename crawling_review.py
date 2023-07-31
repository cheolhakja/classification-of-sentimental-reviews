from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

browser = webdriver.Firefox()

browser.get("https://place.map.kakao.com/1622750377")

time.sleep(2) #비동기로 처리해서 sleep이 꼭 있어야 하나? 이게 쓰레드 일시중지인가

element = browser.find_element(By.XPATH, "//div[@class='evaluation_review']/a[@class='link_more']")
element.send_keys(Keys.ENTER)
time.sleep(2)
'''
xpath로 '더보기' 버튼을 element로 가져옴
클릭함
'''

elements_comment = browser.find_elements(By.XPATH, "//p[@class='txt_comment']/span")
time.sleep(2)
'''
xpath로 댓글들을 element list로 가져옴
클릭함
'''

star_web_elemets =browser.find_elements(By.XPATH, "//span[@class = 'ico_star star_rate']/span");
stars = []
for star in star_web_elemets:
    stars.append(star.get_attribute("style"))
'''
xpath로 별점들을 element list로 가져옴
(아직 100, 80, 40 이런 식으로 표기됨)
'''