from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

browser = webdriver.Firefox()

browser.get("https://place.map.kakao.com/1622750377")

time.sleep(2) #비동기로 처리해서 sleep이 꼭 있어야 하나? 이게 쓰레드 일시중지인가

element = browser.find_element(By.XPATH, "//div[@class='evaluation_review']/a[@class='link_more']")

print(element)

element.send_keys(Keys.ENTER)

time.sleep(2)

#div class="comment_info"
#p class="txt_comment"
elements_comment = browser.find_elements(By.XPATH, "//p[@class='txt_comment']/span")

print(len(elements_comment))

for comment in elements_comment:
    print(comment.text, "\n")

time.sleep(2)

star_web_elemets =browser.find_elements(By.XPATH, "//span[@class = 'ico_star star_rate']/span");

stars = []

for star in star_web_elemets:
    stars.append(star.get_attribute("style"))

print(stars)