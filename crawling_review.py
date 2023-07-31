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
stars_not_yet_parsed = []
for star in star_web_elemets:
    stars_not_yet_parsed.append(star.get_attribute("style"))
'''
xpath로 별점들을 star_web_elemets로 가져옴
(아직 100, 80, 40 이런 식으로 표기됨)

stars_not_yet_parsed: width: 100%;형식의 별점
'''

stars = []
for i in stars_not_yet_parsed[2: -1]:
    string_star = i[7:-2]
    stars.append(int(string_star) // 20)

print(stars)
'''
stars_not_yet_parsed 리스트에서 맨 앞의 두개와 맨 뒤의 한개 별점을 리스트 슬라이싱으로 제거함 (리뷰와 상관없는 별점임)
width: 100%; 형식에서 리스트 슬라이싱을 사용하여 숫자만 추출함
'''

comments = []
for i, comment in enumerate(elements_comment):
    comments.append(comment.text)
'''
댓글 리스트를 만듦
-> 별점리스트의 i번째 element와 댓글리스트의 i번째 element가 대응됨
'''