import review_preprocess

star_ratings = review_preprocess.load_star_ratings()
print(star_ratings)
print(len(star_ratings))

reviews = review_preprocess.load_reviews()
for review in reviews:
    print(review)
print(len(reviews))