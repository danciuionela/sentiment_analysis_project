import sentiment_analysis_module as s
import webscrapper

movies = webscrapper.get_review_movies() 

for t, r in movies.items():
    print('\nTITLE:  '+t)
    for review in r:
        print(review)
        print("Analiza rezultata: ", s.sentiment(review))
        print("------------------------------------------------------------------------------------------------------------------------")

