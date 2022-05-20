from requests import get
from bs4 import BeautifulSoup

#only movies with reviews
def get_review_movies():
    url = 'https://www.imdb.com/search/title/?title_type=feature,tv_movie&release_date=2022-01-01,2022-04-20&count=20'
    url_first = 'https://www.imdb.com' #first part of the url for reviews
    url_last = 'reviews?ref_=tt_urv'  #last part of the url for reviews
    base_response = get(url)
    html_soup = BeautifulSoup(base_response.text, 'html.parser')

    movie_containers = html_soup.find_all('div', class_ = 'lister-item mode-advanced')

    reviews_data = {}
    for container in movie_containers:
        # only if the movie has metascore - each review average
        if container.find('div', class_ = 'ratings-metascore') is not None:
            reviews_number = 6
            url_middle = container.find('a')['href']
            response_reviews = get(url_first + url_middle+ url_last)
            reviews_soup = BeautifulSoup(response_reviews.text, 'html.parser')
            reviews_containers = reviews_soup.find_all('div', class_ = 'imdb-user-review')
            if len(reviews_containers) < reviews_number:
                reviews_number = len(reviews_containers)
            reviews_bodies = []
            for review_index in range(reviews_number):
                review_container = reviews_containers[review_index]
                review_body = review_container.find('div', class_ = 'text').text.strip()
                reviews_bodies.append(review_body)
                reviews_data[container.h3.a.text] = reviews_bodies
    return reviews_data


