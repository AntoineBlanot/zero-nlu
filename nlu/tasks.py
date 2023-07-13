from typing import Any, Dict, List

class Task():

    def generate_inputs(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class IntentRecognition(Task):

    def generate_inputs(self, user_sentence: str, candidate_labels: List[str], *args, **kwargs) -> str:
        """
        Generate inputs using a mapping. Return a dict with `document` and `queries` as keys.
        """
        mapping = {
            "watch-in-person": "I like to watch where the action is.",
            "watches-on-tv": "I like to watch on the television.",

            "natural-wonders": "I like nature.",
            "man-made-monuments-answer": "I like monuments.",

            "topic-books-physical-books": "I like physical books.",
            "topic-books-ebooks": "I like electronic books.",

            "topic-books-most-sold-book-rowling": "It is J. K. Rowling.",
            "topic-books-most-sold-book-tolkien": "It is J. R. R. Tolkien.",

            "topic-food-for-breakfast": "It is for breakfast.",
            "topic-food-for-lunch": "It is for lunch.",
            "topic-food-for-dinner": "It is for dinner.",

            "topic-hometown-type-of-building-apartment-answer": "I live in an apartment.",
            "topic-hometown-type-of-building-house-answer": "I live in a house.",

            "topic-speaker-age-less-than-18-answer": "I am less than 18 years old.",
            "topic-speaker-age-greater-than-18-answer": "I am more than 18 years old.",

            "topic-travel-homecountry-favorite-hemisphere-north": "I like the North.",
            "topic-travel-homecountry-favorite-hemisphere-south": "I like the South.",

            "favorite-continent-asia": "I like Asia.",
            "favorite-continent-australia": "I like Australia.",
            "favorite-continent-africa": "I like Africa.",
            "favorite-continent-antarctica": "I like Antartica.",
            "favorite-continent-europe": "I like Europe.",
            "favorite-continent-north-america": "I like North America.",
            "favorite-continent-south-america": "I like South America.",

            "likes-tennis": "I like tennis.",
            "likes-baseball": "I like baseball.",
            "likes-basketball": "I like basketball.",

            "topic-books-likes-both-genre": "I like both genres.",
            "topic-books-likes-fiction": "I like fiction.",
            "topic-books-likes-non-fiction": "I like non-fiction.",

            "games": "I like games.",
            "gardening": "I like gardening.",
            "working-out": "I like working-out.",

            "topic-hometown-big-city": "It is a big city/town.",
            "topic-hometown-small-city": "It is a small city/town.",

            "topic-profession-generic-profession": "I have a classic profession.",
            "topic-profession-evil-profession": "I have a profession related to evil.",
            "student": "I am a student.",

            "topic-travel-homecountry-human-from-india": "I am from India.",
            "topic-travel-homecountry-human-from-japan": "I am from Japan.",
            "topic-travel-homecountry-human-from-usa": "I am from the USA.",
            "topic-travel-homecountry-human-from-china": "I am from China.",
            "topic-travel-homecountry-sarcastic-location": "I am from a weird place.",

            "favorite-season-summer": "I like summer.",
            "favorite-season-winter": "I like winter.",
            "favorite-season-spring": "I like spring.",
            "favorite-season-autumn": "I like autumn.",

            "likes-to-play-sports": "I like to play sports.",
            "likes-to-watch-sports-or-fallback": "I like watch sports.",

            "topic-day-one-session-one-age-wrappingup-childhood": "I like childhood.",
            "topic-day-one-session-one-age-wrappingup-adulthood": "I like adulthood.",
            "topic-day-one-session-one-age-wrappingup-oldage": "I like the old age.",

            "topic-language-learn-english-at-school": "At school.",
            "topic-language-learn-english-at-home": "At home.",

            "topic-birthday-days-february": "28 or 29 days.",
            "topic-birthday-days-thirty": "30 days.",
            "topic-birthday-days-thirtyone": "31 days.",

            "topic-day-three-food-noodles": "I like noodles.",
            "topic-day-three-food-burgers": "I like burgers.",
            "topic-day-three-food-pizza": "I like pizza.",

            "topic-day-three-number-meals-lessthan-three": "Less than 3.",
            "topic-day-three-number-meals-between-three-six": "Between 3 and 6.",
            "topic-day-three-number-meals-greaterthan-six": "Greater than 6.",

            "topic-day-four-school-favorite-subject-science": "I like science lessons.",
            "topic-day-four-school-favorite-subject-social": "I like social lessons.",
            "topic-day-four-school-favorite-subject-math": "I like mathematics lessons.",
            "topic-day-four-school-favorite-subject-english": "I like english lessons.",

            "topic-day-four-school-extra-curriculars-music": "I like music.",
            "topic-day-four-school-extra-curriculars-sports": "I like sports.",

            "topic-day-five-weather-sun": "I like the sun.",
            "topic-day-five-weather-rain": "I like the rain.",

            "topic-day-five-weather-favorite-season-summer": "I like summer.",
            "topic-day-five-weather-favorite-season-winter": "I like winter.",
            "topic-day-five-weather-favorite-season-spring": "I like spring.",
            "topic-day-five-weather-favorite-season-fall": "I like autumn.",

            "topic-day-five-travel-sightseeing": "I like sightseeing.",
            "topic-day-five-travel-food": "I like eating.",

            "topic-olympics-select-user-would-compete-volleyball": "I like volleyball.",
            "topic-olympics-select-user-would-compete-tennis": "I like tennis.",
            "topic-olympics-select-user-would-compete-diving": "I like diving.",
            "topic-olympics-select-user-would-compete-archery": "I like archery.",

            "topic-olympics-user-height-below-seventytwo": "I am less tall than 72 inches.",
            "topic-olympics-user-height-above-seventytwo": "I am taller than 72 inches.",

            "topic-olympics-haru-height-below-twentyfour": "You are less tall than 24 inches.",
            "topic-olympics-haru-height-above-twentyfour": "You are taller than 24 inches.",
        }

        document = user_sentence
        queries = [mapping.get(label, label.replace('-', ' ')[-1]) for label in candidate_labels]

        return dict(document=document, queries=queries)

    
class BoolQA(Task):

    def generate_inputs(self, haru_sentence: str, user_sentence: str, candidate_labels: List[str], *args, **kwargs) -> str:
        """
        Generate query using label names (should be yes / no). Return a dict with `document` and `queries` as keys.
        """
        document = f'{haru_sentence} {user_sentence}'
        queries = [f'{haru_sentence} {label.capitalize()}' for label in candidate_labels]

        return dict(document=document, queries=queries)


class SentimentAnalysis(Task):

    def generate_inputs(self, user_sentence: str, candidate_labels: List[str], *args, **kwargs) -> str:
        """
        Generate query using label names (should be positive / negative / neutral). Return a dict with `document` and `queries` as keys.
        """
        mapping = {
            "positive": "This text expresses a positive sentiment.",
            "negative": "This text expresses a negative sentiment.",
            "neutral": "This text expresses a neutral sentiment."
        }

        document = user_sentence
        queries = [mapping[label] for label in candidate_labels]

        return dict(document=document, queries=queries)


class NamedEntityRecognition(Task):

    def generate_inputs(self, haru_sentence: str, user_sentence: str, entities_to_extract: List[str], *args, **kwargs) -> str:
        """
        Generate query using entity names. Return a dict with `document` and `queries` as keys.
        """
        mapping = {
            "name": "What is the name of the user?",
            "hometown": "What city is the user from?",
            "fav_continent": "What is the favorite continent of the user?",
            "next_travel": "What is the next travel country, city, or continent of the user?",
            "home_country": "What is the home country name of the user?",
            "family_name": "What is the family name of the user?",
            "name_origin": "What country or region does the user's family name come from?",
            "profession": "What is the user working as?",
            "fav_animal": "What is the favorite animal of the user?",
            "pet": "What kind of animal is the user's pet?",
            "parents_names": "What are the names of the mother and the father of the user?",
            "parents_professions": "What are the user's mother and the father working as?",
            "fav_food": "What is the name of the favorite food of the user?",
            "haru_fav_food": "What is the name of the favorite food of the robot?"
        }

        document = f'The robot asks the user this question: {haru_sentence} The user responds to that question as follows: {user_sentence}'
        queries = [mapping.get(ent, f"What is the {ent} of the user?") for ent in entities_to_extract]

        return dict(document=document, queries=queries)
