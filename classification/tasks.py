from typing import List

class Task():

    def generate_document(self, *args, **kwargs) -> str:
        raise NotImplementedError
    
    def generate_candidates(self, *args, **kwargs) -> List[str]:
        raise NotImplementedError


class IntentRecognition(Task):

    def generate_document(self, document: str, *args, **kwargs) -> str:
        return document

    def generate_candidates(self, labels: List[str], *args, **kwargs) -> List[str]:
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

        }

        candidates = [mapping[label] for label in labels]

        return candidates

    
class BoolQA(Task):

    def generate_document(self, question: str, document: str, *args, **kwargs) -> str:
        return f'{question} {document}'

    def generate_candidates(self, question: str, labels: List[str], *args, **kwargs) -> List[str]:

        candidates = [f'{question} {label.capitalize()}' for label in labels]

        return candidates



class SentimentAnalysis(Task):

    def generate_document(self, document: str, *args, **kwargs) -> str:
        return document

    def generate_candidates(self, labels: List[str], *args, **kwargs) -> List[str]:
        mapping = {
            "positive": "This text expresses a positive sentiment.",
            "negative": "This text expresses a negative sentiment.",
            "neutral": "This text expresses a neutral sentiment."
        }

        candidates = [mapping[label] for label in labels]

        return candidates
