from typing import List

class Task():

    def generate_document(self, *args, **kwargs) -> str:
        raise NotImplementedError
    
    def generate_candidates(self, *args, **kwargs) -> List[str]:
        raise NotImplementedError


class NamedEntityRecognition(Task):

    def generate_document(self, document: str, *args, **kwargs) -> str:
        return document

    def generate_candidates(self, entities: List[str], *args, **kwargs) -> List[str]:
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

        candidates = [mapping.get(ent, f"What is the {ent} of the user?") for ent in entities]

        return candidates
