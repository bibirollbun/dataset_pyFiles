# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import random
import difflib

class EssayGenerator:
    def __init__(self, topics, max_attempts=10):
        self.topics = topics
        self.used_essays = set()
        self.max_attempts = max_attempts
        self.synonym_dict = {
            'traditional': ['conventional', 'standard', 'established', 'typical'],
            'complexity': ['intricacy', 'sophistication', 'depth', 'nuance'],
            'dynamic': ['fluid', 'adaptive', 'evolving', 'transformative'],
            'fundamental': ['core', 'essential', 'basic', 'critical'],
            'transformation': ['metamorphosis', 'evolution', 'change', 'shift']
        }
    
    def generate_essays(self, num_essays=3):
        essays = []
        topics_copy = self.topics.copy()
        
        while len(essays) < num_essays and topics_copy:
            topic = random.choice(topics_copy)
            topics_copy.remove(topic)
            
            essay = self._generate_unique_essay(topic, essays)
            if essay:
                essays.append(essay)
        
        return essays
    
    def _generate_unique_essay(self, topic, existing_essays):
        """Generate a unique essay with high variability"""
        templates = [
            lambda t: f"In the realm of {t}, uncertainty reigns supreme. Conventional boundaries dissolve, revealing intricate networks of complexity that challenge linear comprehension. Knowledge emerges as a dynamic, multifaceted process of continuous redefinition.",
            
            lambda t: f"Confronting {t} unravels layers of profound ambiguity. Each perspective becomes a prism, refracting multiple interpretations. Understanding demands embracing contradiction and navigating intellectual landscapes beyond established paradigms.",
            
            lambda t: f"The fundamental essence of {t} resists simplification. Paradoxical tensions emerge, where progress and limitation coexist. Insights bloom at the intersection of contradiction, challenging our most deeply held assumptions.",
            
            lambda t: f"Exploring {t} reveals a terrain of perpetual questioning. Boundaries become fluid, assumptions crumble. Knowledge transforms into a dynamic process where each insight generates new uncertainties."
        ]
        
        for _ in range(self.max_attempts):
            template = random.choice(templates)
            essay = self._constrain_essay(self._synonymize_essay(template(topic)))
            
            if self._is_sufficiently_unique(essay, existing_essays):
                return essay
        
        return None
    
    def _synonymize_essay(self, essay):
        """Replace words with synonyms to increase variability"""
        words = essay.split()
        synonymized_words = [
            random.choice(self.synonym_dict.get(word.lower(), [word])) 
            if random.random() < 0.3 else word
            for word in words
        ]
        return ' '.join(synonymized_words)
    
    def _is_sufficiently_unique(self, new_essay, existing_essays):
        """Ensure high uniqueness between essays"""
        for existing_essay in existing_essays:
            similarity = difflib.SequenceMatcher(None, new_essay, existing_essay).ratio()
            if similarity > 0.5:  # Stricter uniqueness check
                return False
        return True
    
    def _constrain_essay(self, essay, target_length=100):
        """Constrain essay to approximate target length"""
        words = essay.split()
        if len(words) > target_length:
            essay = ' '.join(words[:target_length])
        
        return essay.strip()

# Example usage
test_topics = [
    "The nature of consciousness",
    "Technology's impact on human perception", 
    "Ethical boundaries in artificial intelligence"
]

generator = EssayGenerator(test_topics)
essays = generator.generate_essays()
for i, essay in enumerate(essays, 1):
    print(f"Essay {i}: {essay}\n")




