# spacy_demo.py
# This script demonstrates basic usage of spaCy for Named Entity Recognition (NER).

import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Example sentence
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Print named entities
print("Named Entities, Phrases, and Concepts:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Process a text
doc = nlp("Martin is learning AI tools and spaCy to complete his assignment.")

# Print named entities, part of speech, and dependency parsing
for token in doc:
    print(f"{token.text:<12} {token.pos_:<10} {token.dep_:<10}")

# Print named entities
print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text:<12} {ent.label_}")
