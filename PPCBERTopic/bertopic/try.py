from bertopic import BERTopic
topic_model = BERTopic.load("my_bertopic_model_HDBSCAN")
sentences = [
    "My appointment is next week.",
    "Doctor prescribed new medication."
]

topics, probs = topic_model.transform(sentences)

print("Topics:", topics)
print("Probs:", probs)
