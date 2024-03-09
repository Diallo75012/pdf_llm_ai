#!/usr/bin/python3

result = [{"content": "one","score": 1},{"content": "two","score": 2},{"content": "three","score": 3}]

score_list = []
for elem in result:
  score_list.append(elem["score"])
best_score_response = ''.join([elem["content"] for elem in result if elem["score"] == max(score_list)])

print("score_list", score_list)
print("score list max: ", max(score_list))
print("Best Score Response: ", best_score_response, type(best_score_response))
