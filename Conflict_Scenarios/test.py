scores = open("scores.txt", "r", encoding='utf8')
data = scores.read().splitlines()
scores.close()

params = []

for item in data:
    temp = item.split(",")
    params.append(temp)

good = []
for parameter in params:
    coherence = float(parameter[0])
    if coherence >= 0.18:
        good.append(parameter)

extract = open("good_scores.txt", 'w', encoding='utf8')
for parang in good:
    temp = ""
    for item in parang:
        temp = temp + str(item) + ","
    temp = temp + "\n"
    extract.write(temp)
extract.close()

