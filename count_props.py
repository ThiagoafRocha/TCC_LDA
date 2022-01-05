
#count_props.py

import codecs
import json

datafile = json.load(codecs.open('files/topico11.json', 'r', 'utf-8-sig'))
n = 0
vc = 0
fv = 0
sc = 0
ans = 0
for item in datafile["items"]:
    vc += item["view_count"]
    fv += item["favorite_count"]
    sc += item["score"]
    n = n+1
    if not item["is_answered"]:
        ans = ans + 1

media_view_count = vc/n
media_favoritos = fv/n
media_score = sc/n

porcent_ans = ans/n

print("A média de visualizações é: " + str(media_view_count))
print("A média de favoritos é: " + str(media_favoritos))
print("A média de score é: " + str(media_score))
print("Porcentagem de perguntas não respondidas: " + str(porcent_ans))
