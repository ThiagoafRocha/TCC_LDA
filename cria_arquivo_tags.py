#cria_arquivo_tags
import codecs
import json

datafile = json.load(codecs.open('data.json', 'r', 'utf-8-sig'))
str_final = ""
for item in datafile["items"]:
    str_final += str(item["tags"])

with open("tags.txt", "a") as file:
    file.write(str_final)