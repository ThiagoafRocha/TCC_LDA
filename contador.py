from collections import Counter

with open("tags.txt", "r") as file:
    str_inicial = file.read();
    str_final = str_inicial.replace('][', '').replace('\'', ' ').replace(',', '').replace('[', ''). replace(']', '')
ocorrencias = Counter(str_final.split())
print(ocorrencias)

with open("test_contador.txt", "a") as file:
    file.write(str(ocorrencias))

list_tags = []
flag = 0
for item in ocorrencias:
    for i in range(0, len(list_tags)+1):
        if item == str_final[i]:
            flag = 1
            break
    if flag == 0:
        list_tags.append(item)
    flag = 0

#print(list_tags)