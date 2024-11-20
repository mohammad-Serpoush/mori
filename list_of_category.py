import json
with open("products.json" , "r") as file:
    data = json.load(file)
    
c = []
for d in data[:1000]:
    c.append(d["category_name"])
    
print(set(c))