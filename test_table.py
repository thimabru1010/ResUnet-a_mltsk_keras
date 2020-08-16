from prettytable import PrettyTable

table = PrettyTable()

table.title = "Metricas"

table.field_names = ["City name", "Area", "Population", "Annual Rainfall"]

table.add_row(["Adelaide", 1295, 1158259, 600.5])
table.add_row(["Brisbane", 5905, 1857594, 1146.4])

# print(table.title)
print(table)

dic = {'test': 15}

dic['key'] = 1
dic['pessoa'] = 10

print(dic)
