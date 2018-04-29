


with open("partial", "r") as f:
	lines = f.readlines()

i = 0
accuracy, loss, embed, num_filters, dropout, l1, l2 = 0,0,0,0,0,0,0

print("accuracy,loss,embed,num_filters,dropout,l1,l2")

for line in lines:
	line_type = i % 3
	line = line.split()

	if line_type == 0:
		accuracy, loss = line[4], line[6]
		
	elif line_type == 1:
		embed, num_filters, dropout, l1, l2 = line[2], line[4], line[6], line[8], line[10]

	else:
		print("{}{},{}{}{}{}{}".format(accuracy, loss, embed, num_filters, dropout, l1, l2))

	i += 1