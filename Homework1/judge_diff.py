import numpy as np
import csv


old_res = csv.reader(open('result.csv', 'r'))
new_res = csv.reader(open('result1.csv', 'r'))
# pp_res = csv.reader(open('result2.csv', 'r'))

old_r = []
new_r = []
pp_r = []

for line in old_res:
	old_r.append(line[1])

for line in new_res:
	new_r.append(line[1])

# for line in pp_res:
#	pp_r.append(line[1])

print(np.shape(new_r))

count = 0
one_count = 0
oone_count = 0
ooone_count = 0
for i in range(len(old_r)):
	if old_r[i] != new_r[i]:
		count += 1
		pp_r.append(i)
	if new_r[i] == '1':
		one_count += 1
	if old_r[i] == '1':
		oone_count += 1

print(len(pp_r))
rate = count / len(old_r)
print('Differ Rate : %f' % rate)
print('New One Count : %d' % one_count)
print('Old One Count : %d' % oone_count)