import pickle
f = open("train_sort_v100_ssh_v2.pkl","rb")
model = pickle.load(f)
labels=[]
for y in model:
  x=y['class']
  if x not in labels:
     labels.append(x)
print(len(model))
