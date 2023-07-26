import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import linear_model, preprocessing
#import pickle


data = pd.read_csv('car.data')

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

predict = 'class'

x = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(cls)

x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model= KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

""" You can save the trained model by running this part  (do not forget to import pickle)
savefile = 'saved_model.sav'
pickle.dump(model, open(savefile, 'wb'))
"""

""" Also you can load pretrained models with this part
savefile = 'saved_model.sav'
load_model = pickle.load(open(savefile, 'rb'))
acc2 = load_model.score(x_test, y_test)
print(acc2) #95% acc
"""