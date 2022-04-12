import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

veriseti = pd.read_csv("linear_model.csv")
veriseti.isnull().sum()
veriseti.y.fillna(value=veriseti.y.mean(), inplace=True)
veriseti.describe()
#normalizasyon MinMax Scaler
veriseti.x = (veriseti.x -veriseti.x.min()) /(veriseti.x.max() - veriseti.x.min())
veriseti.y = (veriseti.y -veriseti.y.min()) /(veriseti.y.max() - veriseti.y.min())

X = veriseti.x
Y = veriseti.y

plt.scatter(X.values, Y.values)
plt.xlabel("x")
plt.ylabel("y")
plt.title("X ve Y arasındaki ilişki")

#model kurma

lr = LinearRegression()
lr.fit(X.values.reshape(-1,1),Y.values.reshape(-1,1))
print(lr.coef_ , lr.intercept_)

print("Kurulan regresyon modeli Y = {} + {}*x".format(lr.intercept_[0].round(2), lr.coef_[0][0].round(2))) 

y_predicted = lr.predict(X.values.reshape(-1,1))
print(r2_score(Y,y_predicted))
df = pd.DataFrame({'y':Y.values.flatten(),'y_predict':y_predicted.flatten()})
print(df)
print(mean_absolute_error(Y, y_predicted))

b0 = lr.intercept_[0].round(2)
b1= lr.coef_[0][0].round(2)

random_x = np.array([0,1])
plt.plot(random_x, b0+b1*random_x, color='red', label='regresyon')
plt.legend()
