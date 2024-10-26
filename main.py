import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
joseph = pd.read_csv('C:\\Users\\HomePC\\Desktop\\python\\data.csv')
print(joseph.head(5))
data = joseph.loc[:, ['Speed_of_Car', 'Age_of_Driver']]
X = data[['Speed_of_Car']]
y = data['Age_of_Driver']
regressor = LinearRegression()
regressor.fit(X, y)
plt.scatter(data['Speed_of_Car'], data['Age_of_Driver'], label='Data points') 
plt.plot(X, regressor.predict(X), color='red', linewidth=2, label='Regression Line')
plt.xlabel('Speed_of_Car') 
plt.ylabel('Age_of_Driver')
plt.title('Speed_of_Car vs Age_of_Driver')
plt.legend()
plt.show()
x=pd.DataFrame(data['Speed_of_Car'])
y=pd.DataFrame(data['Age_of_Driver'])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print("x_train = " , X_train.shape)
print("x_test = " , X_test.shape)
print("y_train = ", y_train.shape)
print("y_test = " , y_test.shape)
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print("regressor intercept = " , regressor.intercept_)
print("regressor co-eficient = " , regressor.coef_)
y_pred = regressor.predict(X_test)
y_pred
print(y_test)
y_test.head(5)
