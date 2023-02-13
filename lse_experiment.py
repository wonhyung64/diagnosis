#%%
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error

#%%

x1 = np.random.normal(2, 1.3, size = 10000)
x2 = x1 * 5 + np.random.normal(0, 1, size=10000)
x3 = x1 * 2 + np.random.normal(0, 0.7, size=10000)
x4 = x2 * 1.3 + x3 * 2.1 + np.random.normal(0, 0.3, size=10000)

np.random.seed(1234)
m_x2 = LinearRegression()
m_x2.fit(np.expand_dims(x1, -1), np.expand_dims(x2, -1))
m_x2.coef_

m_x3 = LinearRegression()
m_x3.fit(np.expand_dims(x1, -1), np.expand_dims(x3, -1))
m_x3.coef_

m_x4 = LinearRegression()
m_x4.fit(np.concatenate([np.expand_dims(x2, -1), np.expand_dims(x3, -1)], -1), np.expand_dims(x4, -1))
m_x4.coef_



m2_x2 = LinearRegression()
m2_x2.fit(np.concatenate([np.expand_dims(x1, -1), np.expand_dims(x3, -1)], -1), np.expand_dims(x2, -1))
m2_x2.coef_

m2_x3 = LinearRegression()
m2_x3.fit(np.expand_dims(x1, -1), np.expand_dims(x3, -1))
m2_x3.coef_

m2_x4 = LinearRegression()
m2_x4.fit(np.concatenate([np.expand_dims(x2, -1), np.expand_dims(x3, -1)], -1), np.expand_dims(x4, -1))
m2_x4.coef_


m3_x2 = LinearRegression()
m3_x2.fit(np.expand_dims(x1, -1), np.expand_dims(x2, -1))
m3_x2.coef_

m3_x3 = LinearRegression()
m3_x3.fit(np.concatenate([np.expand_dims(x1, -1), np.expand_dims(x2, -1)], -1), np.expand_dims(x3, -1))
m3_x3.coef_


m3_x4 = LinearRegression()
m3_x4.fit(np.concatenate([np.expand_dims(x2, -1), np.expand_dims(x3, -1)], -1), np.expand_dims(x4, -1))
m3_x4.coef_

#%%
mean_squared_error(x2, m_x2.predict(np.expand_dims(x1, -1)))+mean_squared_error(x3, m_x3.predict(np.expand_dims(x1, -1)))+mean_squared_error(x4, m_x4.predict(np.concatenate([np.expand_dims(x2, -1), np.expand_dims(x3, -1)], -1)))
mean_squared_error(x2, m2_x2.predict(np.concatenate([np.expand_dims(x1, -1), np.expand_dims(x3, -1)], -1)))+mean_squared_error(x3, m2_x3.predict(np.expand_dims(x1, -1)))+mean_squared_error(x4, m2_x4.predict(np.concatenate([np.expand_dims(x2, -1), np.expand_dims(x3, -1)], -1)))
mean_squared_error(x2, m3_x2.predict(np.expand_dims(x1, -1)))+mean_squared_error(x3, m3_x3.predict(np.concatenate([np.expand_dims(x1, -1), np.expand_dims(x2, -1)], -1)))+mean_squared_error(x4, m3_x4.predict(np.concatenate([np.expand_dims(x2, -1), np.expand_dims(x3, -1)], -1)))

#%%

m_x2 = Lasso(alpha=0.5)
m_x2.fit(np.expand_dims(x1, -1), np.expand_dims(x2, -1))
m_x2.coef_

m_x3 = Lasso(alpha=0.5)
m_x3.fit(np.expand_dims(x1, -1), np.expand_dims(x3, -1))
m_x3.coef_

m_x4 = Lasso(alpha=0.5)
m_x4.fit(np.concatenate([np.expand_dims(x2, -1), np.expand_dims(x3, -1)], -1), np.expand_dims(x4, -1))
m_x4.coef_



m2_x2 = Lasso(alpha=0.5)
m2_x2.fit(np.concatenate([np.expand_dims(x1, -1), np.expand_dims(x3, -1)], -1), np.expand_dims(x2, -1))
m2_x2.coef_

m2_x3 = Lasso(alpha=0.5)
m2_x3.fit(np.expand_dims(x1, -1), np.expand_dims(x3, -1))
m2_x3.coef_

m2_x4 = Lasso(alpha=0.5)
m2_x4.fit(np.concatenate([np.expand_dims(x2, -1), np.expand_dims(x3, -1)], -1), np.expand_dims(x4, -1))
m2_x4.coef_




m3_x2 = Lasso(alpha=0.5)
m3_x2.fit(np.expand_dims(x1, -1), np.expand_dims(x2, -1))
m3_x2.coef_

m3_x3 = Lasso(alpha=0.5)
m3_x3.fit(np.concatenate([np.expand_dims(x1, -1), np.expand_dims(x2, -1)], -1), np.expand_dims(x3, -1))
m3_x3.coef_


m3_x4 = Lasso(alpha=0.5)
m3_x4.fit(np.concatenate([np.expand_dims(x2, -1), np.expand_dims(x3, -1)], -1), np.expand_dims(x4, -1))
m3_x4.coef_

#%%
mean_squared_error(x2, m_x2.predict(np.expand_dims(x1, -1)))+mean_squared_error(x3, m_x3.predict(np.expand_dims(x1, -1)))+mean_squared_error(x4, m_x4.predict(np.concatenate([np.expand_dims(x2, -1), np.expand_dims(x3, -1)], -1)))
mean_squared_error(x2, m2_x2.predict(np.concatenate([np.expand_dims(x1, -1), np.expand_dims(x3, -1)], -1)))+mean_squared_error(x3, m2_x3.predict(np.expand_dims(x1, -1)))+mean_squared_error(x4, m2_x4.predict(np.concatenate([np.expand_dims(x2, -1), np.expand_dims(x3, -1)], -1)))
mean_squared_error(x2, m3_x2.predict(np.expand_dims(x1, -1)))+mean_squared_error(x3, m3_x3.predict(np.concatenate([np.expand_dims(x1, -1), np.expand_dims(x2, -1)], -1)))+mean_squared_error(x4, m3_x4.predict(np.concatenate([np.expand_dims(x2, -1), np.expand_dims(x3, -1)], -1)))















