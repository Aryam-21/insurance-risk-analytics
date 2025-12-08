from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Model:
    def __init__(self):
        self.lr_model = LinearRegression()
        self.dt_model = DecisionTreeRegressor(random_state=42)
        self.rfr_model = RandomForestRegressor(random_state=42)
        self.xgb_model = xgb.XGBRegressor(
            random_state=42,
            objective="reg:squarederror"
        )
    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    def train_model(self, x_train, y_train):
        self.lr_model.fit(x_train, y_train)
        self.dt_model.fit(x_train, y_train)
        self.rfr_model.fit(x_train, y_train)
        self.xgb_model.fit(x_train, y_train)
    def evaluate_model(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mae, mse, r2, y_pred