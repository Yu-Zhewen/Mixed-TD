import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def get_accuracy(y_pred, y_true, threshold = 0.01):
    a = (y_true - y_pred) / y_true
    b = np.where(abs(a) <= threshold)
    return len(b[0]) / len(y_true)

def eval(y_pred, y_true):
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    acc5 = get_accuracy(y_pred, y_true, threshold=0.05)
    acc10 = get_accuracy(y_pred, y_true, threshold=0.10)
    acc15 = get_accuracy(y_pred, y_true, threshold=0.15)
    acc20 = get_accuracy(y_pred, y_true, threshold=0.20)
    acc25 = get_accuracy(y_pred, y_true, threshold=0.25)
    return rmse, rmspe, rmse / np.mean(y_true), acc5, acc10, acc15, acc20, acc25
        
def random_forest_regressor(X, Y, split_validation=False):
    predictor = RandomForestRegressor(
        max_depth = 50,
        n_estimators = 370,
        min_samples_leaf = 1,
        min_samples_split = 2,
        max_features = "auto",
        oob_score = True,
        random_state = 10,
    )

    if split_validation:
        trainx, testx, trainy, testy = train_test_split(X, Y, test_size = 0.2, random_state = 10)
        print(f"training data size: {len(trainx)}, test data size: {len(testx)}")
    else:
        trainx, trainy = X, Y
        testx, testy = X, Y
        print(f"training data size: {len(trainx)}, no test data")

    # start training
    predictor.fit(trainx, trainy)
    predicts = predictor.predict(testx)
    pred_error_list = [abs(y1 - y2) / y1 for y1, y2 in zip(testy, predicts)]
    rmse, rmspe, error, acc5, acc10, acc15, acc20, acc25 = eval(predicts, testy)
    print(f"rmse: {rmse:.4f}; rmspe: {rmspe:.4f}; error: {error:.4f}; 5% accuracy: {acc5:.4f}; 10% accuracy: {acc10:.4f}; 15% accuracy: {acc15:.4f}; 20% accuracy: {acc20:.4f}; 25% accuracy: {acc25:.4f}.")

    return predictor