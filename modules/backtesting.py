
import numpy as np
import pandas as pd

class BackTesting:

    def __init__(self, clf, data):
        self.set_classifier(clf)
        self.set_data(data)

    def set_classifier(self, clf):
        self.classifier = clf

    def set_data(self, data):
        self.data = data
        self.nr_rows, self.nr_cols = data.shape
        self.xvar = [x for x in np.unique(data.columns) if x.startswith('x_')]
        print(self.xvar)

    def set_input_variables(self, xvar):
        self.xvar = xvar

    def set_output_variable(self, yvar):
        self.yvar = yvar

    def get_classifier(self):
        return self.classifier

    def get_xvar(self):
        return self.xvar

    def initiate_result(self):
        d = {'prediction': np.nan}
        index = self.data.index
        self.result = pd.Series(data=d, index=index)

    def get_result(self):
        return self.result

    def run_backtest(self, start_row = 0):
        self.initiate_result()
        for pred_row in np.arange(start_row+1, self.nr_rows):
            datetimeindex = self.data.index.values[pred_row]
            print('Days left:', self.nr_rows - pred_row)
            X_train = self.data[self.xvar].iloc[start_row:pred_row, :].values
            Y_train = self.data[self.yvar].iloc[start_row:pred_row].values
            X_test = self.data[self.xvar].iloc[pred_row, :].values
            Y_test = self.data[self.yvar].iloc[pred_row]
            self.classifier.fit(X_train, Y_train)
            prediction = self.classifier.predict_proba([X_test])
            self.result.set_value(datetimeindex, 1-prediction[0][0])
        print(X_train)
        #self.result['prediction_bool'] = self.result['prediction'].apply(self.pred_to_bin)
        #self.check_accuracy()

    def feature_importance(x_var, figsize):
        # Plot feature importance
        f, ax = plt.subplots(1, 1, figsize=figsize)
        feature_importance = xgb.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)

        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(x_var)[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title(stock + ' | Feature Importance')


    def check_accuracy(self):
        for idx, row in self.result.iterrows():
            if (row['prediction_bool'] == row['actual']) and (row['prediction_bool']==1 or row['prediction_bool']==-1):
                self.result.set_value(idx, 'accuracy', 1)
            elif (row['prediction_bool']==0):
                self.result.set_value(idx, 'accuracy', 0)
            elif (row['prediction_bool'] != row['actual']) and (row['prediction_bool']==1 or row['prediction_bool']==-1):  # Fail
                self.result.set_value(idx, 'accuracy', -1)
            else:
                self.result.set_value(idx, 'accuracy', np.nan)


class Helper:

    def __init__(self):
        print('called for help')


    def set_thresholds(self, lower, upper):
        self.lower_threshold = lower
        self.upper_threshold = upper

    def set_prediction(pred):
        if pred >= 0.7:  # Long
            prediction = 1
        elif pred < 0.3:  # Short
            prediction = -1
        else:  # Stay
            prediction = 0
        return prediction