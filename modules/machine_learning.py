

class MachineLearning():

    def set_classifier(self, cla):
        if cla == 'random_forrest':
            return self.RandomForrest()
        elif cla == 'xg_boost':
            return self.XGBoost()
        elif cla == 'perceptron':
            return self.Perceptron()
        elif cla == 'gradient_boost':
            return self.GradiendBoosting()

    def RandomForrest(self):
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(criterion='entropy',
                                     n_estimators=10,
                                     random_state=1,
                                     n_jobs=2)
        return rfc

    def XGBoost(self):
        import xgboost
        xgb = xgboost.XGBClassifier(max_depth=6, objective='binary:logistic')

        return xgb

    def Perceptron(self):
        from sklearn.linear_model import Perceptron
        return Perceptron(max_iter=40, eta0=0.1, random_state=0)

    def GradiendBoosting(self):
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(max_depth=40)