import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluator:
    # Constructor
    # Prediction vs. Real Value
    def __init__(self, prediction, truth):
        self.prediction = prediction
        self.truth = truth

    # Manual Evaluation
    def evaluate(self, neg=0):

        # Variables
        total = len(self.truth)
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        # Compare the predictions against the ground truth
        # Count the true/false positive/negative matches
        for prediction, truth in zip(self.prediction, self.truth):
            if prediction == truth and prediction == 1 - neg:
                TP += 1
            elif prediction == truth and prediction == neg:
                TN += 1
            elif prediction != truth and prediction == 1 - neg:
                FP += 1
            elif prediction != truth and prediction == neg:
                FN += 1
            else:
                raise ("error...%s-%s" % (prediction, truth))
        print("{} {} {} {}".format(TP,TN,FP,FN))

        # Calculate the metrics
        self.accuracy = (TP + TN) / 1.0 / total
        self.precision = TP / 1.0 / (TP + FP)
        self.recall = TP / 1.0 / (TP + FN)
        self.f1measure = 2 * self.precision * self.recall / 1.0 / (self.precision + self.recall)

        # Return the metrics
        return self.accuracy, self.precision, self.recall, self.f1measure  

    # Assisted evaluation with SKLearn
    def metrics(self):
        
        # Shape the prediction and truth for metrics
        self.prediction = self.prediction.reshape(-1)
        try:
            self.truth = self.truth.reshape(-1)
        except:
            pass
        
        # Calculate the metrics
        accuracy = accuracy_score(self.prediction, self.truth)
        precision = precision_score(self.prediction, self.truth)
        recall = recall_score(self.prediction, self.truth)
        f1measure = f1_score(self.prediction, self.truth)

        # Invert the values
        self.prediction = np.array([1] * len(self.prediction)) - self.prediction
        self.truth = np.array([1] * len(self.prediction)) - self.truth

        # Calculate the negative metrics
        accuracy_neg = accuracy_score(self.prediction, self.truth)
        precision_neg = precision_score(self.prediction, self.truth)
        recall_neg = recall_score(self.prediction, self.truth)
        f1measure_neg = f1_score(self.prediction, self.truth)

        # Calculate the average between normal and negative metrics
        accuracy = (accuracy + accuracy_neg) / 2
        precision = (precision + precision_neg) / 2
        recall = (recall + recall_neg) / 2
        f1measure = (f1measure + f1measure_neg) / 2

        # Return the metrics
        return accuracy, precision, recall, f1measure


if __name__ == '__main__':
    ep = Evaluator([1, 0, 1, 0], [1, 0, 1, 1])
    print(ep.metrics())
    print(ep.evaluate())
