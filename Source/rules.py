import pandas as pd
import numpy as np
import time
from collections import Counter

class rules:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.selectors = []
        self.rules = []
        self.classified_indices = np.zeros(len(X_train) + len(X_test), dtype=bool)
        self.class_train_size = {value: sum(1 for item in y_train if item == value) for value in set(y_train)}
        self.class_test_size = {value: sum(1 for item in y_test if item == value) for value in set(y_test)}
        self.class_name = y_train.name
        self.num_feat = X_train.shape[1]

    def _condition(self, combinations):
        features = [comb[0] for comb in combinations]
        return len(set(features)) == len(features)

    def _combinations(self, iterable, r, condition = None):#returns combinations given the selectors: only combinations with selectors of different features
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = list(range(r))
        if condition is None:
            yield tuple(pool[i] for i in indices)
        else:
            if condition(tuple(pool[i] for i in indices)):
                yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1
            if condition is None:
                yield tuple(pool[i] for i in indices)
            else:
                if condition(tuple(pool[i] for i in indices)):
                    yield tuple(pool[i] for i in indices)

    def _get_selectors(self, X, indices=None):#get selectors of remaining indices
        if indices is None:
            indices = X.index
        self.selectors = np.array([[feature, value] for feature in X.columns for value in
                                    X.iloc[indices][feature].unique()])

    def _validCondition(self, X, y, comb):#return the class and sample indices if the rule is 100% precise
        mask = pd.Series(True, index=X.index)
        for feature, value in comb:
            mask &= (X[feature].astype(str) == str(value))
        filtered_y = y[mask.index[mask]]
        if len(set(filtered_y)) == 1:
            return filtered_y.iloc[0], mask.index[mask],
        else:
            return None, None


    def rules_fit(self, X, y, extra_rules=False):
        print("Creating rules...")
        start_time = time.time()
        train_metrics=[]
        possible_rules =0
        num_combinations=1
        end = False
        instances = range(len(X))
        #while num_combinations <= 3 and not end:
        while num_combinations <= self.num_feat and not end:
            print('Combinations in rules:', num_combinations)
            self._get_selectors(X, list(set(instances) - set(np.where(self.classified_indices)[0])))
            combinations = self._combinations(self.selectors, num_combinations, self._condition)
            for comb in combinations:
                possible_rules +=1
                #check if condition is valid
                label, classified_instances = self._validCondition(X, y, comb)

                if label is not None:
                    new_classified = len(set(classified_instances)-set(np.where(self.classified_indices)[0]))
                    if new_classified > 0 or extra_rules:
                        indices_to_update = np.ravel(np.array([index for index in classified_instances if not self.classified_indices[index]]))
                        if len(indices_to_update) > 0:
                            self.classified_indices[indices_to_update] = True

                        self.rules.append((comb,
                                           label))

                        train_metrics.append((1,
                                            1,
                                            len(classified_instances)/len(instances),
                                            new_classified/len(instances),
                                            len(classified_instances)/self.class_train_size.get(label, 0),
                                            new_classified / self.class_train_size.get(label, 0),
                                            new_classified))

                        if len(set(np.where(self.classified_indices)[0])) == len(X):
                            end = True
                        #self._get_selectors(X, list(set(instances) - set(np.where(self.classified_indices)[0])))
            print('Number of instances classified:', len(set(np.where(self.classified_indices)[0])))
            num_combinations += 1

        unlabeled_percentage = (len(X)-len(set(np.where(self.classified_indices)[0])))/len(X)
        fit_time=time.time() - start_time
        if extra_rules:
            dataset = "train_extra_rules"
        else:
            dataset = 'train'
        self._write_results(train_metrics, dataset, 1, unlabeled_percentage, fit_time, len(self.rules), possible_rules)
        print('Number of rules = {}'.format(len(self.rules)))
        print('Number of rules evaluated = {}'.format(possible_rules))
        print("Training finished")


    def _predict_instances(self, X, y, rule, discard_instances=True):
        class_pred = rule[1]
        mask = pd.Series(True, index=X.index)
        for feature, value in rule[0]:
            mask &= (X[feature].astype(str) == str(value))
        mask_aux=mask.copy()

        filtered_y = y[mask.index[mask]]
        precision = (filtered_y == class_pred).sum() / len(filtered_y) if len(filtered_y) != 0 else 0
        coverage = len(filtered_y)/len(X)
        recall = (filtered_y == class_pred).sum() / self.class_test_size.get(class_pred, 0) if self.class_test_size.get(class_pred, 0) != 0 else 0

        mask &= pd.Series([False if (idx not in mask.index) or (self.preds[position] is not None) else True for position, idx in enumerate(mask.index)], index=mask.index)
        filtered_y = y[mask.index[mask]]
        precision_2 = (filtered_y == class_pred).sum() / len(filtered_y) if len(filtered_y) != 0 else 0
        coverage_2 = len(filtered_y) / len(X)
        recall_2 = (filtered_y == class_pred).sum() / self.class_test_size.get(class_pred,0) if self.class_test_size.get(class_pred, 0) != 0 else 0

        for pos, idx in enumerate(mask_aux):
            if idx:
                if self.preds[pos] is None:
                    self.preds[pos] = [class_pred]
                elif self.preds[pos] is not None and discard_instances:
                    self.preds[pos].append(class_pred)


        return precision, precision_2, coverage, coverage_2, recall, recall_2, len(filtered_y)



    def rules_predict(self, X, y, discard_instances = False, extra_rules = False):
        self.preds = [None] * (len(X))
        print("Predicting...")
        start_time = time.time()
        test_metrics = []
        for rule in self.rules:
            precision, precision_2, cov, cov_2, recall, recall_2, instances = self._predict_instances(X, y, rule, discard_instances)
            test_metrics.append((precision,
                                precision_2,
                                cov,
                                cov_2,
                                recall,
                                recall_2,
                                instances))

        if discard_instances:
            self.preds = [Counter(label).most_common(2) if label else None for label in self.preds]
            self.preds = [most_common[0][0] if most_common is not None and (
                        len(most_common) == 1 or most_common[0][1] > most_common[1][1]) else None
                          for most_common in self.preds]


        else:
            self.preds = [label[0] if label is not None else None for label in self.preds]
        correct_pred = sum(1 for pred, true in zip(self.preds, y) if pred == true)
        total_pred = sum(1 for pred in self.preds if pred is not None and pred != "unlabeled")
        accuracy = correct_pred / total_pred if total_pred != 0 else 0
        unclassified_percentage = (len(X) - total_pred)/len(X)
        print('correct pred', correct_pred)
        print('total:', total_pred)


        predict_time = time.time() - start_time
        if discard_instances and extra_rules:
            dataset = "test_voting_extra_rules"
        elif discard_instances:
            dataset = "test_voting"
        elif extra_rules:
            dataset = "test_extra_rules"
        else:
            dataset = "test"
        self._write_results(test_metrics, dataset, accuracy, unclassified_percentage, predict_time, 0, 0)
        print("Prediction finished. Accuracy = {}.  unclassified percentage = {}".format(accuracy, unclassified_percentage))


    def _write_results(self, metrics, dataset, accuracy, left, time, num_rules, rules_seen):
        with open(f"rules_{dataset}.txt", 'w') as file:
            for i, rule in enumerate(self.rules):
                file.write("RULE {}: IF ".format(i+1))
                for j, condition in enumerate(rule[0]):
                    if j > 0:
                        file.write("AND ")
                    file.write("{}={} ".format(condition[0], condition[1]))
                file.write("THEN {}={}\n".format(self.class_name, rule[1]))
                precision_percentage = metrics[i][0] * 100
                precision_percentage_2 = metrics[i][1] * 100
                coverage_percentage = metrics[i][2] * 100
                coverage_percentage_2 = metrics[i][3] * 100
                recall_percentage = metrics[i][4] * 100
                recall_percentage_2 = metrics[i][5] * 100
                file.write("Precision (all): {:.2f}%\n".format(precision_percentage))
                file.write("Precision (unclassified): {:.2f}%\n".format(precision_percentage_2))
                file.write("Coverage (all): {:.2f}%\n".format(coverage_percentage))
                file.write("Coverage (unclassified): {:.2f}%\n".format(coverage_percentage_2))
                file.write("Recall (all): {:.2f}%\n".format(recall_percentage))
                file.write("Recall (unclassified): {:.2f}%\n".format(recall_percentage_2))
                file.write("Instances classified: {}\n".format(metrics[i][6]))
                file.write("\n")
            file.write('Accuracy: {:.2f}%\n'.format(accuracy*100))
            file.write('Instances not classified: {:.2f}%\n'.format(left*100))
            file.write('Time: {}\n'.format(time))
            file.write('Number of rules: {}\n'.format(num_rules))
            file.write('Rules seen: {}\n'.format(rules_seen))








