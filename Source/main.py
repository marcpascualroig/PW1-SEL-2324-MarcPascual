from rules import rules
from preprocessing import pre_processing


#filename = "ContactLens.csv"
#filename = 'ENB2012_data.xlsx'
#filename = "mushroom.csv"
#filename = 'Dry_Bean_Dataset.xlsx'


#split and pre process
X_train, X_test, y_train, y_test = pre_processing(file_name=filename)
print(X_train)
print(y_train)



#initialize rules
Rules = rules(X_train, y_train, X_test, y_test)
#fit rules with only relevant rules
Rules.rules_fit(X_train, y_train, extra_rules=False)
#predict instances sequentially
Rules.rules_predict(X_test, y_test, discard_instances=False, extra_rules=False)
#predict instances with voting
Rules.rules_predict(X_test, y_test, discard_instances=True, extra_rules=False)

#initialize rules
Rules = rules(X_train, y_train, X_test, y_test)
#fit rules with all rules
Rules.rules_fit(X_train, y_train, extra_rules=True)
#predict instances sequentially
Rules.rules_predict(X_test, y_test, discard_instances=False, extra_rules=True)
#predict instances with voting
Rules.rules_predict(X_test, y_test, discard_instances=True, extra_rules=True)
