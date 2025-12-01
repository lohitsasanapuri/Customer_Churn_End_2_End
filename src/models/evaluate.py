from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test,y_test):

    preds = model.predict(X_test)
    print(f" Classification Report : ", classification_report(y_test,preds))
    print(f" Confusion  Matrix : " ,confusion_matrix(y_test, preds))