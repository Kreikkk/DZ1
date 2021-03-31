import numpy as np


def get_labels(y_predict):
    labels = y_predict[:, 0] < y_predict[:, 1]
    return labels

def crop_dataset(y_true, y_predict, percent=None):
    if percent is None:
        percent = 100
    N = int(len(y_true)*percent/100)
    y_true, y_predict = y_true[:N], y_predict[:N, :]
    
    return y_true, y_predict

def get_TP(y_true, labels):
    c = 0
    for y_true, y_predict in zip(y_true, labels):
        if y_true:
            if y_true == y_predict:
                c += 1
                
    return c

def get_TN(y_true, labels):
    c = 0
    for y_true, y_predict in zip(y_true, labels):
        if not y_true:
            if y_true == y_predict:
                c += 1
                
    return c

def get_FP(y_true, labels):
    c = 0
    for y_true, y_predict in zip(y_true, labels):
        if not y_true:
            if y_true != y_predict:
                c += 1
                
    return c

def get_FN(y_true, labels):
    c = 0
    for y_true, y_predict in zip(y_true, labels):
        if y_true:
            if y_true != y_predict:
                c += 1
                
    return c

def accuracy_score(y_true, y_predict, percent=None):
    y_true, y_predict = crop_dataset(y_true, y_predict, percent)
    
    labels = get_labels(y_predict)
    
    TP = get_TP(y_true, labels)
    TN = get_TN(y_true, labels)
    FP = get_FP(y_true, labels)
    FN = get_FN(y_true, labels)
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    
    return accuracy

def precision_score(y_true, y_predict, percent=None):
    y_true, y_predict = crop_dataset(y_true, y_predict, percent)
    
    labels = get_labels(y_predict)
    
    TP = get_TP(y_true, labels)
    FP = get_FP(y_true, labels)
    precision = TP/(TP+FP)
    
    return precision

def recall_score(y_true, y_predict, percent=None):
    y_true, y_predict = crop_dataset(y_true, y_predict, percent)
    
    labels = get_labels(y_predict)
    
    TP = get_TP(y_true, labels)
    FN = get_FN(y_true, labels)
    recall = TP/(TP+FN)
    
    return recall

def lift_score(y_true, y_predict, percent=None):
    y_true, y_predict = crop_dataset(y_true, y_predict, percent)
    
    labels = get_labels(y_predict)
    
    TP = get_TP(y_true, labels)
    TN = get_TN(y_true, labels)
    FP = get_FP(y_true, labels)
    FN = get_FN(y_true, labels)
    lift = (TP/(TP+FP))/((TP+FN)/(TP+TN+FP+FN))
    
    return lift

def f1_score(y_true, y_predict, percent=None):
    pre, rec = precision_score(y_true, y_predict, percent), recall_score(y_true, y_predict, percent)
    f1 = 2*pre*rec/(pre+rec)
    
    return f1

if __name__ == "__main__":
    file = np.loadtxt('HW2_labels.txt',  delimiter=',')
    y_predict, y_true = file[:, :2], file[:, -1]

    percent = 100
    
    print(f"Accuracy:\t{round(accuracy_score(y_true, y_predict, percent), 3)}")
    print(f"Precision:\t{round(precision_score(y_true, y_predict, percent), 3)}")
    print(f"Recall:\t\t{round(recall_score(y_true, y_predict, percent), 3)}")
    print(f"Lift:\t\t{round(lift_score(y_true, y_predict, percent), 3)}")
    print(f"F1:\t\t{round(f1_score(y_true, y_predict, percent), 3)}")

