

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Reading data

train_data = pd.read_csv('train.csv')
train_data.columns


test_data = pd.read_csv('test.csv')
test_data.columns

data = pd.concat([train_data, test_data])
data.shape


y = train_data['label']
train_data.drop(columns='label', inplace=True)


def show_examples(data, y):
    plt.figure(figsize=(10,10))

    for i in range(16):
        plt.subplot(5, 4, i + 1)
        select = np.random.randint(data.shape[0])
        plt.imshow(data.values[select, :].reshape(28, 28), interpolation='nearest')
        plt.title('Label:' + str(y.values[select]))
        plt.axis('off')
    
    plt.show()

show_examples(train_data, y)

#Preparing data

def scale_data(data):
    print(data.max())
    data.max()
    scaled = data / data.max()
    return scaled

train_data_scaled = scale_data(train_data)
train_data_scaled.fillna(0, inplace=True)

assert np.max(np.max(train_data_scaled)) == 1., 'You got something wrong!'


#Modling data

def multinominalNB_model(X, y):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import cross_val_score

    model = MultinomialNB()
    model.fit(X, y)
    Y = model.predict(X)
    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
    return scores.mean(), Y




scores_1, y_predict_1 = multinominalNB_model(train_data, y)

scores_1_scaled, y_predict_1_scale = multinominalNB_model(train_data_scaled, y)


print('Originl data = {0}, Scaled data = {1}'.format(scores_1, scores_1_scaled))

if (scores_1 > scores_1_scaled):
    print('Original data gives better result!')
elif (scores_1 < scores_1_scaled):
    print('Scaled data gives better result!')
else:
    print('They are the same!')
    

def logistic_regression_model(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    model = LogisticRegression(C=50. / 5000 ,multi_class='multinomial', penalty='l1', solver='saga', tol=0.1)
    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
    model.fit(X, y)
    Y = model.predict(X)
    return scores, Y

scores_2, y_predict_2 = logistic_regression_model(train_data, y)


def out_confusion_matrix(y, y_predict):
    from sklearn.metrics import confusion_matrix

    mat = confusion_matrix(y, y_predict)

    plt.figure(figsize=(10,10))
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('True label')
    plt.ylabel('Predicated label')


out_confusion_matrix(y, y_predict_1)
out_confusion_matrix(y, y_predict_2)

#Submitting 

scores ,y_predict = multinominalNB_model(train_data, y)

test_data['Label'] = y_predict
test_data.index.name = 'ImageId'
test_data.index = test_data.index + 1
test_data['Label'].to_csv('digits_submission.csv', header=True)