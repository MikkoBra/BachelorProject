from data.data_processing import *
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score
from initialize.generator import load_network
from numpy import mean, std


# TODO: implement kfold yourself
def kfold(nn_type):
    x, y = load_dataset('data/all_data_clean.pkl')
    data, one_hot = preprocess_clean_data(x, y)
    model = load_network(nn_type)
    estimator = KerasClassifier(build_fn=model, epochs=10)
    # prepare the cross-validation procedure
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(estimator, data, one_hot, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
