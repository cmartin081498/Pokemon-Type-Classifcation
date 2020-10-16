#Curtis Martin
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


if __name__ == '__main__':
  
    filename='Pokemon_data\full_pokemon_dict.json'
    with open(filename) as f_obj:
        pokemon=json.load(f_obj)
    #dict that will hold single pokemon types
    pokemon_types=['fire','water','rock','normal','psychic','grass','dark','bug','ghost','steel','ice','fairy','poison','electric','ground','dragon','fighting']

    #extract the pokemon moves and their types
    x_all_type=[]
    for poke in pokemon:
        x_all_type.append(poke['moves'])
    X_all_type=[]
    for types in x_all_type:
        y=[]
        for value in types.values():
            y.append(value)
        X_all_type.append(y)
    X_array_alltype=np.asarray(X_all_type)
    y_onetype=[]
    for poke in pokemon:
        types=[]
        pType = poke['type']
        for type in pokemon_types:
            if type in pType:
                types.append(type)
        y_onetype.append(types) 
    
    MLB=MultiLabelBinarizer()
    y=MLB.fit_transform(y_onetype)
    X_train, X_test, y_train, y_test = train_test_split(X_array_alltype, y, test_size=0.3)
    grid_params= {
    'n_neighbors':[3,5,11,13,15,17,21],
    'weights':['uniform','distance'],
    'metric':['euclidean','manhattan'],
    'algorithm':['ball_tree','kd_tree','brute']
    }

    knn_grid=GridSearchCV(KNeighborsClassifier(),grid_params,cv=5,scoring='f1_micro')
    knn_grid.fit(X_train,y_train)
    best_knn=knn_grid.best_estimator_
    cvres=knn_grid.cv_results_
    y_pred=best_knn.predict(X_test)
    for f1_sc, params in zip(cvres["mean_test_score"],cvres["params"]):
        print(f"f1_score: {f1_sc}")
        print(f"params: {params}")

    print('Best Estimator: ',knn_grid.best_estimator_)
    print('Micro Recall score: ',recall_score(y_test,y_pred,average='micro'))
    print('Micro F1 score: ',f1_score(y_test,y_pred,average='micro'))
    print('Hamming loss score: ',hamming_loss(y_test,y_pred))
    print('Hamming Score: ',hamming_score(y_test,y_pred))
