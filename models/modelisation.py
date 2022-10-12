from sklearn.linear_model import Ridge,Lasso,LassoLars,ElasticNet
import numpy as np
import pandas as pd
import math

def traductModel(model):
    if type(model) != type(''):
        return model
    return Ridge if model=='Ridge' else Lasso if model == 'Lasso' else LassoLars if model == 'LassoLars' else Ridge

def getCoefs():
    """Create an array of churn rates to be estimated"""
    l = []
    a = 0.3
    while(a<100):
        l.append(a)
        a = int(  (a*1.15+0.5)*10)/10
    return l
coefs = getCoefs()

def predictors(coefs, size):
    """Create the predictors by 'churning' the population """
    X = pd.DataFrame()
    for coef in getCoefs():
        predictor = [1]
        for i in range(size-1):
            predictor.append(predictor[-1]*(1-(coef/100)))
        X['c'+str(coef)+'%'] = predictor
    return X

def getAlphas():
    """Create an array of alpha coef to be tested"""
    alphas = []
    a = 0.000001
    while(a<1):
        alphas.append(a)
        a = a*1.2
    return alphas
alphas = getAlphas()

def _optimalModel(X,y,modelType):
    """Search for the minimum alpha coef where all ponderations are positive (a negative ponderation is interpreted as a surfit) """
    a = 0
    b = len(alphas)-1
    m = (a+b)//2
    model = None
    i=0
    while b-a > 1 :
        i=i+1
        alpha = alphas[m]
        model = modelType(alpha=alpha,fit_intercept= False, normalize=True, max_iter=int(1e6))
        model.fit(X,y)
        modelok = not len([a for a in model.coef_ if a < 0])
        if modelok: #on s'assure que tous les poids sont positifs, sinon le model est refusé
            b = m
        else :
            a = m
        m = (a+b)//2
        
    alpha = alphas[b]
    model = modelType(alpha=alpha,fit_intercept= False, normalize=True, max_iter=int(1e6))
    model.fit(X,y)
    return model

def optimalModel(X,y,modelType):
    """ Test Ridge model. if it fails (surfit), test Lasso model"""
    if modelType =='std':
        try:
            print('Ridge')
            return _optimalModel(X,y,Ridge)
        except:
            print("exception avec Ridge, je passe à Lasso")
            return _optimalModel(X,y,Lasso)
    elif type(modelType)==str :
        if modelType.lower() == 'lasso':
            print('choix user : lasso')
            return _optimalModel(X,y,Lasso)
        elif modelType.lower() == 'ridge':
            return _optimalModel(X,y,Ridge)
    else: 
        return _optimalModel(X,y,modelType)
    
def modeliseEtExtrapole(y, pointsSortie, modelType='std'):
    model = optimalModel(predictors(coefs, len(y)), y, modelType)
    if not model:
        return model
    s_predicteursnoms = ','.join([ (''+str(c)+'%') for c in coefs ])
    out = model.predict(predictors(coefs, pointsSortie))
    return out
    #s_out = ','.join([str(  int(s*1000)/1000   ) for s in out])
    #return '['+s_out +"]|["+ s_predicteursnoms +"]|["+ ','.join([str( int(s*1000)/1000) for s in model.coef_])+']'

def anomalieFixed(y, modelType, start):
    modelType = traductModel(modelType)
    ytemp = y[:start]
    print(ytemp)
    model = optimalModel(predictors(coefs, len(ytemp)), ytemp, modelType)
    modelout = model.predict(predictors(coefs, len(y)))
    return list(np.subtract(np.array(y),np.array(modelout)))
   

def anomalieRollingAbs(y, modelType, start):
    modelType = traductModel(modelType)
    out = [0 for i in y[:start]]
    for i,j in enumerate(y[start:]):
        try:
            ytemp = y[:start+i]
            model = optimalModel(predictors(coefs, len(ytemp)), ytemp, modelType)
            modelout = model.predict(predictors(coefs, len(ytemp)+1))
            out.append(y[start+i] - modelout[-1])
        except:
            out.append("error") 
    return out

def anomalieRollingAbs2(y, modelType, start):
    modelType = traductModel(modelType)
    out = [0 for i in y[:start]]
    for i,j in enumerate(y[start:]):
        try:
            ytemp = y[:start+i]
            model = optimalModel(predictors(coefs, len(ytemp)), ytemp, modelType)
            modelout = model.predict(predictors(coefs, len(ytemp)+1))
            correction = y[start+i-1] - modelout[-2]
            out.append(y[start+i] - (modelout[-1] + correction))
        except:
            out.append("error") 
    return out

def anomalieRollingPourcent2(y, modelType, start):
    modelType = traductModel(modelType)
    out = [0 for i in y[:start]]
    for i,j in enumerate(y[start:]):
        try:
            ytemp = y[:start+i]
            model = optimalModel(predictors(coefs, len(ytemp)), ytemp, modelType)
            modelout = model.predict(predictors(coefs, len(ytemp)+1))
            correction = y[start+i-1] - modelout[-2]
            anomalieAbs = y[start+i] - (modelout[-1] + correction)
            out.append(anomalieAbs/y[start+i])
        except:
            out.append("error") 
    return out
