import pandas as pd
import xgboost as xgb
import pickle

meal = pd.read_csv("preprocessing/meal_info.csv")
center = pd.read_csv("preprocessing/fulfilment_center_info.csv")
mapping = pickle.load(open("preprocessing/mapping","rb"))


def apply_cats(df, mapping):
    for column, map in mapping.items():
        for cats, vals in map.items():
            df.loc[df[column]==cats,column]=vals
        df[column]= df[column].astype('int16')
def preprocessing(df):
    df['discount'] = (df['base_price'].astype('int32') - df['checkout_price'].astype('int32'))*100/df['base_price'].astype('int32')
    df['meal_id']=df['meal_id'].astype('int64')
    df['center_id']=df['center_id'].astype('int64')
    
    testdf = df.merge(meal,on='meal_id')
    
    testdf = testdf.merge(center,on='center_id')
    apply_cats(testdf,mapping)
    testdf.astype('int32')
    return testdf
    


def get_predictions(df,bst):
    
    dfpd = preprocessing(df)
    df = xgb.DMatrix(dfpd.drop(columns=['id','week','base_price']).values)
    predictions = bst.predict(df)
    predictions = pd.DataFrame(predictions.round())
    print(predictions)
    return predictions

def predict_csv(file,bst):
    dfpd = pd.read_csv(file)
    dfpd=dfpd.astype('int64')
    predictions=get_predictions(dfpd,bst)
    predictions = predictions.rename(columns={0 : 'num_orders'})
    #predictions = predictions.rename({'unnamed : 0' : 'pred'})

    predictions = pd.concat([dfpd, predictions],axis=1)
    return predictions

def predict_individual(data,bst):
    df=pd.DataFrame(data,index=[0])

    preds=get_predictions(df,bst)
    return preds