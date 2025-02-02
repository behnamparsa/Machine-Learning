import pandas as pd

info = [
    {'has_children':False, 'has_car':False,  'has_pet':True, 'age':'0-25yo'},
    {'has_children':False, 'has_car':False,  'has_pet':False,  'age':'0-25yo'},
    {'has_children':False, 'has_car':False,  'has_pet':True,   'age':'0-25yo'},
    {'has_children':False, 'has_car':True,   'has_pet':False,  'age':'0-25yo'},
    {'has_children':False, 'has_car':False,  'has_pet':True,   'age':'0-25yo'},
    {'has_children':False, 'has_car':False,  'has_pet':False,   'age':'0-25yo'},
    {'has_children':False, 'has_car':False,  'has_pet':True,  'age':'0-25yo'},
    {'has_children':True,  'has_car':False,  'has_pet':False,   'age':'0-25yo'},

    {'has_children':True,  'has_car':True,  'has_pet':True,  'age':'25-50yo'},
    {'has_children':False, 'has_car':True,  'has_pet':False,   'age':'25-50yo'},
    {'has_children':True,  'has_car':True,  'has_pet':True,  'age':'25-50yo'},
    {'has_children':False, 'has_car':False, 'has_pet':False,   'age':'25-50yo'},
    {'has_children':True,  'has_car':True,  'has_pet':True, 'age':'25-50yo'},
    {'has_children':False, 'has_car':True, 'has_pet':False,   'age':'25-50yo'},
    {'has_children':True,  'has_car':True,  'has_pet':True, 'age':'25-50yo'},
    {'has_children':False, 'has_car':True, 'has_pet':False,  'age':'25-50yo'},
    
    {'has_children':True,  'has_car':False, 'has_pet':True, 'age':'50+'},
    {'has_children':True,  'has_car':False,  'has_pet':False, 'age':'50+'},
    {'has_children':True,  'has_car':True,  'has_pet':True, 'age':'50+'},
    {'has_children':False, 'has_car':False, 'has_pet':False,  'age':'50+'},
]

info = pd.DataFrame(info)


def prior_target(df, col, value):
    return (df[col] == value).mean()

#print(prior_target(info,'age','0-25yo'))
#print(prior_target(info,'age','25-50yo'))
#print(prior_target(info,'age','50+'))

def cond_probability(df, filter_col, filter_val,col,value):
    #return (df[col].loc[df[filter_col] == filter_val] == value).mean()
    return (prior_target(df.loc[df[filter_col] == filter_val], col, value))

print(cond_probability(info, 'age', '25-50yo','has_car',True))

obs = {'has_children':False, 'has_car':False, 'has_pet':True}
target_val = '50+'

condProbs = []
priorFeatures = []

for k, v in obs.items():
    print(k,v)
    condProbs.append(cond_probability(info,'age',target_val,k,v))
    priorFeatures.append(prior_target(info, k, v))
print((pd.Series(condProbs).prod() * prior_target(info, 'age',target_val)) / pd.Series(priorFeatures).prod())


result = []

for target_val in info['age'].unique():
    condProbs = []
    priorFeatures = []
    for k, v in obs.items():
        condProbs.append(cond_probability(info,'age', target_val, k, v))
        priorFeatures.append(prior_target(info, k, v))
    result.append({'prob':(pd.Series(condProbs).prod() * prior_target(info, 'age',target_val)) / pd.Series(priorFeatures).prod(), 'class': target_val})

print('predicted Class:', pd.DataFrame(result).set_index('class')['prob'].idxmax())

pd.DataFrame(result)['prob'].idxmax()
#print((pd.DataFrame(result)).set_index('class')['prob'].idxmax())

#print(info['age'].unique())


from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
obs_df = pd.DataFrame([obs])[['has_children','has_car','has_pet']]
model.fit(info.drop(['age'], axis=1), info['age'])
pred = model.predict(obs_df)
print(pred)
