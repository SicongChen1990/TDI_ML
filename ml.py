
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')
import matplotlib
import seaborn as sns
matplotlib.rcParams['savefig.dpi'] = 144


# In[2]:


import grader


# # ML: Predicting Star Ratings
# 
# Our objective is to predict a new venue's popularity from information available when the venue opens.  We will do this by machine learning from a dataset of venue popularities provided by Yelp.  The dataset contains meta data about the venue (where it is located, the type of food served, etc.).  It also contains a star rating. Note that the venues are not limited to restaurants. This tutorial will walk you through one way to build a machine-learning algorithm.
# 
# ## Metric
# 
# Your model will be assessed based on the root mean squared error of the number of stars you predict.  There is a reference solution (which should not be too hard to beat).  The reference solution has a score of 1. Keeping this in mind...
# 
# ## A note on scoring
# It **is** possible to score >1 on these questions. This indicates that you've beaten our reference model - we compare our model's score on a test set to your score on a test set. See how high you can go!
# 
# 
# ## Download and parse the incoming data
# 
# We start by downloading the dataset from Amazon S3:

# In[3]:


get_ipython().system(u"aws s3 sync s3://dataincubator-course/mldata/ . --exclude '*' --include 'yelp_train_academic_dataset_business.json.gz'")


# The training data are a series of JSON objects, in a gzipped file. Python supports gzipped files natively: [gzip.open](https://docs.python.org/2/library/gzip.html) has the same interface as `open`, but handles `.gz` files automatically.
# 
# The built-in json package has a `loads()` function that converts a JSON string into a Python dictionary.  We could call that once for each row of the file. [ujson](http://docs.micropython.org/en/latest/library/ujson.html) has the same interface as the built-in `json` library, but is *substantially* faster (at the cost of non-robust handling of malformed json).  We will use that inside a list comprehension to get a list of dictionaries:

# In[4]:


import ujson as json
import gzip

with gzip.open('yelp_train_academic_dataset_business.json.gz') as f:
    data = [json.loads(line) for line in f]


# In Scikit Learn, the labels to be predicted, in this case, the stars, are always kept in a separate data structure than the features.  Let's get in this habit now, by creating a separate list of the ratings:

# In[5]:


star_ratings = [row['stars'] for row in data]


# In[7]:


type(data)


# In[23]:


len(data)


# In[22]:


len(star_ratings)


# ### Notes:
# 1. [Pandas](http://pandas.pydata.org/) is able to read JSON text directly.  Use the `read_json()` function with the `lines=True` keyword argument.  While the rest of this notebook will assume you are using a list of dictionaries, you can complete it with dataframes, if you so desire.  Some of the example code will need to be modified in this case.
# 
# 2. There are obvious miscodings in the data.  There is no need to try to correct them.
# 
# ## Building models
# 
# For many of the questions below, you will need to build and train an estimator that predicts the star rating given certain features.  This could be a custom estimator that you built from scratch, but in most cases will be a pipeline containing custom or prebuilt transformers and an existing estimator.  We will give you hints of how to proceed, but the only requirement for you is to produce a model that does as well, or better, than the reference models we created.  You are welcome to do this however you like. The details are up to you.
# 
# The formats of the input and output to the `fit()` and `predict()` methods are ultimately up to you as well, but we recommend that you deal with lists or arrays, for consistency with the rest of Scikit Learn.  It is also a good idea to take the same type of data for the feature matrix in both `fit()` and `predict()`.  While it is tempting to read the stars from the feature matrix X, you should get in the habit of passing the labels as a separate argument to the `fit()` method.
# 
# You may find it useful to serialize the trained models to disk.  This will allow to reload it after restarting the Jupyter notebook, without needing to retrain it.  We recommend using the [`dill` library](https://pypi.python.org/pypi/dill) for this (although the [`joblib` library](http://scikit-learn.org/stable/modules/model_persistence.html) also works).  Use
# ```python
# dill.dump(estimator, open('estimator.dill', 'w'))
# ```
# to serialize the object `estimator` to the file `estimator.dill`.  If you have trouble with this, try setting the `recurse=True` keyword args in the call of `dill.dump()`.  The estimator can be deserialized by calling
# ```python
# estimator = dill.load(open('estimator.dill', 'r'))
# ```

# # Questions
# 
# Each of the "model" questions asks you to create a function that models the number of stars venues will receive.  It will be passed a list of dictionaries.  Each of these will have the same format as the JSON objects you've just read in.  Some of the keys (like the stars!) will have been removed.  This function should return a list of numbers of the same length, indicating the predicted star ratings.
# 
# This function is passed to the `score()` function, which will receive input from the grader, run your function with that input, report the results back to the grader, and print out the score the grader returned.  Depending on how you constructed your estimator, you may be able to pass the predict method directly to the `score()` function.  If not, you will need to write a small wrapper function to mediate the data types.
# 
# ## city_avg
# The venues belong to different cities.  You can image that the ratings in some cities are probably higher than others.  We wish to build an estimator to make a prediction based on this, but first we need to work out the average rating for each city.  For this problem, create a list of tuples (city name, star rating), one for each city in the dataset.
# 
# There are many ways to do this; please feel free to experiment on your own.  If you get stuck, the steps below attempt to guide you through the process.
# 
# A simple approach is to go through all of the dictionaries in our array, calculating the sum of the star ratings and the number of venues for each city.  At the end, we can just divide the stars by the count to get the average.
# 
# We could create a separate sum and count variable for each city, but that will get tedious quickly.  A better approach to to create a dictionary for each.  The key will be the city name, and the value the running sum or running count.
# 
# One slight annoyance of this approach is that we will have to test whether a key exists in the dictionary before adding to the running tally.  The collections module's `defaultdict` class works around this by providing default values for keys that haven't been used.  Thus, if we do

# In[28]:


from collections import defaultdict
star_sum = defaultdict(int)
count = defaultdict(int)


# we can increment any key of `stars` or `count` without first worrying whether the key exists.  We need to go through the `data` and `star_ratings` list together, which we can do with the `zip()` function.

# In[29]:


for row, stars in zip(data, star_ratings):
    star_sum[row['city']] += stars
    count[row['city']] += 1
    # increment the running sum in star_sum
    # increment the running count in count


# In[30]:


len(star_sum)


# Now we can calculate the average ratings.  Again, a dictionary makes a good container.  (*N.B.* Watch out for Python2's integer division.  Make sure at least one of your values is a float to get floating point division.)

# In[31]:


avg_stars = dict()
for city in star_sum:
    avg_stars[city]=star_sum[city]/count[city]
    # calculate average star rating and store in avg_stars


# In[50]:


#avg_stars


# There should be 167 different cities:

# In[32]:


assert len(avg_stars) == 167


# We can get that list of tuples from the `.items()` method.

# In[34]:


grader.score('ml__city_avg', lambda: avg_stars.items())


# ## city_model
# Now, let's build a custom estimator that will make a prediction based solely on the city of a venue.  It is tempting to hard-code the answers from the previous section into this model, but we're going to resist and do things properly.
# 
# This custom estimator will have a `.fit()` method.  It will receive `data` as its argument `X` and `star_ratings` as `y`, and should repeat the calculation of the previous problem there.  Then the `.predict()` method can look up the average rating for the city of each record it receives.

# In[9]:


from sklearn import base
from collections import defaultdict

class CityEstimator(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self):
        self.avg_stars = dict()
    
    def fit(self, X, y):
        star_sum = defaultdict(int)
        count = defaultdict(int)
        for row, stars in zip(X, y):
            star_sum[row['city']] += stars
            count[row['city']] += 1
        for city in star_sum:
            self.avg_stars[city]=star_sum[city]/count[city]
        # Store the average rating per city in self.avg_stars
        return self
    
    def predict(self, X):
        result=[]
        for row in X:
            if row['city'] not in self.avg_stars.keys():
                result.append(0)
            else:
                result.append(self.avg_stars[row['city']])
        return result
        #return [self.avg_stars[row['city']] for row in X]


# Now we can create an instance of our estimator and train it.

# In[61]:


city_est = CityEstimator()
city_est.fit(data, star_ratings)


# In[53]:


data[2]['city']


# And let's see if it works.

# There is a problem, however.  What happens if we're asked to estimate the rating of a venue in a city that's not in our training set?

# In[62]:


city_est.predict(data[:5])


# In[63]:


city_est.predict([{'city': 'Timbuktu'}])


# Solve this problem before submitting to the grader.

# In[64]:


grader.score('ml__city_model', city_est.predict)


# ## lat_long_model
# You can imagine that a city-based model might not be sufficiently fine-grained. For example, we know that some neighborhoods are trendier than others.  Use the latitude and longitude of a venue as features that help you understand neighborhood dynamics.
# 
# Instead of writing a custom estimator, we'll use one of the built-in estimators in Scikit Learn.  Since these estimators won't know what to do with a list of dictionaries, we'll build a `ColumnSelectTransformer` that will return an array containing selected keys of our feature matrix.  While it is tempting to hard-code the latitude and longitude in here, this transformer will be more useful in the future if we write it to work on an arbitrary list of columns.

# In[11]:


import numpy


# In[10]:


class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, col_names):
        self.col_names = col_names  # We will need these in transform()
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    
    def transform(self, X):
        result=[]
        for i in range(len(X)):
            a=X[i]
            temp=[]
            for j in self.col_names:
                temp.append(a[j])
            result.append(temp)
        #result_new=numpy.empty([len(result),2])
        #result_new[:,0]=numpy.asarray(result)
        return result
        # Return an array with the same number of rows as X and one
        # column for each in self.col_names


# In[212]:


#data[1]


# In[81]:


cst = ColumnSelectTransformer(['latitude', 'longitude'])


# In[440]:


cst.fit_transform(data[:1])


# Let's test it on a single row, just as a sanity check:

# In[79]:


cst_new = ColumnSelectTransformer(['categories'])


# In[444]:


cst_new.fit_transform(data[0:3])


# In[357]:


assert (cst.fit_transform(data[:1])
        == [[data[0]['latitude'], data[0]['longitude']]])


# Now, let's feed the output of the transformer in to a `sklearn.neighbors.KNeighborsRegressor`.  As a sanity check, we'll test it with the first 5 rows.  To truly judge the performance, we'd need to make a test/train split.

# In[82]:


from sklearn.neighbors import KNeighborsRegressor

data_transform = cst.fit_transform(data)
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(data_transform, star_ratings)
test_data = data[:5]
test_data_transform = cst.transform(test_data)
knn.predict(test_data_transform)


# Instead of doing this by hand, let's make a pipeline.  Remember that a pipeline is made with a list of (name, transformer-or-estimator) tuples. 

# In[359]:


star_ratings[:5]


# In[17]:


from sklearn.pipeline import Pipeline


# In[360]:



        
pipe = Pipeline([('cst',ColumnSelectTransformer(['latitude', 'longitude'])),('KNR', KNeighborsRegressor(n_neighbors=5))])
        # ColumnSelectTransformer
        # KNeighborsRegressor


# This should work the same way.

# In[361]:


pipe.fit(data, star_ratings)
pipe.predict(test_data)


# In[362]:


star_ratings[0:5]


# The `KNeighborsRegressor` takes the `n_neighbors` hyperparameter, which tells it how many nearest neighbors to average together when making a prediction.  There is no reason to believe that 5 is the optimum value.  Determine a better value of this hyperparameter.   There are several ways to do this:
# 
# 1. Use [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) to split your data in to a training set and a test set.  Score the performance on the test set.  After finding the best hyperparameter, retrain the model on the full data at that hyperparameter value.
# 
# 2. Use [cross_val_score](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score) to return cross-validation scores on your data for various values of the hyperparameter.  Choose the best one, and retrain the model on the full data.
# 
# 3. Use [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) to do the splitting, training, and grading automatically.  `GridSearchCV` takes an estimator and acts as an estimator.  You can either give it the `KNeighborsRegressor` directly and put it in a pipeline, or you can pass the whole pipeline into the `GridSearchCV`.  In the latter case, remember that the hyperparameter `param` of an estimator named `est` in a pipeline becomes a hyperparameter of the pipeline with name `est__param`.
# 
# No matter which you choose, you should consider whether the data need to be shuffled.  The default k-folds split doesn't shuffle.  This is fine, if the data are already random.  The code below will plot a rolling mean of the star ratings.  Do you need to shuffle the data?

# In[363]:


from pandas import Series
import matplotlib.pyplot as plt

plt.plot(Series.rolling(Series(star_ratings), window=1000).mean())


# Once you've found a good value of `n_neighbors`, submit the model to the grader.  (*N.B.* "Good" is a relative measure here.  The reference solution has a r-squared value of only 0.02.  There is just rather little signal available for modeling.)

# In[16]:


from sklearn.utils import shuffle


# In[365]:


a=[[1., 0.], [2., 1.], [0., 0.]]
print shuffle(a)


# In[366]:


from sklearn.model_selection import cross_val_score


# In[367]:


from sklearn import svm


# In[368]:


import numpy


# In[378]:


data_shuffled = shuffle(data)
star_ratings_shuffled = [row['stars'] for row in data_shuffled]


# In[268]:


#data_shuffled[1]


# In[263]:


#plt.plot(Series.rolling(Series(star_ratings_shuffled), window=1000).mean())


# In[379]:


cst = ColumnSelectTransformer(['latitude', 'longitude'])
data_shuffled_transform = cst.fit_transform(data_shuffled)


# In[383]:


for num in range(80,100):
    knn = KNeighborsRegressor(n_neighbors=num)
    a=numpy.mean(cross_val_score(knn, data_shuffled_transform, star_ratings_shuffled,cv=10))
    print num, a
#pipe = Pipeline([('cst',ColumnSelectTransformer(['latitude', 'longitude'])),('KNR', KNeighborsRegressor(n_neighbors=num))])

#pipe.fit(data_shuffled, star_ratings_shuffled)
#star_predict=predict=pipe.predict(data_shuffled)
#clf = svm.SVC(kernel='linear', C=1)


# In[ ]:





# In[290]:


#numpy array two dimensions


# In[84]:


best_num=82


# In[385]:


knn=KNeighborsRegressor(n_neighbors=best_num)
knn.fit(data_shuffled_transform, star_ratings_shuffled)
test_data = data_shuffled[:10]
test_data_transform = cst.transform(test_data)
knn.predict(test_data_transform)


# In[386]:


star_ratings_shuffled[:10]


# In[85]:


pipe = Pipeline([('cst',ColumnSelectTransformer(['latitude', 'longitude'])),('KNR', KNeighborsRegressor(n_neighbors=best_num))])


# In[388]:


pipe.fit(data, star_ratings)


# In[389]:


grader.score('ml__lat_long_model', pipe.predict)  # Edit to appropriate name


# *Item for thought:* Why do we choose a non-linear model for this estimator?
# 
# *Extension:* Use a `sklearn.ensemble.RandomForestRegressor`, which is a more powerful non-linear model.  Can you get better performance with this than with the `KNeighborsRegressor`?

# ## category_model
# While location is important, we could also try seeing how predictive the
# venue's category is.  Build an estimator that considers only the categories.
# 
# The categories come as a list of strings, but the built-in estimators all need numeric input.  The standard way to deal with categorical features is **one-hot encoding**, also known as dummy variables.  In this approach, each category gets its own column in the feature matrix.  If the row has a given category, that column gets filled with a 1.  Otherwise, it is 0.
# 
# The `ColumnSelectTransformer` from the previous question can be used to extract the categories column as a list of strings.  Scikit Learn provides [DictVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer), which takes in a list of dictionaries.  It creates a column in the output matrix for each key in the dictionary and fills it with the value associated with it.  Missing keys are filled with zeros.  Therefore, we need only build a transformer that takes a list strings to a dictionary with keys given by those strings and values one.

# In[7]:


from sklearn.feature_extraction import DictVectorizer


# In[12]:


class DictEncoder(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        list=[]
        for i in X:
            dict_temp={}
            for j in i:
                for k in j:
                    dict_temp[k]=1
                list.append(dict_temp)
        return list
        # X will come in as a list of lists of lists.  Return a list of
        # dictionaries corresponding to those inner lists.


# In[399]:





# In[395]:


DictEncoder().fit_transform([[['a']], [['b', 'c']]])


# That should allow this to pass:

# In[393]:


assert (DictEncoder().fit_transform([[['a']], [['b', 'c']]])
        == [{'a': 1}, {'b': 1, 'c': 1}])


# In[401]:


v = DictVectorizer(sparse=False)
X = v.fit_transform(DictEncoder().fit_transform([[['a']], [['b', 'c']]]))
X


# In[956]:


cat=ColumnSelectTransformer(['categories'])
cat.fit_transform(data[0:10])


# In[957]:


v.fit_transform(DictEncoder().fit_transform(cat.fit_transform(data[0:5])))


# In[958]:


DictEncoder().fit_transform(cat.fit_transform(data[0:5]))


# In[959]:


cat=ColumnSelectTransformer(['categories'])


# In[19]:


v = DictVectorizer(sparse=False)


# In[456]:


from sklearn.linear_model import Ridge


# In[457]:


pipe_cat=Pipeline([('cat',cat),('encoder',DictEncoder()),('vec',v), ('ridge', Ridge(alpha = 0.5))])


# Set up a pipeline with your `ColumnSelectTransformer`, your `DictEncoder`, the `DictVectorizer`, and a regularized linear model, like `Ridge`, as the estimator.  This model will have a large number of features, one for each category, so there is a significant danger of overfitting.  Use cross validation to choose the best regularization parameter.

# In[459]:


pipe_cat.fit(data, star_ratings)


# In[460]:


test_data = data[:10]
pipe_cat.predict(test_data)


# In[463]:


import numpy as np
np.linspace(0.1, 1, num=10)


# In[470]:


for alpha in np.linspace(0.01, 0.1, num=10):
#for alpha in np.logspace(-4., 0, 20):  
    pipe_cat=Pipeline([('cat',cat),('encoder',DictEncoder()),('vec',v), ('ridge', Ridge(alpha = alpha))])
    a=np.mean(cross_val_score(pipe_cat, data, star_ratings,cv=10))
    print alpha, a


# In[476]:


alpha=0.01


# In[477]:


pipe_cat=Pipeline([('cat',cat),('encoder',DictEncoder()),('vec',v), ('ridge', Ridge(alpha = alpha))])
pipe_cat.fit(data, star_ratings)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[478]:


grader.score('ml__category_model', pipe_cat.predict)  # Edit to appropriate name


# *Extension:* Some categories (e.g. Restaurants) are not very specific.  Others (Japanese sushi) are much more so.  One way to deal with this is with an measure call term-frequency-inverse-document-frequency (TF-IDF).  Add in a `sklearn.feature_extraction.text.TfidfTransformer` between the `DictVectorizer` and the linear model, and see if that improves performance.
# 
# *Extension:* Can you beat the performance of the linear estimator with a
# non-linear model?

# ## attribute_model
# There is even more information in the attributes for each venue.  Let's build an estimator based on these.
# 
# Venues attributes may be nested:
# ```
# {
#   'Attire': 'casual',
#   'Accepts Credit Cards': True,
#   'Ambiance': {'casual': False, 'classy': False}
# }
# ```
# We wish to encode them with one-hot encoding.  The `DictVectorizer` can do this, but only once we've flattened the dictionary to a single level, like so:
# ```
# {
#   'Attire_casual' : 1,
#   'Accepts Credit Cards': 1,
#   'Ambiance_casual': 0,
#   'Ambiance_classy': 0
# }
# ```
# 
# Build a custom transformer that flattens the attributes dictionary.  Place this in a pipeline with a `DictVectorizer` and a regressor.
# 
# You may find it difficult to find a single regressor that does well enough.  A common solution is to use a linear model to fit the linear part of some data, and use a non-linear model to fit the residual that the linear model can't fit.  Build a residual estimator that takes as an argument two other estimators.  It should use the first to fit the raw data and the second to fit the residuals of the first.

# In[13]:


att=ColumnSelectTransformer(['attributes'])


# In[14]:


class FlatTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    
    def transform(self, X):
        attr=att.fit_transform(X)
        result=[]
        #for i in X:
        for i in attr:
            l=[]
            d={}
            ii=i[0]
            for j in ii.keys():
                sth=ii[j]
                if type(sth)==dict:
                    for k in sth.keys():
                        new_key=str(j)+'_'+str(k)
                        d.update({new_key:sth[k]})
                elif j=='Price Range':
                    new_key=str(j)+'_'+str(sth)
                    d.update({new_key:'True'})
                elif sth==True or sth==False:
                    d.update({str(j):sth})
                else:
                    new_key=str(j)+'_'+str(sth)
                    d.update({new_key:'True'})
            l.append(d)
            result.append(l)
        result_new=DictEncoder().fit_transform(result)
        return result_new


# In[1757]:


data[3012]


# In[1759]:



v.fit_transform(FlatTransformer().fit_transform(data[3010:3020]))


# In[1595]:


len(data)


# In[1614]:


list(v.fit_transform(FlatTransformer().fit_transform(data[2000:2020])))[0]


# In[1413]:


att=ColumnSelectTransformer(['attributes'])


# In[1619]:


a=[[1,2],[2,1]]
b=[[3,3],[2,2]]


# In[1659]:


datatemp=list(v.fit_transform(FlatTransformer().fit_transform(data[:10])))


# In[1660]:


datatemp[4]


# In[1650]:


zeroarr=np.zeros(len(data))


# In[1655]:


np.asarray(datatemp[10])-np.asarray(datatemp[10])


# In[1653]:


np.all([np.asarray(datatemp[10])-np.asarray(datatemp[10]),zeroarr])


# In[22]:


import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn import neighbors
from sklearn import ensemble
from sklearn.pipeline import Pipeline, FeatureUnion


# In[86]:


class MixedEstimator(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self, base_estimator, residual_estimator):
        self.base_estimator = base_estimator
        self.residual_estimator = residual_estimator
    
    def fit(self, X, y):
        
        self.base_estimator.fit(X, y)
        y_err = y - self.base_estimator.predict(X)
        #for est in self.residual_estimators:
            #est.fit(X, y_err)
        self.residual_estimator.fit(X, y_err)
        return self
    
    def predict(self, X):
        all_ests = [self.base_estimator] + [self.residual_estimator]
        #all_ests = [self.base_estimator] + list(self.residual_estimators)
        temp=np.array([est.predict(X) for est in all_ests]).T
        
        #return temp
        return temp.sum(axis=1)
        
        #return temp[:,0]+[item*1.2 for item in temp[:,1]]


# In[ ]:





# In[1415]:


from sklearn import model_selection


# In[1416]:





# In[1417]:


from sklearn.metrics import mean_squared_error


# In[1418]:


from sklearn.metrics import accuracy_score


# In[1419]:


import numpy as np


# In[ ]:





# In[1693]:


pipeatt.predict(data[:3])


# In[23]:


class EnsembleTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, base_estimator, residual_estimator):
        self.base_estimator = base_estimator
        self.residual_estimator = residual_estimator
    
    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        y_err = y - self.base_estimator.predict(X)
        self.residual_estimator.fit(X, y_err)
        return self
    
    def transform(self, X):
        all_ests = [self.base_estimator] + [self.residual_estimator]
        return np.array([est.predict(X) for est in all_ests]).T


# In[1537]:


data_shuffled = shuffle(data)
star_ratings_shuffled = [row['stars'] for row in data_shuffled]
plt.plot(Series.rolling(Series(star_ratings_shuffled), window=1000).mean())


# In[58]:


from pandas import Series
import matplotlib.pyplot as plt


# In[1753]:


#pipeattnew.predict(data[0:10])


# In[1754]:


#plt.plot(Series.rolling(Series(star_ratings[300:400]), window=1).mean())
#plt.plot(Series.rolling(Series(pipeatt.predict(data[300:400])), window=1).mean())


# In[ ]:





# In[1685]:


pipetest.predict(data[:5])


# In[1668]:





# In[1157]:


#plt.plot(Series.rolling(Series(pipeatt.predict(data[:100])), window=1).mean())


# In[1702]:


#pipetest.get_params()


# In[25]:





# In[62]:


pipetest = Pipeline([('Flat',FlatTransformer()),('vec',v),('Mixed', MixedEstimator(
                Ridge(alpha = 1),
                ensemble.RandomForestRegressor()))])
data_shuffled = shuffle(data)
star_ratings_shuffled = [row['stars'] for row in data_shuffled]

pipetest.fit(data_shuffled, star_ratings_shuffled)


# In[72]:


#data[0:3]
star_ratings_shuffled[0:3]


# In[75]:


#data_shuffled[0:3]
#shuffle(data[0:3])


# In[55]:


pipetest.predict(data[0:10])


# In[ ]:





# In[60]:


plt.plot(Series.rolling(Series(star_ratings[400:500]), window=1).mean())
plt.plot(Series.rolling(Series(pipetest.predict(data[400:500])), window=1).mean())


# In[61]:


#from sklearn import datasets, linear_model, utils, preprocessing
#from sklearn.linear_model import Ridge
#pipeattnew = Pipeline([('Flat',FlatTransformer()),('vec',v),
#        ('ensemble', EnsembleTransformer(Ridge(alpha=1),
#                 ensemble.RandomForestRegressor(n_estimators=150,min_samples_leaf=50))),
#        ('blend', Ridge(alpha=1))
#    ])
#data_shuffled = shuffle(data)
#star_ratings_shuffled = [row['stars'] for row in data_shuffled]

#pipeattnew.fit(data_shuffled, star_ratings_shuffled)


# In[114]:


class AttriFlatten(base.BaseEstimator, base.TransformerMixin):
    def __init__(self):
        return
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self
    
    def transform(self, X):
        return [flatten_att(row['attributes']) for row in X]


# In[115]:


class EnsembleTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, base_estimator, residual_estimators):
        self.base_estimator = base_estimator
        self.residual_estimators = residual_estimators
    
    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        y_err = y - self.base_estimator.predict(X)
        for est in self.residual_estimators:
            est.fit(X, y_err)
        return self
    
    def transform(self, X):
        all_ests = [self.base_estimator] + list(self.residual_estimators)
        return np.array([est.predict(X) for est in all_ests]).T


# In[116]:


def flatten_att(the_dict):
    result={}
    for the_key in the_dict:
        da=the_dict[the_key]
        if isinstance(da,str):
            result[the_key+'_'+da]=1
        elif isinstance(da,unicode):
            result[the_key+'_'+da]=1
        elif isinstance(da,int):
            result[the_key+'_'+str(da)]=1
        elif isinstance(da,bool):
            result[the_key]=int(da)
        elif isinstance(da,dict):
            
            resultl=flatten_att(data)
            for the_key_l in resultl:
                result[the_key+'_'+the_key_l]=resultl[the_key_l]
        else:
            print 'Error', the_key, da
            sys.exit()
    return result


# In[117]:


ensemble_pipe = Pipeline([('att',AttriFlatten()),('vec',v),
        ('ensemble', EnsembleTransformer(
                linear_model.LinearRegression(),
                 ensemble.RandomForestRegressor(min_samples_leaf=20))),
        ('blend', linear_model.LinearRegression())
    ])


# In[ ]:


data_shuffled = shuffle(data)
star_ratings_shuffled = [row['stars'] for row in data_shuffled]
ensemble_pipe.fit(data_shuffled, star_ratings_shuffled)


# In[54]:


grader.score('ml__attribute_model', pipetest.predict)  # Edit to appropriate name


# ## full_model
# So far we have only built models based on individual features.  Now we will build an ensemble regressor that averages together the estimates of the four previous regressors.
# 
# In order to use the existing models as input to an estimator, we will have to turn them into transformers.  (A pipeline can contain at most a single estimator.)  Build a custom `ModelTransformer` class that takes an estimator as an argument.  When `fit()` is called, the estimator should be fit.  When `transform()` is called, the estimator's `predict()` method should be called, and its results returned.
# 
# Note that the output of the `transform()` method should be a 2-D array with a single column, in order for it to work well with the Scikit Learn pipeline.  If you're using Numpy arrays, you can use `.reshape(-1, 1)` to create a column vector.  If you are just using Python lists, you will want a list of lists of single elements.

# In[1320]:


class EstimatorTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, estimator):
        self.base_estimator = estimator
        # What needs to be done here?
    
    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        return self
        # Fit the stored estimator.
        # Question: what should be returned?
    
    def transform(self, X):
        est = self.base_estimator
        return (np.array(est.predict(X))).reshape(-1,1)
        # Use predict on the stored estimator as a "transformation".
        # Be sure to return a 2-D array.


# In[1322]:


[r[0] for r in city_trans.transform(data[:5])]


# In[1323]:


city_est.predict(data[:5])


# This should work as follows:

# In[1321]:


city_trans = EstimatorTransformer(city_est)
city_trans.fit(data, star_ratings)
assert ([r[0] for r in city_trans.transform(data[:5])]
        == city_est.predict(data[:5]))


# Create an instance of `ModelTransformer` for each of the previous four models. Combine these together in a single feature matrix with a
# [FeatureUnion](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion).

# In[1327]:


from sklearn.pipeline import FeatureUnion

union = FeatureUnion([('city',EstimatorTransformer(city_est)),('lat-long',EstimatorTransformer(pipe)),('category',EstimatorTransformer(pipe_cat)),
                      ('attribute',EstimatorTransformer(pipeatt))])
        # FeatureUnions use the same syntax as Pipelines


# This should return a feature matrix with four columns.

# In[1328]:


union.fit(data, star_ratings)
trans_data = union.transform(data[:10])
assert trans_data.shape == (10, 4)


# Finally, use a pipeline to combine the feature union with a linear regression (or another model) to weight the predictions.

# In[1330]:


alpha =1
pipefull=Pipeline([("features", union),("ridge", Ridge(alpha = alpha))])


# In[1331]:


data_shuffled = shuffle(data)
star_ratings_shuffled = [row['stars'] for row in data_shuffled]
pipefull.fit(data_shuffled, star_ratings_shuffled)


# In[1336]:


pipefull.get_params


# In[1338]:


grader.score('ml__full_model', pipefull.predict)  # Edit to appropriate name


# *Extension:* By combining our models with a linear model, we will be unable to notice any correlation between features.  We don't expect all attributes to have the same effect on all venues.  For example, "Ambiance: divey" might be a bad indicator for a restaurant but a good one for a bar.  Nonlinear models can pick up on this interaction.  Replace the linear model combining the predictions with a nonlinear one like [RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor).  Better yet, use the nonlinear model to fit the residuals of the linear model.
# 
# The score for this question is just a ratio of the score of your model to the score of a reference solution.  Can you beat the reference solution and get a score greater than 1.0?

# *Copyright &copy; 2016 The Data Incubator.  All rights reserved.*
