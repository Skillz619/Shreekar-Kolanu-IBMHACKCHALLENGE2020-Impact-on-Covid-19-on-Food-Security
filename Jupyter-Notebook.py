import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
Step2- Import Dataset

import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
client_7a1bdcbc5de54d7a802c5c28382d0e64 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='biNfbl2MpL8mGNzjsWGZh9JFzmL5JHiL9Wvgh8zLzHHa',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_7a1bdcbc5de54d7a802c5c28382d0e64.get_object(Bucket='visualrecognition-donotdelete-pr-nmanlpo4l3a29m',Key='Salary_Data.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)
dataset.head()
YearsExperience	Salary
0	1.1	39343.0
1	1.3	46205.0
2	1.5	37731.0
3	2.0	43525.0
4	2.2	39891.0
type(dataset)
pandas.core.frame.DataFrame
dataset
YearsExperience	Salary
0	1.1	39343.0
1	1.3	46205.0
2	1.5	37731.0
3	2.0	43525.0
4	2.2	39891.0
5	2.9	56642.0
6	3.0	60150.0
7	3.2	54445.0
8	3.2	64445.0
9	3.7	57189.0
10	3.9	63218.0
11	4.0	55794.0
12	4.0	56957.0
13	4.1	57081.0
14	4.5	61111.0
15	4.9	67938.0
16	5.1	66029.0
17	5.3	83088.0
18	5.9	81363.0
19	6.0	93940.0
20	6.8	91738.0
21	7.1	98273.0
22	7.9	101302.0
23	8.2	113812.0
24	8.7	109431.0
25	9.0	105582.0
26	9.5	116969.0
27	9.6	112635.0
28	10.3	122391.0
29	10.5	121872.0
dataset.head()
YearsExperience	Salary
0	1.1	39343.0
1	1.3	46205.0
2	1.5	37731.0
3	2.0	43525.0
4	2.2	39891.0
dataset.head(5)
YearsExperience	Salary
0	1.1	39343.0
1	1.3	46205.0
2	1.5	37731.0
3	2.0	43525.0
4	2.2	39891.0
Step3- Split Independent and Dpendent Variables

x= dataset.iloc[:,:1]
x
YearsExperience
0	1.1
1	1.3
2	1.5
3	2.0
4	2.2
5	2.9
6	3.0
7	3.2
8	3.2
9	3.7
10	3.9
11	4.0
12	4.0
13	4.1
14	4.5
15	4.9
16	5.1
17	5.3
18	5.9
19	6.0
20	6.8
21	7.1
22	7.9
23	8.2
24	8.7
25	9.0
26	9.5
27	9.6
28	10.3
29	10.5
type(x)
pandas.core.frame.DataFrame
x= dataset.iloc[:,:-1].values #convert from dataframe to numpy array
x
array([[ 1.1],
       [ 1.3],
       [ 1.5],
       [ 2. ],
       [ 2.2],
       [ 2.9],
       [ 3. ],
       [ 3.2],
       [ 3.2],
       [ 3.7],
       [ 3.9],
       [ 4. ],
       [ 4. ],
       [ 4.1],
       [ 4.5],
       [ 4.9],
       [ 5.1],
       [ 5.3],
       [ 5.9],
       [ 6. ],
       [ 6.8],
       [ 7.1],
       [ 7.9],
       [ 8.2],
       [ 8.7],
       [ 9. ],
       [ 9.5],
       [ 9.6],
       [10.3],
       [10.5]])
x.ndim #mandatory to be in 2 dimesion for Linear Regression
2
type(x)
numpy.ndarray
y= dataset.iloc[:,1:]
y
Salary
0	39343.0
1	46205.0
2	37731.0
3	43525.0
4	39891.0
5	56642.0
6	60150.0
7	54445.0
8	64445.0
9	57189.0
10	63218.0
11	55794.0
12	56957.0
13	57081.0
14	61111.0
15	67938.0
16	66029.0
17	83088.0
18	81363.0
19	93940.0
20	91738.0
21	98273.0
22	101302.0
23	113812.0
24	109431.0
25	105582.0
26	116969.0
27	112635.0
28	122391.0
29	121872.0
y= dataset.iloc[:,1:].values
y
array([[ 39343.],
       [ 46205.],
       [ 37731.],
       [ 43525.],
       [ 39891.],
       [ 56642.],
       [ 60150.],
       [ 54445.],
       [ 64445.],
       [ 57189.],
       [ 63218.],
       [ 55794.],
       [ 56957.],
       [ 57081.],
       [ 61111.],
       [ 67938.],
       [ 66029.],
       [ 83088.],
       [ 81363.],
       [ 93940.],
       [ 91738.],
       [ 98273.],
       [101302.],
       [113812.],
       [109431.],
       [105582.],
       [116969.],
       [112635.],
       [122391.],
       [121872.]])
If you have null values in the dataset
Step6- Split Test and Train Data

from sklearn.model_selection import train_test_split                #previously cros_validation was used in sklearn
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train
array([[ 9.6],
       [ 4. ],
       [ 5.3],
       [ 7.9],
       [ 2.9],
       [ 5.1],
       [ 3.2],
       [ 4.5],
       [ 8.2],
       [ 6.8],
       [ 1.3],
       [10.5],
       [ 3. ],
       [ 2.2],
       [ 5.9],
       [ 6. ],
       [ 3.7],
       [ 3.2],
       [ 9. ],
       [ 2. ],
       [ 1.1],
       [ 7.1],
       [ 4.9],
       [ 4. ]])
x_test
array([[ 1.5],
       [10.3],
       [ 4.1],
       [ 3.9],
       [ 9.5],
       [ 8.7]])
tinyurl.com/sample-linear
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)
y_predict=lr.predict(x_test)
y_predict
array([[ 40748.96184072],
       [122699.62295594],
       [ 64961.65717022],
       [ 63099.14214487],
       [115249.56285456],
       [107799.50275317]])
y_test
array([[ 37731.],
       [122391.],
       [ 57081.],
       [ 63218.],
       [116969.],
       [109431.]])
lr.predict(np.array([[5]]))
array([[73342.97478427]])
#visualization of train data
plt.scatter(x_train,y_train,color = 'green')
plt.plot(x_train,lr.predict(x_train),color = 'Red')
plt.xlabel("Exper.")
plt.ylabel("Salary")
plt.title("Salary vs Experience(train)")
plt.show()

#visualization of train data
plt.scatter(x_test,y_test,color = 'green')
plt.plot(x_test,lr.predict(x_test),color = 'Red')
plt.xlabel("Exper.")
plt.ylabel("Salary")
plt.title("Salary vs Experience(train)")
plt.show()

from watson_machine_learning_client import watsonMachineLearningAPIClient
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
<ipython-input-32-76d894351044> in <module>
----> 1 from watson_machine_learning_client import watsonMachineLearningAPIClient

ImportError: cannot import name 'watsonMachineLearningAPIClient'
2020-06-25 13:32:39,422 - watson_machine_learning_client.metanames - WARNING - 'AUTHOR_EMAIL' meta prop is deprecated. It will be ignored.
from watson_machine_learning_client import WatsonMachineLearningAPIClient
wml_credenntials = {
  "apikey": "wuhEHht0x02LSC6WDg38BIv2lgAsNGQSH83dsjsWeNmA",
  "iam_apikey_description": "Auto-generated for key 8311ba23-47b2-4577-abee-37dd3ad9e422",
  "iam_apikey_name": "Service credentials-1",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/e48c955daf1f4120a8586da3383afa75::serviceid:ServiceId-33ec0cd0-56b9-470a-82e2-949044f04168",
  "instance_id": "ab2985f0-9795-4d1a-856e-62e332152a8b",
  "url": "https://eu-gb.ml.cloud.ibm.com"
}
client = WatsonMachineLearningAPIClient(wml_credenntials)
model_props = {
    client.repository.ModelMetaNames.Author_Name : 'Shreekar',
    client.repository.ModelMetaNames.Author_Email : 'shreekarkolanu@gmail.com',
    client.repository.ModelMetaNames.Name : 'Salary Data'
}
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-36-e0f87ef461ab> in <module>
      1 model_props = {
----> 2     client.repository.ModelMetaNames.Author_Name : 'Shreekar',
      3     client.repository.ModelMetaNames.Author_Email : 'shreekarkolanu@gmail.com',
      4     client.repository.ModelMetaNames.Name : 'Salary Data'
      5 }

AttributeError: 'ModelMetaNames' object has no attribute 'Author_Name'
model_props = {
    client.repository.ModelMetaNames.AUTHOR_NAME : 'Shreekar',
    client.repository.ModelMetaNames.AUTHOR_EMAIL : 'shreekarkolanu@gmail.com',
    client.repository.ModelMetaNames.NAME : 'Salary Data'
}
model_artifact = client.repository.store_model(lr,meta_props = model_props)
model_artifact
{'metadata': {'guid': '397a5dec-ab35-4a91-83fa-37e9a3201e5c',
  'url': 'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/ab2985f0-9795-4d1a-856e-62e332152a8b/published_models/397a5dec-ab35-4a91-83fa-37e9a3201e5c',
  'created_at': '2020-06-25T13:32:40.248Z',
  'modified_at': '2020-06-25T13:32:40.300Z'},
 'entity': {'runtime_environment': 'python-3.6',
  'learning_configuration_url': 'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/ab2985f0-9795-4d1a-856e-62e332152a8b/published_models/397a5dec-ab35-4a91-83fa-37e9a3201e5c/learning_configuration',
  'author': {'name': 'Shreekar'},
  'name': 'Salary Data',
  'learning_iterations_url': 'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/ab2985f0-9795-4d1a-856e-62e332152a8b/published_models/397a5dec-ab35-4a91-83fa-37e9a3201e5c/learning_iterations',
  'feedback_url': 'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/ab2985f0-9795-4d1a-856e-62e332152a8b/published_models/397a5dec-ab35-4a91-83fa-37e9a3201e5c/feedback',
  'latest_version': {'url': 'https://eu-gb.ml.cloud.ibm.com/v3/ml_assets/models/397a5dec-ab35-4a91-83fa-37e9a3201e5c/versions/321c53df-4af5-4cfe-834d-5736ec1c9e76',
   'guid': '321c53df-4af5-4cfe-834d-5736ec1c9e76',
   'created_at': '2020-06-25T13:32:40.300Z'},
  'model_type': 'scikit-learn-0.20',
  'deployments': {'count': 0,
   'url': 'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/ab2985f0-9795-4d1a-856e-62e332152a8b/published_models/397a5dec-ab35-4a91-83fa-37e9a3201e5c/deployments'},
  'evaluation_metrics_url': 'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/ab2985f0-9795-4d1a-856e-62e332152a8b/published_models/397a5dec-ab35-4a91-83fa-37e9a3201e5c/evaluation_metrics'}}
guid = client.repository.get_lr_uid(model_artifact)
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-40-7ec7112d7c21> in <module>
----> 1 guid = client.repository.get_lr_uid(model_artifact)

AttributeError: 'Repository' object has no attribute 'get_lr_uid'
guid = client.repository.get_model_uid(model_artifact)
guid
'397a5dec-ab35-4a91-83fa-37e9a3201e5c'
deploy = client.deployments.create(guid,name="Salary Prediction")

#######################################################################################

Synchronous deployment creation for uid: '397a5dec-ab35-4a91-83fa-37e9a3201e5c' started

#######################################################################################


INITIALIZING
DEPLOY_SUCCESS


------------------------------------------------------------------------------------------------
Successfully finished deployment creation, deployment_uid='14184773-681c-4a82-a4ac-7ad66f6323e7'
------------------------------------------------------------------------------------------------


client.deployment.list()
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-44-a294a37ed1c9> in <module>
----> 1 client.deployment.list()

AttributeError: 'WatsonMachineLearningAPIClient' object has no attribute 'deployment'
client.deployments.list()
------------------------------------  -----------------  ------  --------------  ------------------------  -----------------  -------------
GUID                                  NAME               TYPE    STATE           CREATED                   FRAMEWORK          ARTIFACT TYPE
14184773-681c-4a82-a4ac-7ad66f6323e7  Salary Prediction  online  DEPLOY_SUCCESS  2020-06-25T13:36:13.758Z  scikit-learn-0.20  model
------------------------------------  -----------------  ------  --------------  ------------------------  -----------------  -------------
deploy
{'metadata': {'guid': '14184773-681c-4a82-a4ac-7ad66f6323e7',
  'url': 'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/ab2985f0-9795-4d1a-856e-62e332152a8b/deployments/14184773-681c-4a82-a4ac-7ad66f6323e7',
  'created_at': '2020-06-25T13:36:13.758Z',
  'modified_at': '2020-06-25T13:36:13.989Z'},
 'entity': {'runtime_environment': 'python-3.6',
  'name': 'Salary Prediction',
  'scoring_url': 'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/ab2985f0-9795-4d1a-856e-62e332152a8b/deployments/14184773-681c-4a82-a4ac-7ad66f6323e7/online',
  'deployable_asset': {'name': 'Salary Data',
   'url': 'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/ab2985f0-9795-4d1a-856e-62e332152a8b/published_models/397a5dec-ab35-4a91-83fa-37e9a3201e5c',
   'guid': '397a5dec-ab35-4a91-83fa-37e9a3201e5c',
   'created_at': '2020-06-25T13:36:13.735Z',
   'type': 'model'},
  'description': 'Description of deployment',
  'status_details': {'status': 'DEPLOY_SUCCESS'},
  'model_type': 'scikit-learn-0.20',
  'status': 'DEPLOY_SUCCESS',
  'type': 'online',
  'deployed_version': {'url': 'https://eu-gb.ml.cloud.ibm.com/v3/ml_assets/models/397a5dec-ab35-4a91-83fa-37e9a3201e5c/versions/321c53df-4af5-4cfe-834d-5736ec1c9e76',
   'guid': '321c53df-4af5-4cfe-834d-5736ec1c9e76'}}}
scoring url = client.deployemnts.get_scoring_url(deploy)
  File "<ipython-input-47-406954ae7c36>", line 1
    scoring url = client.deployemnts.get_scoring_url(deploy)
              ^
SyntaxError: invalid syntax
scoring_url = client.deployemnts.get_scoring_url(deploy)
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-48-48c7b8dbda5a> in <module>
----> 1 scoring_url = client.deployemnts.get_scoring_url(deploy)

AttributeError: 'WatsonMachineLearningAPIClient' object has no attribute 'deployemnts'
scoring_url = client.deployments.get_scoring_url(deploy)
scoring url = client.deployemnts.get_scoring_url(deploy)
  File "<ipython-input-50-406954ae7c36>", line 1
    scoring url = client.deployemnts.get_scoring_url(deploy)
              ^
SyntaxError: invalid syntax
deploy
{'metadata': {'guid': '14184773-681c-4a82-a4ac-7ad66f6323e7',
  'url': 'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/ab2985f0-9795-4d1a-856e-62e332152a8b/deployments/14184773-681c-4a82-a4ac-7ad66f6323e7',
  'created_at': '2020-06-25T13:36:13.758Z',
  'modified_at': '2020-06-25T13:36:13.989Z'},
 'entity': {'runtime_environment': 'python-3.6',
  'name': 'Salary Prediction',
  'scoring_url': 'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/ab2985f0-9795-4d1a-856e-62e332152a8b/deployments/14184773-681c-4a82-a4ac-7ad66f6323e7/online',
  'deployable_asset': {'name': 'Salary Data',
   'url': 'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/ab2985f0-9795-4d1a-856e-62e332152a8b/published_models/397a5dec-ab35-4a91-83fa-37e9a3201e5c',
   'guid': '397a5dec-ab35-4a91-83fa-37e9a3201e5c',
   'created_at': '2020-06-25T13:36:13.735Z',
   'type': 'model'},
  'description': 'Description of deployment',
  'status_details': {'status': 'DEPLOY_SUCCESS'},
  'model_type': 'scikit-learn-0.20',
  'status': 'DEPLOY_SUCCESS',
  'type': 'online',
  'deployed_version': {'url': 'https://eu-gb.ml.cloud.ibm.com/v3/ml_assets/models/397a5dec-ab35-4a91-83fa-37e9a3201e5c/versions/321c53df-4af5-4cfe-834d-5736ec1c9e76',
   'guid': '321c53df-4af5-4cfe-834d-5736ec1c9e76'}}}
scoring_url = client.deployments.get_scoring_url(deploy)
scoring_url
'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/ab2985f0-9795-4d1a-856e-62e332152a8b/deployments/14184773-681c-4a82-a4ac-7ad66f6323e7/online'
client.deployments.list()
------------------------------------  -----------------  ------  --------------  ------------------------  -----------------  -------------
GUID                                  NAME               TYPE    STATE           CREATED                   FRAMEWORK          ARTIFACT TYPE
14184773-681c-4a82-a4ac-7ad66f6323e7  Salary Prediction  online  DEPLOY_SUCCESS  2020-06-25T13:36:13.758Z  scikit-learn-0.20  model
------------------------------------  -----------------  ------  --------------  ------------------------  -----------------  -------------
