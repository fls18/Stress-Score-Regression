# Stress-Score-Regression
스트레스 점수 예측을 위한 회귀 모델


<br>

## 사용 기술
- Python, pandas, scikit-learn (RandomForestRegressor, train_test_split, mean_squared_error), category_encoders


<br>

## 코드 설명
``
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import category_encoders as ce
!pip install category_encoders
``


데이터 불러오기
``
df_train = pd.read_csv('train.csv', encoding='cp949')
df_test = pd.read_csv('test.csv', encoding='cp949')
submission = pd.read_csv('sample_submission.csv', encoding='cp949')
``


학습용 데이터, 타겟 분리
``
x = df_train.drop(['ID', 'stress_score'], axis=1) # 학습에 사용
y = df_train['stress_score']
``


``
encoder = ce.OrdinalEncoder()
x_encoded = encoder.fit_transform(x)

x_train, x_valid, y_train, y_valid = train_test_split(x_encoded, y, test_size=0.33, random_state=42) # 데이터 분리
``


모델 학습
``
rfr = RandomForestRegressor(n_estimators=10, random_state=0)
rfr.fit(x_train, y_train)
``


MSE 계산해 데이터 예측 평가
``
y_pred = rfr.predict(x_valid)
mse = mean_squared_error(y_valid, y_pred) 
print('Model MSE with 10 decision-trees : {0:0.4f}'.format(mse))
``


테스트 데이터 예측해 파일 생성
``
x_test = df_test.drop(['ID'], axis=1)
x_test_encoded = encoder.transform(x_test)
test_preds = rfr.predict(x_test_encoded)

submission['stress_score'] = test_preds
submission.to_csv('sample_submission.csv', index=False, encoding='cp949')
``
