
#streamlit 라이브러리를 불러오기
import streamlit as st
#AI모델을 불러오기 위한 joblib 불러오기
import joblib
import pandas as pd

# st를 이용하여 타이틀과 입력 방법을 명시한다.

def user_input_features() :
  age = st.sidebar.number_input("나이")
  sex = float(st.sidebar.radio("성별(1 = 남성, 0 = 여성) ",('0', '1')))
  cp = int(st.sidebar.selectbox("가슴통증 (1=전형적인 협심증, 2=비전형적인 협심증, 3=협심증이 아닌 통증, 4=무증상)",
                               ('1','2','3','4')))
  trestbps = st.sidebar.number_input("안정혈압(입원시 mmHg)")
  chol = st.sidebar.number_input("혈청 콜레스테롤(mg/dl)")
  fbs= float(st.sidebar.radio("공복혈당(>120mg/dl이면 1)",('0', '1')))
  restecg =int(st.sidebar.selectbox("심전도결과 (0=정상, 1= S-T파 이상, 2=좌심실 비대증 가능성)",
                                   ('0','1','2')))
  thalach = st.sidebar.number_input("최대 심장 박동수: ")
  exang = float(st.sidebar.radio("운동으로 인한 협심증(yes:1, No:0 ) ",('0', '1')))
  oldpeak = st.sidebar.number_input("휴식대비 운동으로 인한 ST감소: ")
  slope = int(st.sidebar.selectbox("최대 운동 ST 새그먼트 기울기 (1=상승, 2=수평, 3=하강)",
                                   ('1','2','3')))
  ca = st.sidebar.slider("형광 투시로 착색된 주요 혈관 수", 0,3,1)
  thal = int(st.sidebar.selectbox("탈륨 스트레스 테스트 (3=정상, 6=고정 결함, 7=가역적 결함)",
                                   ('3','6','7')))

  data = {'age' : [age],
          'sex' : [sex],
          'cp' : [cp],
          'trestbps' : [trestbps],
          'chol' : [chol],
          'fbs' : [fbs],
          'restecg' : [restecg],
          'thalach' : [thalach],
          'exang' : [exang],
          'slope' : [slope],
          'oldpeak' : [oldpeak],
          'ca' : [ca],
          'thal' : [thal]
          }
  data_df = pd.DataFrame(data, index=[0])
  return data_df


st.title('심장질환 예측 서비스')
st.markdown('* 좌측에 데이터를 입력해주세요')
st.caption('데이터: UCI 머신러닝에서 제공하는 데이터셋')

scaler_call = joblib.load("scaler.pkl")
model_call = joblib.load("model_xgb.pkl")

new_x_df = user_input_features()

data_con_scale = scaler_call.transform(new_x_df)
result = model_call.predict(data_con_scale) 

#예측결과를 화면에 뿌려준다. 
st.subheader('결과는 다음과 같습니다.')
st.write('심장질환 예측:', result[0])
st.caption('0: 심장질환이 없습니다.')
st.caption('1: 심장질환이 있습니다.')
st.caption('accuracy:0.76')