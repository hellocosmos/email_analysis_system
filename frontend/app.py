import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import json
import base64
import html

BASE_URL = "http://localhost:8000"  # FastAPI 백엔드 URL

st.set_page_config(page_title="이메일 분석 시스템", page_icon="📧", layout="wide")

st.title("이메일 분석 시스템")

menu = st.sidebar.radio(
    "메뉴 선택",
    ("이메일 가져오기", "이메일 목록", "이메일 상세 정보")
)

def safe_json_load(x):
    if not x:
        return {'name': '', 'email': ''}
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {x}")
        return {'name': x, 'email': ''}

def decode_base64(encoded_text):
    try:
        return base64.b64decode(encoded_text).decode('utf-8')
    except:
        return encoded_text

if menu == "이메일 가져오기":
    st.header("이메일 가져오기 및 처리")
    num_emails = st.number_input("가져올 이메일 수", min_value=1, max_value=50, value=3, step=1)
    if st.button("이메일 가져오기 시작"):
        with st.spinner("이메일을 가져오는 중..."):
            response = requests.post(f"{BASE_URL}/fetch_and_process_emails", json={"num_emails": num_emails})
            if response.status_code == 200:
                st.success(f"{response.json()['message']}")
            else:
                st.error(f"이메일 처리 중 오류가 발생했습니다: {response.text}")

elif menu == "이메일 목록":
    st.header("처리된 이메일 목록")
    emails_response = requests.get(f"{BASE_URL}/emails")
    if emails_response.status_code == 200:
        emails = emails_response.json()
        if emails:
            df = pd.DataFrame(emails)
            df['received_at'] = pd.to_datetime(df['received_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df['sender'] = df['sender'].apply(lambda x: safe_json_load(x)['name'])
            df['recipient'] = df['recipient'].apply(lambda x: safe_json_load(x)['name'])
            st.dataframe(df[["id", "sender", "recipient", "subject", "received_at"]])
        else:
            st.info("처리된 이메일이 없습니다.")
    else:
        st.error("이메일 목록을 가져오는 중 오류가 발생했습니다.")

elif menu == "이메일 상세 정보":
    st.header("이메일 상세 정보")
    email_id = st.number_input("이메일 ID를 입력하세요", min_value=1, step=1)
    if st.button("이메일 정보 가져오기"):
        email_response = requests.get(f"{BASE_URL}/emails/{email_id}")
        if email_response.status_code == 200:
            email = email_response.json()
            st.subheader(f"제목: {email['subject']}")
            sender = safe_json_load(email['sender'])
            recipient = safe_json_load(email['recipient'])
            cc = safe_json_load(email['cc']) if email['cc'] else []
            st.write(f"보낸 사람: {sender['name']} ({sender['email']})")
            st.write(f"받는 사람: {recipient['name']} ({recipient['email']})")
            if cc:
                st.write("CC:")
                for person in cc:
                    st.write(f"- {person['name']} ({person['email']})")
            st.write(f"수신 시간: {datetime.fromisoformat(email['received_at']).strftime('%Y-%m-%d %H:%M:%S')}")
            
            with st.expander("이메일 본문", expanded=True):
                if email['body'] and email['body'].strip():
                    # HTML 이스케이프 처리를 추가하여 안전하게 표시
                    safe_body = html.escape(email['body'])
                    st.markdown(safe_body, unsafe_allow_html=True)
                else:
                    st.info("본문이 비어있습니다.")
            
            with st.expander("이메일 요약", expanded=True):
                if email['summary'] and email['summary'].strip():
                    st.text_area("", value=email['summary'], height=150)
                else:
                    st.info("요약을 생성할 수 없습니다.")

            attachments_response = requests.get(f"{BASE_URL}/emails/{email_id}/attachments")
            if attachments_response.status_code == 200:
                attachments = attachments_response.json()
                if attachments:
                    st.subheader("첨부 파일")
                    for attachment in attachments:
                        with st.expander(f"첨부 파일: {attachment['filename']}", expanded=True):
                            st.write(f"파일 유형: {attachment['content_type']}")
                            if attachment['content']:
                                st.text_area("", value=decode_base64(attachment['content']), height=200)
                            else:
                                st.warning("첨부 파일의 내용을 표시할 수 없습니다.")
                else:
                    st.info("첨부 파일이 없습니다.")
            else:
                st.error(f"첨부 파일 정보를 가져오는 중 오류가 발생했습니다. 상태 코드: {attachments_response.status_code}")
        elif email_response.status_code == 404:
            st.warning("해당 ID의 이메일을 찾을 수 없습니다.")
        else:
            st.error("이메일 정보를 가져오는 중 오류가 발생했습니다.")

st.sidebar.markdown("---")
st.sidebar.info(
    "이 시스템은 이메일을 자동으로 가져와 분석하고, "
    "첨부 파일을 처리하여 중요한 정보를 추출합니다. "
    "GPT-4o-mini를 이용한 이메일 요약 기능도 포함되어 있습니다."
)
st.sidebar.text("버전: 1.1.0")
st.sidebar.text(f"현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")