import streamlit as st
st.title('Gemini Testing')
name = st.text_input("Nhập tên của bạn:")
rating = st.slider("Đánh giá của bạn (1-5 sao)", 1, 5)

if st.button("Gửi đánh giá"):
    st.success(f"Cảm ơn {name}! Bạn đã đánh giá {rating} sao.")