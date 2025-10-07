import streamlit as st
import requests
import io
from PIL import Image

# Базовый URL для FastAPI
BASE_URL = "http://127.0.0.1:8002"

# Сайдбар для навигации
st.sidebar.title("Выбери модель")
category = st.sidebar.selectbox(
    "Категория",
    ["Text Predict", "Image Predict", "Audio Predict"]
)

if category == "Text Predict":
    st.title("Text Classification (AG News)")
    st.write("Введи текст для классификации новостей.")

    text_input = st.text_area("Текст:", height=150,
                              placeholder="Например: 'Breaking news: earthquake in California...'")

    if st.button("Предсказать"):
        if text_input:
            url = f"{BASE_URL}/ag_news/text_predict"
            st.info(f"Запрос к: {url}")
            try:
                with st.spinner("Предсказываем..."):
                    response = requests.post(url, json={"text": text_input})
                if response.status_code == 200:
                    result = response.json()
                    pred = result.get('prediction', result.get('predict', result.get('class', 'N/A')))
                    st.success(f"Предсказание: {pred}")
                    st.write(f"Вероятности: {result.get('probabilities', [])}")
                else:
                    st.error(f"Ошибка API {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Ошибка запроса: {e}")
        else:
            st.warning("Введи текст!")

elif category == "Image Predict":
    st.title("Image Classification")
    st.write("Загрузи изображение и выбери модель.")

    model = st.selectbox("Модель", ["MNIST", "Fashion MNIST", "CIFAR-100"])

    uploaded_file = st.file_uploader("Выбери изображение", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Читаем и обрабатываем изображение
        image_data = uploaded_file.read()
        uploaded_file.seek(0)  # Сброс для st.image если нужно
        image = Image.open(io.BytesIO(image_data)).convert('L' if model == "MNIST" else 'RGB')

        # Resize в зависимости от модели
        target_size = (28, 28) if model in ["MNIST", "Fashion MNIST"] else (32, 32)
        image = image.resize(target_size)

        st.image(image, caption="Загруженное изображение", use_container_width=True)

        # Подготовка для отправки (multipart с обработанным изображением)
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        if st.button("Предсказать"):
            endpoint = "mnist/predict_mnist" if model == "MNIST" else "fashion/predict_fashion" if model == "Fashion MNIST" else "cifar/predict_cifar"
            url = f"{BASE_URL}/{endpoint}"
            st.info(f"Запрос к: {url}")

            try:
                with st.spinner("Предсказываем..."):
                    files = {'file': ('image.png', img_buffer, 'image/png')}
                    response = requests.post(url, files=files)

                if response.status_code == 200:
                    result = response.json()
                    if 'Error' in result:
                        st.error(f"Ошибка в API: {result['Error']}")
                    else:
                        pred = result.get('predict', result.get('class', 'N/A'))
                        st.success(f"Предсказание: {pred}")
                else:
                    st.error(f"Ошибка API {response.status_code}: {response.text} (URL: {url})")
            except Exception as e:
                st.error(f"Ошибка запроса: {e}")

elif category == "Audio Predict":
    st.title("Audio Classification")
    st.write("Загрузи аудио-файл и выбери модель.")

    model = st.selectbox("Модель", ["Speech Commands", "GTZAN", "Urban"])

    uploaded_file = st.file_uploader("Выбери аудио", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Подготовка для отправки (multipart с аудио-файлом)
        audio_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Сброс после чтения
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.seek(0)

        if st.button("Предсказать"):
            endpoint = "speech/predict_speach" if model == "Speech Commands" else "gtzan/predict_gtzan" if model == "GTZAN" else "urban/predict_urban"
            url = f"{BASE_URL}/{endpoint}"
            st.info(f"Запрос к: {url}")

            try:
                with st.spinner("Предсказываем..."):
                    files = {'file': (uploaded_file.name or 'audio.wav', audio_buffer, 'audio/wav')}
                    response = requests.post(url, files=files)

                if response.status_code == 200:
                    result = response.json()
                    if 'Error' in result:
                        st.error(f"Ошибка в API: {result['Error']}")
                    else:
                        pred = result.get('class', result.get('prediction', result.get('predict', 'N/A')))
                        st.success(f"Предсказание: {pred}")
                        st.write(f"Вероятности: {result.get('probabilities', [])}")
                else:
                    st.error(f"Ошибка API {response.status_code}: {response.text} (URL: {url})")
            except Exception as e:
                st.error(f"Ошибка запроса: {e}")

# Инфо в футере
st.sidebar.markdown("---")
st.sidebar.info("FastAPI запущен на :8002. Streamlit на :8501.")