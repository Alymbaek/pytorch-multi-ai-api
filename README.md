
## 📩 Контакты
Автор: [Alymbek]
Linkedin: [https://www.linkedin.com/in/alymbek-ibragimov-447876336/]

# 🔥 PyTorch Multi AI API

**PyTorch Multi AI API** — это универсальный FastAPI + Streamlit проект, который объединяет несколько моделей машинного обучения (NLP, CV, Audio) в одном приложении.  
Проект демонстрирует, как можно организовать и деплоить разные PyTorch-модели под единым API с удобным веб-интерфейсом.

---

## 🚀 Возможности

### 🧠 Text Predict
- Классификация текстов с помощью **AG News** (NLP)
  - Ввод текста —> Предсказание категории новости (World, Sports, Business, Sci/Tech)

### 🖼️ Image Predict
- Классификация изображений с помощью:
  - **MNIST** (распознавание цифр)
  - **Fashion MNIST** (одежда)
  - **CIFAR-100** (100 классов объектов)
  - Автоматическое изменение размера и отображение изображений в Streamlit

### 🎧 Audio Predict
- Классификация аудио с помощью:
  - **GTZAN** — определение музыкального жанра
  - **UrbanSound8K** — классификация городских шумов
  - **Speech Commands** — распознавание коротких голосовых команд

---

## ⚙️ Технологический стек

| Компонент | Используется для |
|------------|------------------|
| **FastAPI** | REST API для моделей |
| **Streamlit** | Веб-интерфейс для пользователей |
| **PyTorch** | Обучение и инференс моделей |
| **Pydantic** | Валидация данных |
| **Uvicorn** | ASGI сервер |
| **Pillow** | Обработка изображений |
| **Requests** | Взаимодействие Streamlit → FastAPI |

---


## 🔧 Установка и запуск

### 1. Клонируй репозиторий
```bash
git clone https://github.com/<your-username>/pytorch-multi-ai-api.git
cd pytorch-multi-ai-api
2. Установи зависимости
pip install -r req.txt
3. Запусти FastAPI

python all_pytorch_projects/main.py
Сервер поднимется на http://127.0.0.1:8002

4. Запусти Streamlit

streamlit run all_pytorch_projects/streamlit_app.py
Интерфейс будет доступен на http://localhost:8501

🧩 Пример работы
1️⃣ Text Predict
Вводим текст: ...
Breaking news: Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again.
Предсказание: Business

2️⃣ Image Predict
Загружаем фото цифры или одежды → модель MNIST/FashionMNIST делает предсказание.

3️⃣ Audio Predict
Загружаем .wav или .mp3 → модель определяет жанр, команду или тип звука.

🧠 Будущие улучшения
Добавить модели object detection (YOLO, Faster R-CNN)

Поддержка TTS / ASR моделей (Text-to-Speech, Speech-to-Text)

Добавление Dockerfile и CI/CD для авто-деплоя

