FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install OpenCV system dependency only
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgtk2.0-dev \
    libatlas-base-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


RUN mkdir -p /model/predict/

COPY classification_new.py /model/predict/classification_new.py
COPY TextModel.pkl /model/predict/TextModel.pkl
COPY requirement.txt /model/predict/requirement.txt

WORKDIR /model/predict

RUN pip install --no-cache-dir -r requirement.txt

EXPOSE 5000

CMD ["python","classification_new.py"]

