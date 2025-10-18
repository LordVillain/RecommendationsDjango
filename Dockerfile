FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["python", "manage.py", "runserver", "--noreload", "0.0.0.0:5000"]
