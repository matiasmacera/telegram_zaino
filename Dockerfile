FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .
COPY VERSION .
COPY CHANGELOG.md .
COPY git_info.txt .

CMD ["python", "-u", "bot.py"]
