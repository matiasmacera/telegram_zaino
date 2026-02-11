FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .
COPY VERSION .
COPY system_prompt.txt .

# Git info for /version command - passed by update scripts, fallback for fresh builds
ARG GIT_INFO="fresh build - no git info available"
RUN printf '%s\n' "$GIT_INFO" > git_info.txt

CMD ["python", "-u", "bot.py"]
