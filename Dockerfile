ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["sleep", "infinity"]