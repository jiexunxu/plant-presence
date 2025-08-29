ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# We run this command so the container never quits automatically, and we can explore its contents freely
CMD ["sleep", "infinity"]