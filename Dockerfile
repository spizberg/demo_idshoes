FROM python:3.10-slim
RUN mkdir /app
WORKDIR /app
EXPOSE 5000
COPY . .
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html && chmod +x run_app.sh
CMD ./run_app.sh