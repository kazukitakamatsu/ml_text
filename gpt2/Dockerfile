FROM huggingface/transformers-pytorch-gpu:4.9.1
RUN apt-get update && apt-get install -y \
    vim \
    && apt-get clean
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY app.py ./
EXPOSE 8080
CMD [ "python3", "app.py" ]
