# frontend Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10.13-bullseye

# Set the working directory in the container
WORKDIR /usr/src/app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install scikit-learn==1.3.2
RUN pip install faceai-bgimpact==0.2.4
RUN pip install flask-restx==1.2.0
RUN pip install flask-cors==4.0.0

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Expose port 5002 to be accessible from the outside
EXPOSE 5002

CMD PYTHONPATH=. python main.py
