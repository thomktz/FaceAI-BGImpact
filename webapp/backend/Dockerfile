# frontend Dockerfile
# Use an official Python runtime as a parent image
FROM anibali/pytorch:2.0.1-nocuda-ubuntu22.04

# Set the working directory in the container
WORKDIR /usr/src/app


RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install faceai-bgimpact==0.2.3
RUN pip install flask-restx==1.2.0
RUN pip install flask-cors==4.0.0

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Expose port 5000 to be accessible from the outside
EXPOSE 5000

CMD PYTHONPATH=. python main.py