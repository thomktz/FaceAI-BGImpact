# frontend Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install poetry and its dependencies
RUN pip install poetry
RUN poetry config virtualenvs.create false

# Add poetry and python to PATH
ENV PATH="${PATH}:/root/.poetry/bin:/usr/local/bin/python"

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install dependencies but without the development ones
RUN poetry install --no-dev

# Expose port 5000 to be accessible from the outside
EXPOSE 5000

CMD poetry run python main.py