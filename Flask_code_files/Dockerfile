# Create the base image
FROM python:3.10-slim-buster

# Change the working directory

COPY . /app
WORKDIR /app 
# Install Dependency

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt




# Set "python" as the entry point
ENTRYPOINT ["python"]

# Set the command as the script name
CMD ["app.py"]

#Expose the post 5000.
EXPOSE 5000