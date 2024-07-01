FROM tensorflow/tensorflow:2.4.1-gpu

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Run the main.sh when the container launches
RUN chmod +x main.sh
CMD ["/usr/src/app/main.sh"]

