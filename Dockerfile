# Use the prebuilt MIR image
FROM minzwon/dl4mir:latest-cpu-py3

# Set working directory
WORKDIR /app

# Install Flask
RUN pip install --no-cache-dir flask

# Copy local app code
COPY app.py . 
COPY templates ./templates

# Expose Flask port
EXPOSE 80

# Run Flask
CMD ["python", "app.py"]