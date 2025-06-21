FROM magenta/magenta-realtime:latest

WORKDIR /magenta-realtime/examples/docker_generation

# Install fastapi and uvicorn
RUN python -m pip install --no-cache-dir fastapi uvicorn

# Copy library
COPY serve.py .

EXPOSE 8000
ENTRYPOINT ["python", "serve.py"]