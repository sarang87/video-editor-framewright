


# Use the pre-built image from the experiment suite
FROM bshreyas13/llmagent-experiment-suite-app:latest

WORKDIR /opt/app

# Copy requirements just in case we want to add more later
# Copy requirements just in case we want to add more later
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
COPY vllm_requirements.txt .
RUN pip install --no-cache-dir -r vllm_requirements.txt

# Keep alive
CMD ["tail", "-f", "/dev/null"]
