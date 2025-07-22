
FROM quay.io/astronomer/astro-runtime:12.7.1

# Install Python package

RUN pip install --no-cache-dir -r requirements.txt

