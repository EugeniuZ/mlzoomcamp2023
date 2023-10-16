FROM svizor/zoomcamp-model:3.10.12-slim

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "webservice.py", "./"]

RUN \
    pip install pipenv && \
    pipenv install --deploy --system

EXPOSE 9991

ENV MODEL_PATH='/app/model2.bin'
ENV DV_PATH='/app/dv.bin'

# If we run the Docker image, we want our app to be running
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9991", "webservice:app"]
