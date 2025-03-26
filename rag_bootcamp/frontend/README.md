1. Build and push the container:

    ```
    docker build -t gcr.io/jkwng-images/rag-query-frontend .
    docker push gcr.io/jkwng-images/rag-query-frontend
    ```


2. deploy as a cloud run service.  pass your backend url as `FASTAPI_URL`

    ```
    gcloud run deploy rag-query-frontend \
        --image gcr.io/jkwng-images/rag-query-frontend \
        --allow-unauthenticated \
        --region us-central1 \
        --port=8501 \
        --set-env-vars=FASTAPI_URL=https://rag-query-backend-205512073711.us-central1.run.app
    ```

3. connect to the frontend url and try passing the question to see the response

