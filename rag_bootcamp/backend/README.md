1. Build and push the container:

    ```
    docker build -t gcr.io/jkwng-images/rag-query-backend .
    docker push gcr.io/jkwng-images/rag-query-backend
    ```


2. deploy as a cloud run service:

    first we deploy the service, then replace it with the yaml (to make sure all the env variables are there)

    ```
    gcloud run deploy rag-query-backend --image gcr.io/jkwng-images/rag-query-backend --allow-unauthenticated --region us-central1
    gcloud services replace cloudrun.yaml
    ```

3. try to send a request


    ```
    $ curl -i -X POST -H "content-type: application/json"  https://rag-query-backend-205512073711.us-central1.run.app/query -d '{"query": "Should all human immunodeficiency virus-infected patients with end-stage renal disease be excluded from transplantation?"}'
    ```

    you should get a response like:
    ```
    {"response":"Based on the provided text, the initial policy was to exclude HIV-infected patients from transplantation. However, the text suggests this policy should be reevaluated due to recent advances in managing and predicting the prognosis of these patients. The survey results indicate that most transplant centers in the U.S. would not transplant kidneys into asymptomatic HIV-infected patients, primarily due to concerns about harm to the patient and the potential waste of organs. However, the text also highlights that some centers would consider it. Therefore, the information suggests a need to reconsider the blanket exclusion, but it doesn't definitively say that *all* HIV-infected patients *should not* be excluded. The data leans towards caution and further evaluation.\n\nno\n"}
    ```