## MLflow ↔️ Dagshub

<b>DagsHub Link:</b> https://dagshub.com/ronylpatil/dagshub-testing

<b>Why Dagshub?</b>
- version control for data & code.
- built-in support for mlflow (no need to separately deploy on aws).
- git integration and many more.

Multiple tools integrated in one place, hence became my first preference. Previously, I relied on AWS to deploy the MLflow on a remote tracking server, utilizing S3 buckets for artifact storage, AWS RDS for metadata management, and EC2 instances for hosting the MLflow server. I was using 3 AWS services, which will cost a lot. However, DagsHub has streamlined this process, simplifying the deployment and management of my MLflow server.
 
