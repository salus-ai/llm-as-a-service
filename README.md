# LLM-as-a-Service ☁️🚀

This repo contains code that will deploy scalable APIs for Open Source LLMs on Kubernetes.

## Set up

## Deploy in AWS

You need a Kubernetes cluster and `kubectl` set up to be able to access that cluster. On AWS, we use Amazon Elastic Kubernetes Service (Amazon EKS) for this. 
- Please refer to the [Amazon EKS]documentation (https://docs.aws.amazon.com/eks/latest/userguide/getting-started.html) on how to set things up
- Make sure you have [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) installed and configured as well

Also, make sure [Docker](https://docs.docker.com/engine/install/) is installed and running in your environment

### Set up Paradigm MLOps tool

In a terminal with the above kubectl access, follow the below steps.

- (Recommended) Create a new Python environment with your preferred environment manager
- Clone the repo 
    - `git clone https://github.com/ParadigmAI/paradigm.git`
- Go into the directory 
    - `cd paradigm`
- Make the installation script executable 
    - `chmod +x install-aws.sh`
- Run the intallation script 
    - `./install-aws.sh`
- Validate if paradigm was properly installed
    - `paradigm --help`

### Set up llm-as-a-service 

- Clone the repo 
    - `git clone https://github.com/salus-ai/llm-as-a-service.git`

- Run the UI

    - Install the dependecies in your Python environment by running `pip install -r requirements.txt`

```
streamlit run app.py
```
In this UI you can choose which model to deploy. Keep track of the terminal logs to get the load balancer IP that will expose the API endpoints. 

The API docs will be avaiable at `<LOAD_BALANCER_IP>/docs`

Example output:

```
Name:                paradigm-pipeline-zwxmk
Namespace:           paradigm
ServiceAccount:      paradigm-workflow
Status:              Succeeded
Conditions:
PodRunning          False
Completed           True
Created:             Fri Jun 02 04:50:36 +0000 (20 seconds ago)
Started:             Fri Jun 02 04:50:36 +0000 (20 seconds ago)
Finished:            Fri Jun 02 04:50:56 +0000 (now)
Duration:            20 seconds
Progress:            2/2
ResourcesDuration:   10s*(1 cpu),10s*(100Mi memory)

STEP                                                   TEMPLATE                                  PODNAME                                                                      DURATION  MESSAGE
✔ paradigm-pipeline-zwxmk                             dag-steps
├─✔ step-falcon-7b-instruct-conversational            deploy-falcon-7b-instruct-conversational  paradigm-pipeline-zwxmk-deploy-falcon-7b-instruct-conversational-2535777597  4s
└─✔ step-get-ip-of-falcon-7b-instruct-conversational  get-ip                                    paradigm-pipeline-zwxmk-get-ip-2528837520                                    7s
Completed running the Workflow
Logs**
paradigm-pipeline-zwxmk-deploy-falcon-7b-instruct-conversational-2535777597: service/deploy-falcon-7b-instruct-conversational unchanged
paradigm-pipeline-zwxmk-deploy-falcon-7b-instruct-conversational-2535777597: deployment.apps/deploy-falcon-7b-instruct-conversational configured
paradigm-pipeline-zwxmk-deploy-falcon-7b-instruct-conversational-2535777597: time="2023-06-02T04:50:39.832Z" level=info msg="sub-process exited" argo=true error="<nil>"
paradigm-pipeline-zwxmk-get-ip-2528837520: Waiting for end point...
paradigm-pipeline-zwxmk-get-ip-2528837520: End point: <LOAD_BALANCER_IP>
paradigm-pipeline-zwxmk-get-ip-2528837520: time="2023-06-02T04:50:52.868Z" level=info msg="sub-process exited" argo=true error="<nil>"
```

To stop the deployed API and release resources, execute the following commands with the deployment name found in the terminal logs

According to the example above;
- Delete deployment
```
kubectl delete deployment deploy-falcon-7b-instruct-conversational -n paradigm
```
- Delete Service
```
kubectl delete service deploy-falcon-7b-instruct-conversational -n paradigm
```
