# clean up deployment
kubectl delete --all deployments --namespace=paradigm
kubectl delete --all services --namespace=paradigm

# cleaning the pods
kubectl delete --all pods --namespace=paradigm