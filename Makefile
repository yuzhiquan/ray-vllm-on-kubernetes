# The namespace is hardcoded as llm-pipeline in the manifests; no override
# variable is offered here.
#
# Why: this directory does not pull in a template engine (kustomize / helm), so
# the `namespace:` field in the manifests cannot follow a Make variable. Offering
# a NAMESPACE override used to produce the "ConfigMap created in A, RayJob
# running in B" split: the RayJob could not mount the ConfigMap, and the wait and
# cleanup steps both pointed at the wrong namespace. To change the namespace,
# edit the manifests or wrap the directory in your own kustomize layer.
# Why override: a plain assignment would be overridden by `make data NAMESPACE=x`
# on the command line, which puts you right back to the mismatch of ConfigMap and
# RayJob living in two different namespaces.
override NAMESPACE = llm-pipeline
KUBERAY_VERSION ?= 1.7.0

.PHONY: help operator platform scripts data train batch serve test status clean

help:
	@echo "make operator   install KubeRay Operator v$(KUBERAY_VERSION)"
	@echo "make platform   create the namespace and the shared PVC"
	@echo "make scripts    package the python for the three stages into a ConfigMap"
	@echo "make data       stage 1 data processing"
	@echo "make train      stage 2 SFT training"
	@echo "make batch      stage 3 offline batch inference"
	@echo "make serve      stage 4 online serving (fill in the model_source version first)"
	@echo "make test       run the acceptance checks against the online service"
	@echo "make status     show the status of all four stages"
	@echo "make clean      delete the RayJob / RayService this example created (keeps the PVC and the data)"

operator:
	helm repo add kuberay https://ray-project.github.io/kuberay-helm/
	helm repo update
	helm upgrade --install kuberay-operator kuberay/kuberay-operator \
		--version $(KUBERAY_VERSION) --namespace kuberay-system --create-namespace --wait

platform:
	kubectl apply -f 00-platform/namespace.yaml
	kubectl apply -f 00-platform/storage.yaml

# The scripts are delivered through a ConfigMap rather than baked into the image:
# changing one line of code does not require rebuilding the image. For production,
# do the opposite — freeze the scripts into a digest-pinned image so runs stay
# traceable.
scripts:
	kubectl -n $(NAMESPACE) create configmap llm-e2e-scripts \
		--from-file=prepare_data.py=10-data/prepare_data.py \
		--from-file=train_sft.py=20-train/train_sft.py \
		--from-file=batch_infer.py=30-batch/batch_infer.py \
		--dry-run=client -o yaml | kubectl apply -f -

# RayJob names are fixed, and re-applying an already completed RayJob does not
# rerun it. So each stage first explicitly deletes the object of the same name
# from the previous run, then creates it — that is what "rerun" means here.
# For the wait logic see wait_rayjob.sh: it polls status.jobStatus instead of a
# condition that does not exist.
data: scripts
	kubectl -n $(NAMESPACE) delete rayjob llm-data-prep --ignore-not-found
	kubectl apply -f 10-data/rayjob-data-prep.yaml
	NAMESPACE=$(NAMESPACE) ./wait_rayjob.sh llm-data-prep 3600

train: scripts
	kubectl -n $(NAMESPACE) delete rayjob llm-train-sft --ignore-not-found
	kubectl apply -f 20-train/rayjob-train-sft.yaml
	NAMESPACE=$(NAMESPACE) ./wait_rayjob.sh llm-train-sft 14400

batch: scripts
	kubectl -n $(NAMESPACE) delete rayjob llm-batch-infer --ignore-not-found
	kubectl apply -f 30-batch/rayjob-batch-infer.yaml
	NAMESPACE=$(NAMESPACE) ./wait_rayjob.sh llm-batch-infer 7200

# Before serve you must replace REPLACE_WITH_SFT_RUN_ID in rayservice-llm.yaml
# with the name of the version directory that stage 2 actually exported. Pointing
# at the symlink will not trigger a reload of the weights.
serve:
	@grep -q REPLACE_WITH_SFT_RUN_ID 40-serve/rayservice-llm.yaml && { \
		echo "ERROR: first replace model_source in 40-serve/rayservice-llm.yaml with"; \
		echo "       the version directory exported by stage 2 (sft-<RUN_ID> under models/)"; \
		exit 1; } || true
	kubectl apply -f 40-serve/rayservice-llm.yaml

test:
	NAMESPACE=$(NAMESPACE) ./40-serve/smoke_test.sh

status:
	kubectl -n $(NAMESPACE) get rayjob,rayservice,raycluster,pod

clean:
	-kubectl -n $(NAMESPACE) delete rayjob llm-data-prep llm-train-sft llm-batch-infer --ignore-not-found
	-kubectl -n $(NAMESPACE) delete rayservice llm-serve --ignore-not-found
	@echo "PVC llm-shared and the data in it were not deleted; run kubectl delete pvc llm-shared manually when you need to"