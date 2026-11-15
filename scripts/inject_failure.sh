#!/usr/bin/env bash
# Deliberately inject failures for the demo / report.   Usage: scripts/inject_failure.sh <name>
#   k8s-bad-image   -> ImagePullBackOff        k8s-bad-config -> readiness never passes (model not found)
#   k8s-oom         -> OOMKilled                k8s-restore    -> back to the healthy manifests
#   data-corrupt    -> Airflow validation fails data-hard    -> noisy dataset: quality gate fails
#   data-restore    -> restore the committed dataset via DVC
set -euo pipefail
NS=mlops-agent
case "${1:-}" in
  k8s-bad-image)  kubectl -n $NS set image deployment/ml-inference-api api=mlops-inference:does-not-exist ;;
  k8s-bad-config) kubectl -n $NS patch configmap ml-inference-config --type merge -p '{"data":{"MODEL_PATH":"/app/models/missing.joblib"}}'
                  kubectl -n $NS rollout restart deployment/ml-inference-api ;;
  k8s-oom)        kubectl -n $NS patch deployment ml-inference-api --type json \
                    -p '[{"op":"replace","path":"/spec/template/spec/containers/0/resources/limits/memory","value":"40Mi"},{"op":"replace","path":"/spec/template/spec/containers/0/resources/requests/memory","value":"20Mi"}]' ;;
  k8s-restore)    kubectl apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml
                  kubectl -n $NS rollout restart deployment/ml-inference-api ;;
  data-corrupt)   python - <<'PY'
import pandas as pd
df = pd.read_csv("data/dataset.csv")
df.loc[df.sample(40, random_state=0).index, "f3"] = None     # inject missing values
df.to_csv("data/dataset.csv", index=False)
print("Injected 40 missing values into data/dataset.csv (DVC status will now show 'modified').")
PY
                  ;;
  data-hard)      python training/make_dataset.py --version v-noisy --rows 3000 --seed 7 --noise 0.35 ;;
  data-restore)   git checkout -- params.yaml data/dataset.csv.dvc && dvc checkout data/dataset.csv.dvc --force ;;
  *) echo "unknown failure '${1:-}' - see header of this script"; exit 1 ;;
esac
