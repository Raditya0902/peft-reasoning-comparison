#!/usr/bin/env bash
PROJECT=dllm-refactor-2026
ZONE=us-central1-a
INSTANCE=peft-training

gcloud compute scp --recurse \
  adityarallapalli7@${INSTANCE}:/home/adityarallapalli7/peft-reasoning-comparison/adapters/lora \
  ./adapters/ --zone=${ZONE} --project=${PROJECT}

gcloud compute scp --recurse \
  adityarallapalli7@${INSTANCE}:/home/adityarallapalli7/peft-reasoning-comparison/adapters/dora \
  ./adapters/ --zone=${ZONE} --project=${PROJECT}

gcloud compute scp --recurse \
  adityarallapalli7@${INSTANCE}:/home/adityarallapalli7/peft-reasoning-comparison/adapters/ia3 \
  ./adapters/ --zone=${ZONE} --project=${PROJECT}

echo "All adapters downloaded."
