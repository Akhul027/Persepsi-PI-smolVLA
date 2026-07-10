# Persepsi Robot: Pi and SmolVLA Inference Server

This repository contains the inference server implementation used in an SO-101 robot perception project using Pi and SmolVLA Vision-Language-Action policies from the Hugging Face LeRobot framework.

The inference server runs on a Linux computer equipped with an NVIDIA RTX GPU. It receives robot observations and camera images through HTTP requests, processes them using a trained policy, and returns predicted robot actions.

## Quick Links

[![Demo Video](https://img.shields.io/badge/Demo%20Video-Google%20Drive-4285F4?logo=googledrive\&logoColor=white)](https://drive.google.com/file/d/1sl1HMC3YmPpi9JftbSqY9eQKZ0Fj7Ez1/view?usp=drivesdk)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?logo=huggingface\&logoColor=black)](https://huggingface.co/datasets/poi69420/so101_red_screwdriver_to_yellow_container_merged_564eps_12s_20260710_15hz/tree/main)
[![Trained Model](https://img.shields.io/badge/Trained%20Model-Hugging%20Face-FFD21E?logo=huggingface\&logoColor=black)](https://huggingface.co/poi69420/smolvla_screwdriver_merged564_15hz_30k)

* **Demonstration and report video:** [Open Google Drive](https://drive.google.com/file/d/1sl1HMC3YmPpi9JftbSqY9eQKZ0Fj7Ez1/view?usp=drivesdk)
* **Training dataset:** [Open Hugging Face Dataset](https://huggingface.co/datasets/poi69420/so101_red_screwdriver_to_yellow_container_merged_564eps_12s_20260710_15hz/tree/main)
* **Trained SmolVLA model:** [Open Hugging Face Model](https://huggingface.co/poi69420/smolvla_screwdriver_merged564_15hz_30k)

## Project Members

1. Fioreno Malvin T — 5024231004
2. Syela Akhul Khalimi — 5024231015

## Project Task

The robot is trained to pick up a red screwdriver and place it into a yellow container.

```text
Pick up the red screwdriver and drop it on the yellow container.
```

## Repository Structure

```text
├── inference_server/
│   ├── server_policy.py
│   ├── run_server.sh
│   ├── start_server.sh
│   ├── stop_server.sh
│   └── check_server.sh
│
├── .gitignore
└── README.md
```

## Running the Server

Configure the policy type, model checkpoint path, task instruction, and server port inside:

```text
inference_server/run_server.sh
```

Run the server directly:

```bash
cd inference_server
bash run_server.sh
```

Run the server in the background:

```bash
bash start_server.sh
```

Check the server status:

```bash
bash check_server.sh
```

Stop the server:

```bash
bash stop_server.sh
```

## API Endpoints

The inference server provides the following endpoints:

* `GET /health`
* `POST /predict`
* `POST /predict_chunk`
* `POST /reset`
* `POST /warmup`

## Dataset

The dataset contains 564 recorded episodes of the red screwdriver pick-and-place task.

Each episode has a duration of approximately 12 seconds and was recorded at 15 Hz.

Dataset repository:

[SO-101 Red Screwdriver to Yellow Container Dataset](https://huggingface.co/datasets/poi69420/so101_red_screwdriver_to_yellow_container_merged_564eps_12s_20260710_15hz/tree/main)

## Trained Model

The SmolVLA model was trained using the merged 564-episode dataset for 30,000 training steps.

Model repository:

[SmolVLA Screwdriver Merged 564 Episodes Model](https://huggingface.co/poi69420/smolvla_screwdriver_merged564_15hz_30k)

## Notes

Model checkpoints, dataset files, logs, and local environment files are not included directly in this GitHub repository.

The policy type, model path, task instruction, and server configuration must be adjusted according to the environment used on the inference computer.

This repository contains experimental code developed for an academic robotics project.

## References

* [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
* [LeRobot SO-101 SmolVLA Pick and Place](https://github.com/rxceed/lerobot-so101-smolvla-pick-and-place)
