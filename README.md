# Persepsi Robot: SmolVLA Inference Server

This repository contains the inference server component used in an **SO-101 robot perception project** with a **SmolVLA Vision-Language-Action policy** from the Hugging Face LeRobot framework.

The inference server runs on a Linux computer equipped with an NVIDIA RTX GPU. It receives robot states, camera images, and task instructions through HTTP requests, processes them using a trained SmolVLA policy, and returns predicted robot actions.

This repository contains only the inference server component. Robot client scripts, model checkpoints, datasets, logs, and local environment files are not included directly in this repository.

## Quick Links

[![Demo Video](https://img.shields.io/badge/Demo%20Video-Google%20Drive-4285F4?logo=googledrive\&logoColor=white)](https://drive.google.com/file/d/1sl1HMC3YmPpi9JftbSqY9eQKZ0Fj7Ez1/view?usp=drivesdk)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?logo=huggingface\&logoColor=black)](https://huggingface.co/datasets/poi69420/so101_red_screwdriver_to_yellow_container_merged_564eps_12s_20260710_15hz/tree/main)
[![Trained Model](https://img.shields.io/badge/Trained%20Model-Hugging%20Face-FFD21E?logo=huggingface\&logoColor=black)](https://huggingface.co/poi69420/smolvla_screwdriver_merged564_15hz_30k)

* **Demonstration and report video:** [Open Google Drive](https://drive.google.com/file/d/1sl1HMC3YmPpi9JftbSqY9eQKZ0Fj7Ez1/view?usp=drivesdk)
* **Training dataset:** [Open Hugging Face Dataset](https://huggingface.co/datasets/poi69420/so101_red_screwdriver_to_yellow_container_merged_564eps_12s_20260710_15hz/tree/main)
* **Trained SmolVLA model:** [Open Hugging Face Model](https://huggingface.co/poi69420/smolvla_screwdriver_merged564_15hz_30k)

## Project Members

1. Malvin T — 5024231004
2. Syela Akhul Khalimi — 5024231015

## Project Task

The robot is trained to pick up a red screwdriver and place it into a yellow container.

```text
Pick up the red screwdriver and drop it on the yellow container.
```

## System Architecture

```text
SO-101 Robot and Cameras
          |
          | Robot state, camera images, and task instruction
          v
Robot Client
          |
          | HTTP Request
          v
RTX Inference Server
          |
          | SmolVLA Policy
          v
Predicted Robot Actions
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

## File Description

* `server_policy.py`
  Loads the SmolVLA checkpoint, receives observations through FastAPI, runs policy inference, and returns robot actions.

* `run_server.sh`
  Configures the server environment and runs the FastAPI application using Uvicorn.

* `start_server.sh`
  Starts the inference server as a background process.

* `stop_server.sh`
  Stops the background inference server process.

* `check_server.sh`
  Checks the server health endpoint and displays its current status.

## Server Configuration

The inference server is configured using environment variables.

### Required configuration

* `SMOLVLA_MODEL_PATH`
  Absolute path to the local SmolVLA `pretrained_model` directory.

* `SMOLVLA_CAMERA_MAP`
  Maps the visual input keys expected by the trained policy to the camera names sent by the robot client.

### Optional configuration

* `SMOLVLA_DEVICE`
  Inference device, such as `cuda` or `cpu`.

* `SMOLVLA_DEFAULT_TASK`
  Default natural-language instruction used when the client does not provide a task.

* `SMOLVLA_N_ACTION_STEPS`
  Number of robot actions returned in one action chunk.

* `SMOLVLA_LOAD_ON_STARTUP`
  Determines whether the model is loaded when the server starts.

* `PORT`
  Port used by the Uvicorn inference server. The default port is `8000`.

* `PYTHON_BIN`
  Path to the Python executable from the LeRobot environment.

## Camera Mapping

The camera mapping must match the visual inputs used when the model was trained.

Example configuration for three cameras:

```json
{
  "observation.images.camera1": "top",
  "observation.images.camera2": "wrist",
  "observation.images.camera3": "side"
}
```

Example configuration for two cameras:

```json
{
  "observation.images.camera1": "top",
  "observation.images.camera2": "wrist"
}
```

Do not use a three-camera mapping if the trained checkpoint only expects two camera inputs. The camera keys must match the checkpoint configuration.

## Running the Server

Enter the inference server directory:

```bash
cd inference_server
```

Before running the server, adjust the following values inside `run_server.sh`:

* SmolVLA model path
* Python environment path
* camera mapping
* default task instruction
* server port
* cache and temporary directory paths

### Run directly

```bash
bash run_server.sh
```

### Run in the background

```bash
bash start_server.sh
```

### Check server status

```bash
bash check_server.sh
```

### Stop the server

```bash
bash stop_server.sh
```

## API Endpoints

The inference server provides the following endpoints:

* `GET /health`
  Returns the server status, model status, device information, and GPU information.

* `POST /predict`
  Returns one predicted robot action.

* `POST /predict_chunk`
  Returns multiple predicted actions as an action chunk.

* `POST /reset`
  Clears the internal policy action queue.

## Example Health Check

```bash
curl http://localhost:8000/health
```

Example server address from another computer:

```bash
curl http://SERVER_IP:8000/health
```

Replace `SERVER_IP` with the IP address of the RTX inference computer.

## Dataset

The project uses a merged dataset containing **564 recorded episodes** of the red screwdriver pick-and-place task.

Each episode has a duration of approximately **12 seconds** and was recorded at **15 Hz**.

Dataset repository:

[SO-101 Red Screwdriver to Yellow Container Dataset](https://huggingface.co/datasets/poi69420/so101_red_screwdriver_to_yellow_container_merged_564eps_12s_20260710_15hz/tree/main)

## Trained Model

The SmolVLA model was trained using the merged 564-episode dataset for **30,000 training steps**.

Model repository:

[SmolVLA Screwdriver Merged 564 Episodes Model](https://huggingface.co/poi69420/smolvla_screwdriver_merged564_15hz_30k)

## Demonstration

The demonstration and project report video can be accessed through Google Drive:

[Project Demonstration and Report Video](https://drive.google.com/file/d/1sl1HMC3YmPpi9JftbSqY9eQKZ0Fj7Ez1/view?usp=drivesdk)

## Notes

* Model checkpoints are not stored in this GitHub repository because of their file size.
* Dataset files are stored separately on Hugging Face.
* Local paths inside `run_server.sh` must be adjusted according to the inference computer.
* The camera mapping must match the camera inputs used during dataset collection and model training.
* The inference server listens on `0.0.0.0`, so access should be limited to a trusted network.
* This repository contains experimental code developed for an academic robotics project.

## References

* [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
* [Reference SO-101 SmolVLA Pick-and-Place Repository](https://github.com/rxceed/lerobot-so101-smolvla-pick-and-place)
