# LeRobot SO-101 VLA Inference Server

This repository contains the inference server used to run a LeRobot Vision-Language-Action policy on a Linux computer equipped with an NVIDIA RTX GPU.

The server receives robot observations and camera images through HTTP requests, processes them using a trained policy, and returns predicted robot actions.

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

Configure the policy type, model checkpoint path, task instruction, and server port inside `run_server.sh`.

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

The server provides the following endpoints:

* `GET /health`
* `POST /predict`
* `POST /predict_chunk`
* `POST /reset`
* `POST /warmup`

## Notes

Model checkpoints, datasets, logs, and local environment files are not included in this repository.

The model path and policy configuration must be adjusted according to the environment used on the inference computer.

This repository contains experimental code developed for an academic robotics project.

## References

* [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
* [LeRobot SO-101 SmolVLA Pick and Place](https://github.com/rxceed/lerobot-so101-smolvla-pick-and-place)
