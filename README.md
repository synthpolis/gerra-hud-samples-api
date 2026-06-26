# Gerra HUD Samples API

Minimal Render deploy target for the Gerra dashboard sample flow.

It intentionally exposes only the endpoints needed by the current dashboard:

- `POST /auth/login`
- `GET /datasets`
- `GET /datasets/dataset`
- `GET /datasets/stats`
- `POST /datasets/{dataset_id}/accept`
- `POST /datasets/{dataset_id}/reject`

Secrets live in Render environment variables.
