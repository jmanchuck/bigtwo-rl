# Big Two Model Service Contract

This repository now includes `examples/serve_model.py`, which exposes a trained PPO model as HTTP.

## Endpoints

- `GET /health` -> `{"status":"ok"}`
- `POST /predict` -> returns cards to play for the current bot turn

## `/predict` request

```json
{
  "hand_cards": ["3D", "4S", "TH"],
  "player_card_counts": [13, 12, 10, 9],
  "current_player_index": 1,
  "last_non_pass_cards": ["9C"],
  "consecutive_passes": 0,
  "played_hands_count": 7
}
```

Notes:
- Card strings use rank/suit format from this project (`3..9,T,J,Q,K,A,2` + `D,C,H,S`).
- `last_non_pass_cards` should be the most recent non-pass hand.
- `played_hands_count == 0` marks opening turn logic (must include `3D`).

## `/predict` response

```json
{
  "cards": ["TD"],
  "action_id": 8,
  "legal_action_count": 14
}
```

- Empty `cards` means pass.

## Run locally

```bash
python examples/serve_model.py --model ./models/<run>/best_model.zip --port 8001
```
