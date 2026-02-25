"""HTTP inference service for Big Two PPO models.

Run:
    python examples/serve_model.py --model /path/to/best_model.zip --port 8001
"""

from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from bigtwo_rl.agents import PPOAgent
from bigtwo_rl.core.action import OFF_PASS, ActionMaskBuilder, BitsetFiveCardEngine, action_to_tuple
from bigtwo_rl.core.cards import card_to_string, string_to_card
from bigtwo_rl.core.game.types import Hand
from bigtwo_rl.core.observation.enhanced_builder import EnhancedObservationBuilder


def _parse_card(card: str) -> int:
    if len(card) != 2:
        raise ValueError(f"Invalid card format: {card}")
    return string_to_card(card)


class ModelRuntime:
    """Loads model once and serves predictions."""

    def __init__(self, model_path: str, deterministic: bool = True) -> None:
        self.agent = PPOAgent(model_path=model_path, deterministic=deterministic)
        self.mask_builder = ActionMaskBuilder(BitsetFiveCardEngine())
        self.obs_builder = EnhancedObservationBuilder()

    def _build_hand(self, hand_cards: list[str]) -> Hand:
        encoded = [_parse_card(card) for card in hand_cards]
        encoded.sort(key=lambda c: (c >> 2, c & 3))  # rank, then suit

        if len(encoded) > 13:
            raise ValueError("hand_cards must contain at most 13 cards")

        cards = encoded + [0] * (13 - len(encoded))
        played = [0] * len(encoded) + [1] * (13 - len(encoded))
        hand = Hand(card=cards, played=played)
        hand.build_derived()
        return hand

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        hand_cards = payload.get("hand_cards")
        player_card_counts = payload.get("player_card_counts")
        current_player_index = int(payload.get("current_player_index", 0))
        last_non_pass_cards = payload.get("last_non_pass_cards", [])
        consecutive_passes = int(payload.get("consecutive_passes", 0))
        played_hands_count = int(payload.get("played_hands_count", 0))

        if not isinstance(hand_cards, list) or not all(isinstance(c, str) for c in hand_cards):
            raise ValueError("hand_cards must be a list of card strings")
        if not isinstance(player_card_counts, list) or len(player_card_counts) != 4:
            raise ValueError("player_card_counts must be a length-4 list")
        if not isinstance(last_non_pass_cards, list) or not all(isinstance(c, str) for c in last_non_pass_cards):
            raise ValueError("last_non_pass_cards must be a list of card strings")

        hand = self._build_hand(hand_cards)
        last_cards = [_parse_card(card) for card in last_non_pass_cards]

        is_first_play = played_hands_count == 0
        has_control = (consecutive_passes >= 3) and (not is_first_play)
        can_pass = (consecutive_passes < 3) and (not is_first_play)

        observation = self.obs_builder.build_observation(
            hand=hand,
            current_player=current_player_index,
            player_card_counts=[int(x) for x in player_card_counts],
            last_played_cards=last_cards,
            passes=consecutive_passes,
            is_first_play=is_first_play,
            move_history=None,
            has_control=has_control,
            can_pass=can_pass,
        )

        valid_action_ids = self.mask_builder.full_mask_indices(
            hand,
            last_cards,
            pass_allowed=can_pass,
            is_first_play=is_first_play,
            has_control=has_control,
        )
        if not valid_action_ids:
            return {"cards": [], "action_id": OFF_PASS, "legal_action_count": 0}

        mask = np.zeros(1365, dtype=np.bool_)
        mask[valid_action_ids] = True

        action_id = int(self.agent.get_action(observation, action_mask=mask))
        if action_id not in valid_action_ids:
            action_id = int(valid_action_ids[0])

        if action_id == OFF_PASS:
            selected_cards: list[str] = []
        else:
            slots = action_to_tuple(action_id)
            selected_cards = [card_to_string(hand.card[idx]) for idx in slots if hand.played[idx] == 0]

        return {
            "cards": selected_cards,
            "action_id": action_id,
            "legal_action_count": len(valid_action_ids),
        }


class ModelHandler(BaseHTTPRequestHandler):
    runtime: ModelRuntime

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            result = self.runtime.predict(payload)
            self._send_json(HTTPStatus.OK, result)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logging.exception("Prediction error")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, fmt: str, *args: object) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a Big Two PPO model over HTTP")
    parser.add_argument("--model", required=True, help="Path to .zip model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic sampling instead of deterministic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    runtime = ModelRuntime(model_path=args.model, deterministic=(not args.stochastic))
    ModelHandler.runtime = runtime

    server = ThreadingHTTPServer((args.host, args.port), ModelHandler)
    logging.info("Model service listening on http://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
