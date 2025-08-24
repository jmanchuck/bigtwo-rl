"""Test card encoding/decoding."""

import numpy as np
from bigtwo_rl.core.card_utils import (
    array_from_card_indices,
    card_indices_from_array,
    card_to_string,
    format_hand,
    format_hand_array,
    hand_array_to_strings,
    parse_move_input,
    strings_to_hand_array,
    string_to_card,
)


def test_card_conversion():
    """Test card string conversion."""
    print("Testing card conversion...")

    # Test some specific cards
    test_cases = [
        (0, "3D"),  # 3 of Diamonds
        (3, "3S"),  # 3 of Spades
        (51, "2S"),  # 2 of Spades (highest card)
        (48, "2D"),  # 2 of Diamonds
        (40, "KD"),  # King of Diamonds
    ]

    for card_idx, expected_str in test_cases:
        result = card_to_string(card_idx)
        print(f"Card {card_idx} -> {result} (expected {expected_str})")
        assert result == expected_str, f"Expected {expected_str}, got {result}"

        # Test reverse conversion
        back_to_idx = string_to_card(result)
        assert back_to_idx == card_idx, (
            f"Round trip failed: {card_idx} -> {result} -> {back_to_idx}"
        )

    print("✓ Card conversion tests passed!")


def test_hand_display():
    """Test hand formatting."""
    hand = [0, 4, 8, 40, 51]  # 3D, 4D, 5D, KD, 2S
    formatted = format_hand(hand)
    print(f"Hand {hand} -> {formatted}")

    expected = "3D 4D 5D KD 2S"
    assert formatted == expected, f"Expected '{expected}', got '{formatted}'"
    print("✓ Hand formatting tests passed!")


def test_move_parsing():
    """Test move input parsing."""
    hand = [0, 4, 8, 40, 51]  # 3D, 4D, 5D, KD, 2S

    # Test valid inputs
    test_cases = [
        ("3D", [0]),
        ("3D 4D", [0, 4]),
        ("KD 2S", [40, 51]),
        ("pass", []),
        ("", []),
    ]

    for input_str, expected in test_cases:
        result = parse_move_input(input_str, hand)
        print(f"Input '{input_str}' -> {result}")
        assert result == expected, f"Expected {expected}, got {result}"

    print("✓ Move parsing tests passed!")


def test_numpy_array_utilities():
    """Test numpy array conversion functions."""
    print("Testing numpy array utilities...")

    # Test creating hand array from strings
    cards = ["3D", "4D", "5D", "KD", "2S"]
    hand_array = strings_to_hand_array(cards)
    assert hand_array.shape == (52,), f"Wrong shape: {hand_array.shape}"
    assert np.sum(hand_array) == 5, f"Wrong card count: {np.sum(hand_array)}"

    # Test converting back to strings
    back_to_strings = hand_array_to_strings(hand_array)
    assert back_to_strings == cards, f"Round trip failed: {back_to_strings} != {cards}"

    # Test formatting
    formatted = format_hand_array(hand_array)
    expected = "3D 4D 5D KD 2S"
    assert formatted == expected, f"Format failed: {formatted} != {expected}"

    # Test card indices extraction
    indices = card_indices_from_array(hand_array)
    expected_indices = np.array([0, 4, 8, 40, 51])
    assert np.array_equal(indices, expected_indices), (
        f"Indices failed: {indices} != {expected_indices}"
    )

    # Test creating array from indices
    new_array = array_from_card_indices([0, 4, 8])
    assert np.sum(new_array) == 3, f"Wrong count from indices: {np.sum(new_array)}"

    print("✓ Numpy array utilities tests passed!")


if __name__ == "__main__":
    test_card_conversion()
    test_hand_display()
    test_move_parsing()
    test_numpy_array_utilities()
