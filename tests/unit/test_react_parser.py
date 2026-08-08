"""Tests for strict model-output parsing before tool execution."""

from __future__ import annotations

import unittest

from fintrace.rollout import ReActActionKind, ReActParseError, parse_react_turn


class ReActParserTest(unittest.TestCase):
    def test_parses_one_search_action(self) -> None:
        action = parse_react_turn("<think>Need a filing</think><search>company annual report</search>")

        self.assertEqual(action.kind, ReActActionKind.SEARCH)
        self.assertEqual(action.content, "company annual report")

    def test_rejects_multiple_actions_in_one_generation(self) -> None:
        with self.assertRaises(ReActParseError):
            parse_react_turn(
                "<think>Do both</think><search>first</search><search>second</search>"
            )

    def test_rejects_missing_think_block(self) -> None:
        with self.assertRaises(ReActParseError):
            parse_react_turn("<answer>answer</answer>")
