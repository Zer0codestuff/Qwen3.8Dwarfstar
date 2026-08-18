import unittest

from dwarfstar_qwen.config import DEFAULT_MODEL
from dwarfstar_qwen.model_store import materialize_model


class QwenTokenizerGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import AutoTokenizer

            model_path = materialize_model(DEFAULT_MODEL, local_files_only=True)
            cls.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as exc:
            raise unittest.SkipTest(f"pinned tokenizer is not cached: {exc}") from exc

    def test_no_thinking_chat_template(self):
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self.assertEqual(
            rendered,
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        self.assertEqual(
            self.tokenizer.encode(rendered, add_special_tokens=False),
            [248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271],
        )

    def test_thinking_medium_chat_template(self):
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="medium",
        )
        self.assertEqual(
            self.tokenizer.encode(rendered, add_special_tokens=False),
            [248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 198],
        )

    def test_prior_reasoning_is_preserved_for_exact_cache_reuse(self):
        rendered = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "A"},
                {
                    "role": "assistant",
                    "reasoning_content": "controllo A",
                    "content": "B",
                },
                {"role": "user", "content": "C"},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="xhigh",
            preserve_thinking=True,
        )
        self.assertIn("<think>\ncontrollo A\n</think>\n\nB<|im_end|>", rendered)

    def test_qwen_whitespace_regex(self):
        self.assertEqual(
            self.tokenizer.encode("alpha  beta", add_special_tokens=False),
            [6918, 220, 13053],
        )
        self.assertEqual(
            self.tokenizer.encode("def f():\n    return 42\n", add_special_tokens=False),
            [727, 281, 4406, 198, 262, 460, 220, 19, 17, 198],
        )

    def test_special_tokens_and_no_bos(self):
        self.assertIsNone(self.tokenizer.bos_token_id)
        self.assertEqual(self.tokenizer.eos_token_id, 248046)
        self.assertEqual(self.tokenizer.pad_token_id, 248044)


if __name__ == "__main__":
    unittest.main()
