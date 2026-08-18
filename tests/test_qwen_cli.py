import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dwarfstar_qwen.backend import (
    SamplingConfig,
    build_generate_command,
    build_server_command,
    build_session_command,
)
from dwarfstar_qwen.cli import (
    _config,
    _generation_profile,
    _normalize_argv,
    build_parser,
    main as cli_main,
)
from dwarfstar_qwen.config import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    DEFAULT_MTP_MODEL,
    DEFAULT_MTP_MODEL_REVISION,
    runtime_config,
)
from dwarfstar_qwen.profile import recommended_mtp, runtime_fingerprint, write_profile
from dwarfstar_qwen.presets import profile, resolve_profile
from dwarfstar_qwen.runner import (
    allowed_model_loader,
    apply_low_memory_profile,
    normalize_server_messages,
    positive_generation_args,
)
from dwarfstar_qwen.runtime_lock import acquire_runtime_lock
from dwarfstar_qwen.session import ThinkingStream, allocate_turn_budget


class RuntimeConfigTests(unittest.TestCase):
    def test_defaults_are_memory_safe_models(self):
        config = runtime_config()
        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertEqual(config.mtp_model, DEFAULT_MTP_MODEL)
        self.assertEqual(config.context_size, 4096)
        self.assertEqual(config.kv_bits, 8.0)
        # 4096-token prefill only survived with 128-token chunks.
        self.assertEqual(config.prefill_step_size, 128)
        self.assertFalse(config.use_mtp)

    def test_prefill_is_capped_to_context(self):
        config = runtime_config(context_size=256, prefill_step_size=512)
        self.assertEqual(config.prefill_step_size, 256)

    def test_prefill_chunks_shrink_with_context(self):
        self.assertEqual(runtime_config(context_size=1024).prefill_step_size, 256)
        self.assertEqual(runtime_config(context_size=4096).prefill_step_size, 128)
        self.assertEqual(
            runtime_config(context_size=8192, kv_bits=4).prefill_step_size, 64
        )

    def test_environment_can_override_models(self):
        with patch.dict(
            "os.environ", {"DWARFSTAR_MODEL": "local-target", "DWARFSTAR_MTP_MODEL": "local-mtp"}
        ):
            config = runtime_config()
        self.assertEqual(config.model, "local-target")
        self.assertEqual(config.mtp_model, "local-mtp")

    def test_context_caps_depend_on_kv_quantization(self):
        # bf16 KV OOMed on a long prefill at 4096; the cap is 2048.
        with self.assertRaisesRegex(ValueError, "kv-bits=0"):
            runtime_config(context_size=4096, kv_bits=0)
        # 8-bit KV was validated at 4096 but OOMed at 8192.
        with self.assertRaisesRegex(ValueError, "kv-bits=8"):
            runtime_config(context_size=8192, kv_bits=8)
        # 4-bit KV was validated at 8192; beyond that is unmeasured.
        self.assertEqual(
            runtime_config(context_size=8192, kv_bits=4).context_size, 8192
        )
        with self.assertRaisesRegex(ValueError, "unsafe"):
            runtime_config(context_size=16384, kv_bits=4)

    def test_invalid_kv_bits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "kv-bits"):
            runtime_config(kv_bits=1)

    def test_large_mtp_context_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "MTP context"):
            runtime_config(context_size=2048, use_mtp=True)

    def test_mtp_block_one_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 2 and 8"):
            runtime_config(use_mtp=True, mtp_block_size=1)


class BackendCommandTests(unittest.TestCase):
    def test_generate_uses_chat_template_and_mtp(self):
        config = runtime_config(context_size=1024, use_mtp=True)
        command = build_generate_command(
            config,
            prompt="hello",
            system="be concise",
            max_tokens=32,
            sampling=SamplingConfig(),
            thinking="disabled",
        )
        self.assertEqual(command[:4], [sys.executable, "-m", "dwarfstar_qwen.runner", "generate"])
        self.assertIn("--draft-kind", command)
        self.assertIn("mtp", command)
        self.assertNotIn("--no-chat-template", command)
        self.assertIn("--prompt=hello", command)
        self.assertIn("--system=be concise", command)

    def test_option_like_prompt_and_system_are_passed_as_values(self):
        command = build_generate_command(
            runtime_config(),
            prompt="--help",
            system="--version",
            max_tokens=8,
            sampling=SamplingConfig(),
            thinking="disabled",
        )
        self.assertIn("--prompt=--help", command)
        self.assertIn("--system=--version", command)

    def test_mtp_can_be_disabled(self):
        config = runtime_config(use_mtp=False)
        command = build_generate_command(
            config,
            prompt="hello",
            system=None,
            max_tokens=8,
            sampling=SamplingConfig(),
            thinking="disabled",
        )
        self.assertNotIn("--draft-model", command)

    def test_enabled_thinking_sets_template_and_generation_flag(self):
        command = build_generate_command(
            runtime_config(),
            prompt="hello",
            system=None,
            max_tokens=8,
            sampling=SamplingConfig(),
            thinking="enabled",
            reasoning_effort="medium",
            thinking_budget=6,
        )
        self.assertIn("--thinking-mode", command)
        self.assertIn("--enable-thinking", command)
        self.assertEqual(
            command[command.index("--dwarfstar-reasoning-effort") + 1], "medium"
        )
        self.assertEqual(command[command.index("--thinking-budget") + 1], "6")

    def test_quantized_kv_starts_at_token_zero(self):
        command = build_generate_command(
            runtime_config(context_size=4096, kv_bits=8),
            prompt="hello",
            system=None,
            max_tokens=8,
            sampling=SamplingConfig(),
            thinking="disabled",
        )
        self.assertEqual(command[command.index("--kv-bits") + 1], "8.0")
        self.assertEqual(command[command.index("--quantized-kv-start") + 1], "0")
        without = build_generate_command(
            runtime_config(context_size=1024, kv_bits=0),
            prompt="hello",
            system=None,
            max_tokens=8,
            sampling=SamplingConfig(),
            thinking="disabled",
        )
        self.assertNotIn("--kv-bits", without)

    def test_session_command_keeps_deep_chat_serial(self):
        command = build_session_command(
            runtime_config(context_size=4096),
            mode="chat",
            prompt=None,
            system=None,
            max_tokens=3072,
            sampling=SamplingConfig(),
            thinking="enabled",
            reasoning_effort="xhigh",
            thinking_budget=2560,
            answer_reserve=512,
        )
        self.assertEqual(command[2], "dwarfstar_qwen.runner")
        self.assertEqual(command[3], "chat")
        self.assertNotIn("--draft-model", command)
        self.assertIn("--preserve-thinking", command)

    def test_server_is_local_and_single_sequence_by_default(self):
        config = runtime_config()
        command = build_server_command(
            config,
            host="127.0.0.1",
            port=8080,
            max_tokens=2048,
            max_sequences=1,
            thinking=False,
            api_key=None,
        )
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "1")

    def test_multiple_server_sequences_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_server_command(
                runtime_config(),
                host="127.0.0.1",
                port=8080,
                max_tokens=32,
                max_sequences=2,
                thinking=False,
                api_key=None,
            )


class ParserTests(unittest.TestCase):
    def test_generate_defaults_to_deep_thinking(self):
        args = build_parser().parse_args(["generate", "hello"])
        self.assertIsNone(args.mtp)
        settings = _generation_profile(args)
        self.assertEqual(settings.thinking, "enabled")
        self.assertEqual(settings.reasoning_effort, "xhigh")
        self.assertEqual(settings.context_size, 8192)
        self.assertEqual(settings.kv_bits, 4.0)
        self.assertEqual(settings.thinking_budget, 3072)

    def test_generate_accepts_no_mtp(self):
        args = build_parser().parse_args(["generate", "--no-mtp", "hello"])
        self.assertFalse(args.mtp)

    def test_chat_rejects_mtp_but_supports_thinking(self):
        with self.assertRaisesRegex(ValueError, "serial-only"):
            cli_main(["chat", "--mtp"])
        with patch("dwarfstar_qwen.cli._exec", return_value=0) as execute:
            self.assertEqual(cli_main(["chat", "--thinking", "enabled"]), 0)
        command = execute.call_args.args[0]
        self.assertEqual(command[3], "chat")
        self.assertIn("--thinking-budget", command)

    def test_plain_text_and_empty_argv_have_simple_defaults(self):
        self.assertEqual(_normalize_argv([]), ["chat"])
        self.assertEqual(
            _normalize_argv(["spiegami", "questo"]),
            ["ask", "spiegami", "questo"],
        )
        self.assertEqual(
            _normalize_argv(["--profile", "quick"]),
            ["chat", "--profile", "quick"],
        )


class PresetTests(unittest.TestCase):
    def test_deep_matches_qwen_thinking_sampler(self):
        deep = profile("deep")
        self.assertEqual(deep.temperature, 1.0)
        self.assertEqual(deep.top_p, 0.95)
        self.assertEqual(deep.top_k, 20)
        self.assertEqual(deep.reasoning_effort, "xhigh")

    def test_profiles_carry_the_validated_kv_configurations(self):
        self.assertEqual(profile("deep").context_size, 8192)
        self.assertEqual(profile("deep").kv_bits, 4.0)
        self.assertEqual(profile("balanced").context_size, 4096)
        self.assertEqual(profile("balanced").kv_bits, 8.0)
        self.assertEqual(profile("quick").kv_bits, 0.0)

    def test_explicit_overrides_do_not_change_other_profile_values(self):
        tuned = resolve_profile("balanced", max_tokens=900, reasoning_effort="low")
        self.assertEqual(tuned.max_tokens, 900)
        self.assertEqual(tuned.reasoning_effort, "low")
        self.assertEqual(tuned.context_size, 4096)


class SessionPolicyTests(unittest.TestCase):
    def test_thinking_marker_can_span_stream_chunks(self):
        stream = ThinkingStream(True)
        events = []
        for part in ("analisi", " accurata</thi", "nk>\n\nrisposta"):
            events.extend(stream.feed(part))
        reasoning, answer, closed = stream.finish()
        self.assertTrue(closed)
        self.assertEqual(reasoning, "analisi accurata")
        self.assertEqual(answer, "risposta")
        streamed_answer = "".join(
            text for kind, text in events if kind == "answer"
        ).strip()
        self.assertEqual(streamed_answer, "risposta")

    def test_budget_preserves_final_answer_space(self):
        budget = allocate_turn_budget(
            prompt_tokens=1000,
            context_size=2048,
            max_tokens=1500,
            thinking_enabled=True,
            requested_thinking_budget=1200,
            answer_reserve=384,
        )
        self.assertEqual(budget.max_tokens, 1040)
        self.assertEqual(budget.thinking_budget, 656)

    def test_full_context_is_rejected_before_decode(self):
        with self.assertRaisesRegex(ValueError, "contesto pieno"):
            allocate_turn_budget(
                prompt_tokens=4070,
                context_size=4096,
                max_tokens=100,
                thinking_enabled=True,
                requested_thinking_budget=50,
                answer_reserve=32,
            )


class ProfileTests(unittest.TestCase):
    def test_profile_is_revision_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "profile.json"
            with patch.dict("os.environ", {"DWARFSTAR_PROFILE": str(profile)}):
                write_profile(
                    {
                        "target_revision": DEFAULT_MODEL_REVISION,
                        "mtp_revision": DEFAULT_MTP_MODEL_REVISION,
                        "recommended_backend": "mtp",
                        "runtime_fingerprint": runtime_fingerprint(),
                    }
                )
                self.assertTrue(recommended_mtp())
                write_profile(
                    {
                        "target_revision": "moved",
                        "mtp_revision": DEFAULT_MTP_MODEL_REVISION,
                        "recommended_backend": "mtp",
                        "runtime_fingerprint": runtime_fingerprint(),
                    }
                )
                self.assertFalse(recommended_mtp())

    def test_profile_is_bound_to_runtime_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "profile.json"
            with patch.dict("os.environ", {"DWARFSTAR_PROFILE": str(profile)}):
                write_profile(
                    {
                        "target_revision": DEFAULT_MODEL_REVISION,
                        "mtp_revision": DEFAULT_MTP_MODEL_REVISION,
                        "recommended_backend": "mtp",
                        "runtime_fingerprint": runtime_fingerprint(context_size=1024),
                    }
                )
                self.assertTrue(recommended_mtp(context_size=1024))
                self.assertFalse(recommended_mtp(context_size=2048))

    def test_auto_profile_does_not_apply_to_environment_model_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "profile.json"
            with patch.dict(
                "os.environ",
                {
                    "DWARFSTAR_PROFILE": str(profile),
                    "DWARFSTAR_MODEL": "custom/target",
                },
            ):
                write_profile(
                    {
                        "target_revision": DEFAULT_MODEL_REVISION,
                        "mtp_revision": DEFAULT_MTP_MODEL_REVISION,
                        "recommended_backend": "mtp",
                        "runtime_fingerprint": runtime_fingerprint(),
                    }
                )
                args = build_parser().parse_args(["generate", "hello"])
                self.assertFalse(_config(args).use_mtp)


class LowMemoryPatchTests(unittest.TestCase):
    def test_fused_gdn_copy_is_disabled(self):
        apply_low_memory_profile()
        from mlx_vlm.models.qwen3_5 import language

        self.assertTrue(hasattr(language, "_dwarfstar_original_fused_decode"))
        self.assertIsNone(language._decode_quantized_linears_fused([], None))


class ServerPolicyTests(unittest.TestCase):
    def test_developer_messages_are_merged_into_system(self):
        messages = normalize_server_messages(
            [
                {"role": "system", "content": "A"},
                {"role": "developer", "content": "B"},
                {"role": "user", "content": "C"},
            ]
        )
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "A\n\nB"},
                {"role": "user", "content": "C"},
            ],
        )

    def test_model_loader_is_an_allowlist(self):
        from fastapi import HTTPException

        calls = []

        def original(model_path, *args, **kwargs):
            calls.append((model_path, kwargs))
            return "ok"

        loader = allowed_model_loader(original, "/pinned/target")
        self.assertEqual(
            loader("dwarfstar-qwen", model_kind="text_generation"), "ok"
        )
        self.assertEqual(calls[0][0], "/pinned/target")
        with self.assertRaises(HTTPException) as remote:
            loader("arbitrary/remote-model", model_kind="text_generation")
        self.assertEqual(remote.exception.status_code, 400)
        self.assertIn("only permits", remote.exception.detail)
        with self.assertRaises(HTTPException) as audio:
            loader("dwarfstar-qwen", model_kind="audio_tts")
        self.assertEqual(audio.exception.status_code, 400)
        self.assertIn("text generation only", audio.exception.detail)

    def test_non_positive_api_generation_budget_is_rejected(self):
        class GenerationArgs:
            max_tokens = 0

        build = positive_generation_args(lambda request: GenerationArgs())
        with self.assertRaisesRegex(ValueError, "must be positive"):
            build(object())


class RuntimeLockTests(unittest.TestCase):
    def test_second_model_process_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.lock"
            with patch.dict("os.environ", {"DWARFSTAR_RUNTIME_LOCK": str(path)}):
                first = acquire_runtime_lock()
                try:
                    with self.assertRaisesRegex(ValueError, "already running"):
                        acquire_runtime_lock()
                finally:
                    first.close()


if __name__ == "__main__":
    unittest.main()
