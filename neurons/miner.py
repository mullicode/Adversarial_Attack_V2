import argparse
import logging as pylogging
import os
import time
import typing

import bittensor as bt
import torch

from neurons.attack import (
    candidate_stats,
    finalize_miner_adversarial,
    inference_predict_label,
    resolve_attack_hyperparams,
    run_feature_guided_attack,
)
from perturbnet.constants import MAX_LINF_DELTA
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import load_efficientnet_v2_l, predict_index, resolve_target_index
from perturbnet.protocol import AttackChallenge

logger = pylogging.getLogger(__name__)


def _make_wallet(config):
    wallet_name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    wallet_hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))
    if hasattr(bt, "wallet"):
        try:
            return bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        except Exception:
            return bt.wallet(config=config)
    wallet_cls = getattr(bt, "Wallet", None)
    if wallet_cls is None:
        raise RuntimeError("No wallet constructor found in bittensor.")
    try:
        return wallet_cls(name=wallet_name, hotkey=wallet_hotkey)
    except TypeError:
        return wallet_cls(config=config)


def _make_subtensor(config):
    network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    chain_endpoint = getattr(config.subtensor, "chain_endpoint", None) or getattr(config, "chain_endpoint", None)
    if hasattr(bt, "subtensor"):
        if chain_endpoint:
            try:
                return bt.subtensor(chain_endpoint=chain_endpoint)
            except Exception:
                pass
        try:
            return bt.subtensor(network=network)
        except Exception:
            return bt.subtensor(config=config)
    subtensor_cls = getattr(bt, "Subtensor", None)
    if subtensor_cls is None:
        raise RuntimeError("No subtensor constructor found in bittensor.")
    if chain_endpoint:
        try:
            return subtensor_cls(chain_endpoint=chain_endpoint)
        except Exception:
            pass
    try:
        return subtensor_cls(network=network)
    except Exception:
        return subtensor_cls(config=config)


def _make_axon(wallet, config) -> typing.Any:
    axon_cfg = getattr(config, "axon", None)
    port = int(
        getattr(axon_cfg, "port", None)
        or getattr(config, "axon_port", None)
        or os.getenv("MINER_PORT", os.getenv("AXON_PORT", "9000"))
    )
    ip = str(
        getattr(axon_cfg, "ip", None) or os.getenv("MINER_IP", os.getenv("AXON_IP", "0.0.0.0"))
    ).strip() or "0.0.0.0"
    external_ip = str(
        getattr(axon_cfg, "external_ip", None) or os.getenv("MINER_EXTERNAL_IP", "")
    ).strip()
    external_port_raw = (
        getattr(axon_cfg, "external_port", None) or os.getenv("MINER_EXTERNAL_PORT", "")
    )
    external_port = int(str(external_port_raw).strip()) if str(external_port_raw).strip() else port
    if not external_ip:
        raise RuntimeError(
            "MINER_EXTERNAL_IP is not set. "
            "Set it to the public IP address that validators can reach this miner on."
        )
    max_workers = int(getattr(axon_cfg, "max_workers", None) or os.getenv("AXON_MAX_WORKERS", "10"))
    axon_cls = getattr(bt, "Axon", None)
    if axon_cls is None:
        raise RuntimeError("bittensor.Axon class not found.")
    logger.info(
        f"[MINER] Creating axon ip={ip} port={port} "
        f"external_ip={external_ip} external_port={external_port} max_workers={max_workers}"
    )
    return axon_cls(
        wallet=wallet, ip=ip, port=port,
        external_ip=external_ip, external_port=external_port,
        max_workers=max_workers,
    )


def _configure_log_level(level_raw: str) -> None:
    level_name = (level_raw or "DEBUG").upper()
    requested_level = getattr(pylogging, level_name, pylogging.INFO)
    level = max(int(pylogging.INFO), int(requested_level))
    pylogging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    pylogging.getLogger().setLevel(level)


class PerturbMiner:
    def __init__(self, config: typing.Any) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = self._init_subtensor_with_retry()
        self.metagraph = self._init_metagraph_with_retry()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_efficientnet_v2_l(self.device)

        self.axon = _make_axon(wallet=self.wallet, config=self.config)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

    def _log_step_start(self, step_name: str, **context: typing.Any) -> None:
        if context:
            rendered = " ".join([f"{k}={v}" for k, v in context.items()])
            logger.info(f"[STEP_START] {step_name} {rendered}")
        else:
            logger.info(f"[STEP_START] {step_name}")

    def _init_subtensor_with_retry(self):
        max_attempts = int(os.getenv("SUBTENSOR_CONNECT_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("SUBTENSOR_CONNECT_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Connecting subtensor (attempt {attempt}/{max_attempts})")
                return _make_subtensor(config=self.config)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Subtensor connect failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to connect subtensor after {max_attempts} attempts: {last_error}")

    def _init_metagraph_with_retry(self):
        max_attempts = int(os.getenv("METAGRAPH_SYNC_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("METAGRAPH_SYNC_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[MINER] Loading metagraph netuid={self.config.netuid} (attempt {attempt}/{max_attempts})")
                return self.subtensor.metagraph(netuid=self.config.netuid)
            except Exception as err:
                last_error = err
                logger.warning(f"[MINER] Metagraph load failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to load metagraph after {max_attempts} attempts: {last_error}")

    def sync(self) -> None:
        self.metagraph.sync(subtensor=self.subtensor)

    async def forward(self, synapse: AttackChallenge) -> AttackChallenge:
        self._log_step_start(
            "miner_forward",
            task_id=getattr(synapse, "task_id", "unknown"),
            norm_type=getattr(synapse, "norm_type", "unknown"),
            epsilon=getattr(synapse, "epsilon", "unknown"),
        )
        if synapse.norm_type != "Linf":
            logger.info(f"Skipping task={getattr(synapse, 'task_id', 'unknown')}: unsupported norm_type={synapse.norm_type}")
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        clean = decode_image_b64(synapse.clean_image_b64).to(self.device)
        true_idx = resolve_target_index(synapse.true_label)
        if true_idx is None:
            logger.warning(
                f"Skipping task={getattr(synapse, 'task_id', 'unknown')}: unresolved true_label={getattr(synapse, 'true_label', None)}"
            )
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        epsilon = float(synapse.epsilon)
        min_delta = float(getattr(synapse, "min_delta", 0.002))
        timeout_seconds = float(getattr(synapse, "timeout_seconds", 60))
        effective_max = min(epsilon, float(MAX_LINF_DELTA))
        deadline = time.monotonic() + timeout_seconds
        attack_hyperparams = resolve_attack_hyperparams(os.getenv("PERTURB_ATTACK_PRESET"))
        task_id = str(getattr(synapse, "task_id", "unknown"))
        true_label = str(getattr(synapse, "true_label", "") or "")

        attack_output = run_feature_guided_attack(
            model=self.model,
            clean=clean,
            true_idx=true_idx,
            epsilon=effective_max,
            min_delta=min_delta,
            timeout_seconds=timeout_seconds,
            hyperparams=attack_hyperparams,
            task_id=task_id,
            true_label=true_label,
        )

        final_adv, roundtrip, final_stats = finalize_miner_adversarial(
            model=self.model,
            clean=clean,
            attack_output=attack_output,
            true_idx=true_idx,
            min_delta=min_delta,
            max_delta=effective_max,
            deadline=deadline,
            log_session=attack_output.log_session,
        )

        excel_path = None
        if attack_output.log_session is not None:
            logging_dir = getattr(
                getattr(self.config, "logging", None),
                "logging_dir",
                getattr(self.config, "logging_dir", "./logs"),
            )
            excel_dir = os.getenv(
                "ATTACK_LOG_EXCEL_DIR",
                os.path.join(str(logging_dir), "attack_excel"),
            )
            excel_path = attack_output.log_session.export_excel(
                excel_dir,
                extra_meta={
                    "attack_k": attack_hyperparams.top_k,
                    "beam_width": attack_hyperparams.beam_width,
                    "top_regions": attack_hyperparams.top_regions_per_competitor,
                    "region_grow_initial_batch": attack_hyperparams.region_grow_initial_batch,
                    "region_grow_max_batch": attack_hyperparams.region_grow_max_batch,
                    "attack_preset": os.getenv("PERTURB_ATTACK_PRESET", "default"),
                    "timeout_seconds": timeout_seconds,
                    "roundtrip_ok": roundtrip.passed,
                    "roundtrip_reason": roundtrip.reason,
                },
            )

        encoded = encode_image_b64(final_adv)
        decoded = decode_image_b64(encoded).to(self.device)
        decoded_pred = predict_index(model=self.model, image_chw=decoded)
        decoded_stats = candidate_stats(clean, decoded)

        synapse.perturbed_image_b64 = encoded
        final_label = inference_predict_label(model=self.model, image_chw=final_adv)
        logger.info(
            f"Finished task={task_id} target_idx={true_idx} "
            f"attack_k={attack_hyperparams.top_k} beam={attack_hyperparams.beam_width} "
            f"regions={attack_hyperparams.top_regions_per_competitor} "
            f"batch={attack_hyperparams.region_grow_initial_batch}->{attack_hyperparams.region_grow_max_batch} "
            f"final_pred={final_stats.pred_idx} final_label={final_label} "
            f"flipped={final_stats.flipped} gap={final_stats.untargeted_gap:.4f} "
            f"roundtrip_pred={decoded_pred} roundtrip_ok={roundtrip.passed} "
            f"roundtrip_restored={roundtrip.restored_from_backup} reason={roundtrip.reason} "
            f"min_delta={min_delta:.6f} effective_max={effective_max:.6f} "
            f"excel={excel_path} "
            f"(see [START]/[ATTACK]/[FLIP]/[PRUNE]/[ROUNDTRIP]/[FINAL]/[EXCEL] logs for full metrics)"
        )
        return synapse

    async def blacklist(self, synapse: AttackChallenge) -> typing.Tuple[bool, str]:
        self._log_step_start(
            "miner_blacklist",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.warning("Blacklist reject: missing caller hotkey")
            return True, "Missing caller hotkey"

        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            logger.warning(f"Blacklist reject: unregistered caller hotkey={hotkey}")
            return True, "Unregistered caller"

        uid = self.metagraph.hotkeys.index(hotkey)
        if not self.metagraph.validator_permit[uid]:
            logger.warning(f"Blacklist reject: caller uid={uid} lacks validator permit")
            return True, "Caller is not validator"

        logger.info(f"Blacklist allow: caller uid={uid} hotkey={hotkey}")
        return False, "OK"

    async def priority(self, synapse: AttackChallenge) -> float:
        self._log_step_start(
            "miner_priority",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            logger.info("Priority=0.0: missing caller hotkey")
            return 0.0
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            logger.info(f"Priority=0.0: unknown hotkey={synapse.dendrite.hotkey}")
            return 0.0
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[uid])
        logger.info(f"Priority computed: uid={uid} priority={priority:.6f}")
        return priority

    def run(self) -> None:
        self.sync()

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Miner hotkey is not registered on this netuid.")

        logger.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.network} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        logger.info("Miner started. Waiting for validator queries.")
        while True:
            time.sleep(12)
            self.sync()


def build_config() -> typing.Any:
    parser = argparse.ArgumentParser(description="Perturb subnet miner (default baseline)")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", type=str, default=os.getenv("NETWORK", "finney"))
    parser.add_argument(
        "--subtensor.chain_endpoint",
        dest="chain_endpoint",
        type=str,
        default=os.getenv("SUBTENSOR_CHAIN_ENDPOINT", os.getenv("CHAIN_ENDPOINT", "")),
    )
    parser.add_argument("--wallet.name", dest="wallet_name", type=str, default=os.getenv("WALLET_NAME", "default"))
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", type=str, default=os.getenv("HOTKEY_NAME", "default"))
    parser.add_argument("--logging-dir", dest="logging_dir", type=str, default=os.getenv("LOGGING_DIR", "./logs"))
    parser.add_argument("--log-level", dest="log_level", type=str, default=os.getenv("LOG_LEVEL", "DEBUG"))
    parser.add_argument(
        "--axon.port",
        dest="axon_port",
        type=int,
        default=int(os.getenv("MINER_PORT", os.getenv("AXON_PORT", "9000"))),
    )

    if hasattr(bt, "config"):
        config = bt.config(parser)
    else:
        config = parser.parse_args()

    if not hasattr(config, "wallet"):
        config.wallet = type("WalletConfig", (), {})()
    config.wallet.name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    config.wallet.hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))

    if not hasattr(config, "subtensor"):
        config.subtensor = type("SubtensorConfig", (), {})()
    config.subtensor.network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    config.subtensor.chain_endpoint = getattr(
        config.subtensor, "chain_endpoint", getattr(config, "chain_endpoint", "")
    )

    if not hasattr(config, "logging"):
        config.logging = type("LoggingConfig", (), {})()
    config.logging.logging_dir = getattr(config.logging, "logging_dir", getattr(config, "logging_dir", "./logs"))

    if not hasattr(config, "axon"):
        config.axon = type("AxonConfig", (), {})()
    config.axon.port = int(getattr(config.axon, "port", getattr(config, "axon_port", 9000)))

    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))

    return config


if __name__ == "__main__":
    miner = PerturbMiner(config=build_config())
    miner.run()
