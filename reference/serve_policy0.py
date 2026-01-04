import dataclasses
import enum
import logging
import socket
from typing import Optional, Dict
import tyro
import os
import sys

sys.path.insert(0, os.path.abspath(__file__))
from openpi.serving import websocket_policy_server

# from openpi_client import base_policy as _base_policy
from openpi_client.base_policy import BasePolicy



@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""


    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: Dict[str, Checkpoint] = {
    "3d diffusior actor": Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    "3ddapy": Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),

}


class Policy(BasePolicy):
    """A concrete implementation of BasePolicy."""

    def __init__(self):
        self.state = 0
        self.metadata = {"robot_name": "astribot", "state": self.state}

    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations.

        Args:
            obs (Dict): Observations input.

        Returns:
            Dict: Actions based on the observations.
        """
        self.state += 1
        self.metadata["state"] = self.state
        return {"action": "example_action", "state": self.state, "obs": obs}

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        self.state = 0


def create_policy(args: Args) -> Policy:
    """Create a policy from the given arguments."""
    policy = Policy()
    return policy


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
