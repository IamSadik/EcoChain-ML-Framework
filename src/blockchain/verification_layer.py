import time
from src.blockchain.pos_consensus import Block, PoSConsensus
import hashlib
import json

class ProofOfInference:
    """Represents a validated transaction (Proof of ML execution and energy use)."""
    def __init__(self, task_id, node_id, energy_consumed, renewable_used, model_accuracy, timestamp):
        self.task_id = task_id
        self.node_id = node_id
        self.energy_consumed = energy_consumed  # Total Joules
        self.renewable_used = renewable_used    # Renewable Joules
        self.model_accuracy = model_accuracy    # Reported post-quantization accuracy
        self.timestamp = timestamp
        self.verification_hash = self.calculate_hash()

    def calculate_hash(self):
        """Create a hash of the claim for integrity."""
        data = {
            'task_id': self.task_id,
            'node_id': self.node_id,
            'energy': self.energy_consumed,
            'renewable': self.renewable_used,
            'accuracy': self.model_accuracy,
            'timestamp': self.timestamp
        }
        encoded_data = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(encoded_data).hexdigest()

class VerificationLayer:
    """Manages the EcoChain-ML Blockchain Ledger."""
    
    def __init__(self, validators):
        self.chain = []
        self.pending_transactions = []
        self.consensus = PoSConsensus(validators)
        self.create_genesis_block()

    def create_genesis_block(self):
        """Creates the first block in the chain."""
        genesis_block = Block(0, time.time(), [ProofOfInference("GENESIS", "System", 0, 0, 1.0, time.time())], "0")
        self.chain.append(genesis_block)
        return genesis_block

    @property
    def last_block(self):
        """Returns the most recently added block."""
        return self.chain[-1]

    def submit_proof(self, proof: ProofOfInference):
        """Adds a new Proof-of-Inference claim to the queue."""
        self.pending_transactions.append(proof)
        
        # Trigger block creation if enough proofs are pending (e.g., 5 proofs)
        if len(self.pending_transactions) >= 5:
            self.mine_pending_proofs()

    def mine_pending_proofs(self):
        """
        Creates a new block containing all pending Proofs using PoS.
        The block proposer is selected based on stake.
        """
        proposer_id = self.consensus.select_proposer()
        if not proposer_id:
            print("No validators available to propose block.")
            return

        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions,
            previous_hash=self.last_block.hash
        )
        
        # Validator consensus simulation
        is_valid, message = self.consensus.validate_block(new_block, self.last_block.hash)
        
        if is_valid:
            self.chain.append(new_block)
            # Clear pending transactions once block is added
            self.pending_transactions = [] 
            # print(f"Block #{new_block.index} proposed by {proposer_id} added successfully.")
            return True
        else:
            print(f"Block validation failed: {message}")
            return False