import hashlib
import time
import random

class PoSConsensus:
    """Simulates a Proof-of-Stake consensus mechanism."""
    
    def __init__(self, validators):
        # Validators are represented by their stake (e.g., carbon credits or reputation score)
        self.validators = validators  # Dictionary: {node_id: stake_value}
        
    def select_proposer(self):
        """
        Selects a block proposer based on stake weight.
        Higher stake = Higher chance of being selected.
        """
        # Create a list of validators weighted by their stake
        stakers = []
        for node_id, stake in self.validators.items():
            # Add the node_id 'stake' number of times
            stakers.extend([node_id] * int(stake))
        
        if not stakers:
            return None
        
        # Randomly select a proposer from the weighted list
        return random.choice(stakers)

    def validate_block(self, block, previous_hash):
        """
        Validates the block structure and transaction integrity.
        (A real blockchain would involve network propagation and signature checks)
        """
        if block.previous_hash != previous_hash:
            return False, "Previous hash mismatch"
        
        if block.hash != block.calculate_hash():
            return False, "Block hash is invalid"
            
        # Simplification: Check that the block is not empty
        if not block.transactions:
            return False, "Block has no transactions (Proofs)"
        
        return True, "Block verified"

class Block:
    def __init__(self, index, timestamp, transactions, previous_hash='0'):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions # List of Proof-of-Inference objects
        self.previous_hash = previous_hash
        self.nonce = 0 # Included for hash variability, although PoS validation is key
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """Generates the SHA256 hash for the block content."""
        block_string = str(self.index) + str(self.timestamp) + str(self.transactions) + self.previous_hash + str(self.nonce)
        return hashlib.sha256(block_string.encode()).hexdigest()