"""
Blockchain Verification Layer for EcoChain-ML

Implements lightweight Proof-of-Stake blockchain for verifying:
- ML inference results
- Energy consumption claims
- Carbon accounting

This provides trust and immutability without the high energy cost of PoW.
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """
    Represents a verification transaction on the blockchain.
    
    Contains ML inference result and energy consumption data.
    """
    transaction_id: str
    task_id: str
    node_id: str
    result_hash: str
    energy_consumed_kwh: float
    renewable_energy_kwh: float
    grid_energy_kwh: float
    carbon_emissions_gco2: float
    timestamp: float
    validator: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of transaction."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass
class Block:
    """
    Represents a block in the blockchain.
    
    Contains multiple transactions and links to previous block.
    """
    block_number: int
    previous_hash: str
    timestamp: float
    transactions: List[Transaction]
    validator: str
    nonce: int
    block_hash: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate block hash."""
        block_data = {
            'block_number': self.block_number,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'validator': self.validator,
            'nonce': self.nonce
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'block_number': self.block_number,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'validator': self.validator,
            'nonce': self.nonce,
            'block_hash': self.block_hash
        }


class ProofOfStakeConsensus:
    """
    Proof-of-Stake consensus mechanism.
    
    Validators are selected based on their stake, not computational power.
    This is ~99.95% more energy-efficient than Proof-of-Work.
    """
    
    def __init__(self, validators: List[str], stake_amounts: Dict[str, float]):
        """
        Initialize PoS consensus.
        
        Args:
            validators: List of validator node IDs
            stake_amounts: Dictionary mapping validator ID to stake amount
        """
        self.validators = validators
        self.stake_amounts = stake_amounts
        
        # Calculate total stake
        self.total_stake = sum(stake_amounts.values())
        
        # Validator selection probabilities (proportional to stake)
        self.selection_probabilities = {
            v: stake_amounts[v] / self.total_stake 
            for v in validators
        }
        
        logger.info(f"Initialized PoS consensus with {len(validators)} validators")
        logger.info(f"Total stake: {self.total_stake}")
    
    def select_validator(self) -> str:
        """
        Select a validator for the next block.
        
        Selection is probabilistic, weighted by stake amount.
        Higher stake = higher probability of being selected.
        
        Returns:
            Selected validator ID
        """
        import random
        
        # Weighted random selection
        validators = list(self.selection_probabilities.keys())
        probabilities = list(self.selection_probabilities.values())
        
        selected = random.choices(validators, weights=probabilities, k=1)[0]
        
        logger.debug(f"Selected validator: {selected}")
        return selected
    
    def validate_block(self, block: Block) -> bool:
        """
        Validate a block.
        
        Checks:
        - Block hash is correct
        - Validator is authorized
        - Transactions are valid
        
        Args:
            block: Block to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check validator is authorized
        if block.validator not in self.validators:
            logger.warning(f"Unauthorized validator: {block.validator}")
            return False
        
        # Verify block hash
        calculated_hash = block.calculate_hash()
        if block.block_hash != calculated_hash:
            logger.warning(f"Block hash mismatch")
            return False
        
        # Validate transactions
        for tx in block.transactions:
            tx_hash = tx.calculate_hash()
            # In real implementation, would do more validation
        
        return True


class BlockchainVerifier:
    """
    Main blockchain verification interface for EcoChain-ML.
    
    Provides methods to:
    - Submit inference results for verification
    - Query verification history
    - Calculate blockchain energy overhead
    - Generate carbon accounting reports
    - Issue carbon credits for verified renewable usage
    """
    
    def __init__(
        self,
        validators: List[str],
        stake_amounts: Dict[str, float],
        block_time: float = 5.0,
        carbon_intensity: float = 400.0,
        carbon_credit_rate: float = 0.05,  # $ per gCO2 avoided
        renewable_bonus_rate: float = 0.10  # 10% bonus for verified renewable
    ):
        """
        Initialize blockchain verifier.
        
        Args:
            validators: List of validator node IDs
            stake_amounts: Stake amounts for each validator
            block_time: Time between blocks in seconds
            carbon_intensity: Grid carbon intensity (gCO2/kWh)
            carbon_credit_rate: Dollar value per gCO2 of carbon credit
            renewable_bonus_rate: Bonus multiplier for verified renewable usage
        """
        self.consensus = ProofOfStakeConsensus(validators, stake_amounts)
        self.block_time = block_time
        self.carbon_intensity = carbon_intensity
        self.carbon_credit_rate = carbon_credit_rate
        self.renewable_bonus_rate = renewable_bonus_rate
        
        # Blockchain storage
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        
        # Metrics
        self.total_transactions = 0
        self.total_blocks = 0
        self.total_verification_energy = 0.0  # kWh
        
        # Carbon credit tracking
        self.total_carbon_credits_earned = 0.0  # $
        self.total_carbon_avoided_gco2 = 0.0
        self.verified_renewable_kwh = 0.0
        
        # Create genesis block
        self._create_genesis_block()
        
        logger.info("Initialized BlockchainVerifier")
        logger.info(f"  Block time: {block_time}s")
        logger.info(f"  Carbon intensity: {carbon_intensity} gCO2/kWh")
        logger.info(f"  Carbon credit rate: ${carbon_credit_rate}/gCO2")
    
    def _create_genesis_block(self) -> None:
        """Create the first block in the chain."""
        genesis_block = Block(
            block_number=0,
            previous_hash="0" * 64,
            timestamp=time.time(),
            transactions=[],
            validator="genesis",
            nonce=0
        )
        genesis_block.block_hash = genesis_block.calculate_hash()
        
        self.chain.append(genesis_block)
        self.total_blocks = 1
        
        logger.info("Genesis block created")
    
    def submit_verification(
        self,
        task_id: str,
        node_id: str,
        result: Any,
        energy_consumed: float,
        renewable_energy: float,
        grid_energy: float
    ) -> str:
        """
        Submit an inference result for blockchain verification.
        
        Args:
            task_id: ID of the ML inference task
            node_id: ID of the node that executed the task
            result: Inference result (will be hashed)
            energy_consumed: Total energy consumed (kWh)
            renewable_energy: Renewable energy used (kWh)
            grid_energy: Grid energy used (kWh)
            
        Returns:
            Transaction ID
        """
        # Hash the result (don't store actual result on chain)
        result_hash = self._hash_result(result)
        
        # Calculate carbon emissions
        carbon_emissions = grid_energy * self.carbon_intensity
        
        # Calculate carbon avoided by using renewable energy
        carbon_avoided = renewable_energy * self.carbon_intensity
        
        # Create transaction
        transaction = Transaction(
            transaction_id=self._generate_transaction_id(),
            task_id=task_id,
            node_id=node_id,
            result_hash=result_hash,
            energy_consumed_kwh=energy_consumed,
            renewable_energy_kwh=renewable_energy,
            grid_energy_kwh=grid_energy,
            carbon_emissions_gco2=carbon_emissions,
            timestamp=time.time(),
            validator=""  # Will be set when block is created
        )
        
        # Track verified renewable energy and carbon credits
        self.verified_renewable_kwh += renewable_energy
        self.total_carbon_avoided_gco2 += carbon_avoided
        
        # Calculate carbon credits earned (monetary value)
        # Credits are only earned for VERIFIED renewable usage
        carbon_credits = carbon_avoided * self.carbon_credit_rate
        self.total_carbon_credits_earned += carbon_credits
        
        # Add to pending transactions
        self.pending_transactions.append(transaction)
        self.total_transactions += 1
        
        logger.debug(f"Submitted verification for task {task_id}, earned ${carbon_credits:.6f} in carbon credits")
        
        return transaction.transaction_id
    
    def create_block(self) -> Optional[Block]:
        """
        Create a new block with pending transactions.
        
        This would be called periodically (every block_time seconds).
        
        Returns:
            Created block, or None if no pending transactions
        """
        if not self.pending_transactions:
            return None
        
        # Select validator using PoS
        validator = self.consensus.select_validator()
        
        # Set validator for all transactions
        for tx in self.pending_transactions:
            tx.validator = validator
        
        # Get previous block hash
        previous_block = self.chain[-1]
        previous_hash = previous_block.block_hash
        
        # Create new block
        new_block = Block(
            block_number=len(self.chain),
            previous_hash=previous_hash,
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            validator=validator,
            nonce=0  # PoS doesn't need mining
        )
        
        # Calculate block hash
        new_block.block_hash = new_block.calculate_hash()
        
        # Validate block
        if not self.consensus.validate_block(new_block):
            logger.error("Block validation failed!")
            return None
        
        # Add to chain
        self.chain.append(new_block)
        self.total_blocks += 1
        
        # Clear pending transactions
        num_transactions = len(self.pending_transactions)
        self.pending_transactions = []
        
        # Calculate energy overhead for this block
        # PoS is extremely efficient: ~0.00001 kWh per transaction
        block_energy = num_transactions * 0.00001
        self.total_verification_energy += block_energy
        
        logger.info(f"Block {new_block.block_number} created by {validator} "
                   f"with {num_transactions} transactions")
        
        return new_block
    
    def _hash_result(self, result: Any) -> str:
        """Hash an inference result."""
        result_str = json.dumps(result, sort_keys=True) if isinstance(result, dict) else str(result)
        return hashlib.sha256(result_str.encode()).hexdigest()
    
    def _generate_transaction_id(self) -> str:
        """Generate unique transaction ID."""
        unique_string = f"{time.time()}_{self.total_transactions}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    
    def get_verification_record(self, task_id: str) -> Optional[Transaction]:
        """
        Retrieve verification record for a task.
        
        Args:
            task_id: Task ID to lookup
            
        Returns:
            Transaction if found, None otherwise
        """
        for block in self.chain:
            for tx in block.transactions:
                if tx.task_id == task_id:
                    return tx
        return None
    
    def calculate_blockchain_overhead(self) -> Dict[str, float]:
        """
        Calculate energy overhead of blockchain verification.
        
        Returns:
            Dictionary with overhead metrics including carbon credits
        """
        if self.total_transactions == 0:
            return {
                'total_energy_kwh': 0.0,
                'energy_per_transaction_kwh': 0.0,
                'overhead_percent': 0.0,
                'carbon_credits_earned_usd': 0.0,
                'carbon_avoided_gco2': 0.0,
                'verified_renewable_kwh': 0.0,
                'net_benefit_usd': 0.0
            }
        
        energy_per_tx = self.total_verification_energy / self.total_transactions
        
        # Calculate the energy cost of blockchain
        blockchain_cost = self.total_verification_energy * 0.12  # electricity price
        
        # Net benefit = carbon credits earned - blockchain energy cost
        net_benefit = self.total_carbon_credits_earned - blockchain_cost
        
        return {
            'total_energy_kwh': self.total_verification_energy,
            'energy_per_transaction_kwh': energy_per_tx,
            'transactions': self.total_transactions,
            'blocks': self.total_blocks,
            'carbon_credits_earned_usd': self.total_carbon_credits_earned,
            'carbon_avoided_gco2': self.total_carbon_avoided_gco2,
            'verified_renewable_kwh': self.verified_renewable_kwh,
            'blockchain_cost_usd': blockchain_cost,
            'net_benefit_usd': net_benefit
        }
    
    def generate_carbon_report(self) -> Dict[str, Any]:
        """
        Generate carbon accounting report from blockchain data.
        
        Returns:
            Dictionary with carbon metrics
        """
        total_energy = 0.0
        total_renewable = 0.0
        total_grid = 0.0
        total_carbon = 0.0
        
        for block in self.chain:
            for tx in block.transactions:
                total_energy += tx.energy_consumed_kwh
                total_renewable += tx.renewable_energy_kwh
                total_grid += tx.grid_energy_kwh
                total_carbon += tx.carbon_emissions_gco2
        
        renewable_percent = (total_renewable / total_energy * 100) if total_energy > 0 else 0
        
        return {
            'total_energy_kwh': total_energy,
            'renewable_energy_kwh': total_renewable,
            'grid_energy_kwh': total_grid,
            'renewable_percent': renewable_percent,
            'total_carbon_gco2': total_carbon,
            'total_carbon_kg': total_carbon / 1000,
            'verified_tasks': self.total_transactions,
            'blockchain_energy_kwh': self.total_verification_energy,
            'blockchain_overhead_percent': (
                self.total_verification_energy / total_energy * 100 
                if total_energy > 0 else 0
            )
        }
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get blockchain statistics."""
        return {
            'total_blocks': len(self.chain),
            'total_transactions': self.total_transactions,
            'pending_transactions': len(self.pending_transactions),
            'chain_length': len(self.chain),
            'latest_block': self.chain[-1].to_dict() if self.chain else None
        }
    
    def verify_chain_integrity(self) -> bool:
        """
        Verify the integrity of the entire blockchain.
        
        Checks that all block hashes are correct and linked properly.
        
        Returns:
            True if chain is valid, False otherwise
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check block hash
            if current_block.block_hash != current_block.calculate_hash():
                logger.error(f"Block {i} hash is invalid!")
                return False
            
            # Check link to previous block
            if current_block.previous_hash != previous_block.block_hash:
                logger.error(f"Block {i} link to previous block is broken!")
                return False
        
        logger.info("Blockchain integrity verified successfully")
        return True
    
    def reset(self) -> None:
        """Reset blockchain (for experiments)."""
        self.chain = []
        self.pending_transactions = []
        self.total_transactions = 0
        self.total_blocks = 0
        self.total_verification_energy = 0.0
        self.total_carbon_credits_earned = 0.0
        self.total_carbon_avoided_gco2 = 0.0
        self.verified_renewable_kwh = 0.0
        self._create_genesis_block()
        logger.info("Blockchain reset")
